#!/usr/bin/env python3

"""Main entry point for the AI English Learning Assistant."""

import sys
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# --- Path Correction --- 
# Add the project root directory (parent of 'ai_english_assistant') to the Python path
# This allows running the script directly from the root or subdirectory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Correction ---

# --- Logging Setup --- 
def setup_logging():
    log_dir = Path(project_root) / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "assistant.log"

    log_level = logging.INFO # Set default level
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Configure root logger
    logger = logging.getLogger() # Get root logger
    logger.setLevel(log_level)
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    # Avoid adding handler if already present (e.g., during testing or re-runs)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(console_handler)

    # File Handler (Rotating)
    # Rotate logs after 5MB, keep 3 backup logs
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_handler.setFormatter(log_format)
    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == str(log_file) for h in logger.handlers):
        logger.addHandler(file_handler)

    logging.getLogger("openai").setLevel(logging.WARNING) # Reduce verbosity from openai library
    logging.getLogger("httpx").setLevel(logging.WARNING) # Reduce verbosity from http library used by openai
    logging.getLogger("pyttsx3").setLevel(logging.WARNING) # Reduce verbosity from tts library

    return logging.getLogger(__name__) # Return logger for main module

logger = setup_logging() # Initialize logging
# --- End Logging Setup ---


from ai_english_assistant import Assistant
# from ai_english_assistant.config.settings import settings # Can import settings directly if needed

def run_console_app():
    """Runs a simple interactive console loop for the assistant."""
    logger.info("--- Initializing AI English Learning Assistant --- ")
    try:
        assistant = Assistant()
    except Exception as e:
        logger.critical(f"Fatal error during initialization: {e}", exc_info=True) # Log traceback
        print(f"Fatal error during initialization: {e}") # Also print critical error to console
        print("Please check your configuration (config.toml), environment variables (e.g., OPENAI_API_KEY), and logs/assistant.log.")
        sys.exit(1)
    logger.info("------------------------------------------------")

    if not assistant.llm_client:
        logger.warning("LLM Client could not be initialized. Core functionality may be limited.")
        
    # Determine if TTS should be used based on config and potential command-line args (future)
    use_tts_in_console = assistant.tts_synthesizer.enabled
    if use_tts_in_console:
        logger.info("TTS is enabled. Assistant responses will be spoken.")
    else:
        logger.info("TTS is disabled or failed to initialize.")

    print("\nStarting interactive session.") # Keep user-facing prints clean
    print("Type your message, or:")
    print("  'quit' - to exit the application.")
    print("  'new'  - to start a new conversation session.")
    print("  'tts on'/'tts off' - to toggle speech output for this session.")
    print("  'prompt <name>' - to switch to a prompt defined in config.toml.")
    print("------------------------------------------------")


    while True:
        try:
            user_text = input("You: ")
            logger.info(f"User input: {user_text}") # Log user input
        except EOFError:
            # Handle Ctrl+D or end of input stream gracefully
            print("\nExiting...")
            logger.info("EOF detected, exiting.")
            break
        except KeyboardInterrupt:
             # Handle Ctrl+C gracefully
             print("\nExiting...")
             logger.info("Keyboard interrupt detected, exiting.")
             break

        command = user_text.lower().strip()

        if command == 'quit':
            print("Assistant: Goodbye!")
            logger.info("User requested quit.")
            break
        elif command == 'new':
            assistant.start_new_session()
            response_text = "Okay, let's start a fresh conversation!"
            print(f"Assistant: {response_text}")
            if use_tts_in_console:
                 assistant.tts_synthesizer.speak(response_text)
            continue
        elif command == 'tts on':
            if assistant.tts_synthesizer.enabled:
                use_tts_in_console = True
                logger.info("TTS output toggled ON for this session.")
                print("Assistant: Okay, I will speak my responses.")
                assistant.tts_synthesizer.speak("TTS enabled.")
            else:
                logger.warning("User tried to enable TTS, but it's not available.")
                print("Assistant: Sorry, TTS is not available (check config or installation).")
            continue
        elif command == 'tts off':
             use_tts_in_console = False
             logger.info("TTS output toggled OFF for this session.")
             print("Assistant: Okay, I will only provide text responses.")
             continue
        elif command.startswith("prompt "):
            prompt_name = user_text[7:].strip()
            logger.info(f"User requested prompt switch to: {prompt_name}")
            if assistant.switch_prompt(prompt_name):
                response_text = f"Switched to prompt '{prompt_name}'. Let's start fresh."
                print(f"Assistant: {response_text}")
                if use_tts_in_console:
                    assistant.tts_synthesizer.speak(response_text)
            else:
                 # Error/warning already logged by switch_prompt
                 response_text = f"Sorry, I couldn't find or load a prompt named '{prompt_name}'."
                 print(f"Assistant: {response_text}")
            continue

        # If not a command, process as user input to the LLM
        if not assistant.llm_client or not assistant.llm_client.is_ready():
             logger.error("LLM client not ready, cannot process user input.")
             print("Assistant: Sorry, I can't process requests as the LLM client is unavailable.")
             continue

        # !! 修改: 处理流式响应 (Modification: Handle streaming response) !!
        print("Assistant: ", end='', flush=True) # Print prefix without newline
        full_response = ""
        try:
            stream_generator = assistant.get_response_stream(user_text)
            for chunk in stream_generator:
                 if not chunk.startswith("[ERROR:"):
                      print(chunk, end='', flush=True) # Print chunks as they arrive
                      full_response += chunk
                 else:
                      # Handle potential error message yielded by the generator
                      print(f"\n[Stream Error: {chunk[7:-1]}]") 
                      logger.error(f"Error received from stream: {chunk}")
                      full_response = None # Indicate error
                      break # Stop processing stream on error
            print() # Add newline after stream finishes or errors
        except RuntimeError as e:
             # Handle critical errors raised by get_response_stream (e.g., client not ready)
             print(f"\n[Critical Error: {e}]")
             logger.error(f"RuntimeError during get_response_stream: {e}", exc_info=True)
             full_response = None
        except Exception as e:
             # Catch any other unexpected errors during stream iteration
             print(f"\n[Unexpected Error: {e}]")
             logger.error(f"Unexpected error processing stream: {e}", exc_info=True)
             full_response = None
        
        # 语音合成现在需要完整的响应 (TTS now needs the full response)
        if use_tts_in_console and full_response:
             logger.info(f"Speaking full response (length: {len(full_response)})")
             assistant.tts_synthesizer.speak(full_response)
        elif use_tts_in_console and full_response is None:
            logger.warning("Skipping TTS because response generation failed.")

if __name__ == "__main__":
    run_console_app()
    logger.info("Application finished.") 