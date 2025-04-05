from typing import Dict, List, Optional, Any, Generator
from collections import deque
import logging

# Import necessary components
from .config.settings import settings
from .llm.openai_client import OpenAIClient
from .tts.speech_synthesizer import SpeechSynthesizer
from .prompts import PromptManager # Import the new manager

logger = logging.getLogger(__name__)

# Constants for default values if needed
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
DEFAULT_PROMPT_IDENTIFIER = "default_tutor" # Corresponds to key in config.toml [prompts]

class Assistant:
    """AI英语学习助手主类 (Main class for the AI English Learning Assistant)."""

    def __init__(self):
        """初始化助手，加载配置和组件 (Initializes the assistant, loading configuration and components)."""
        self.settings = settings
        logger.debug(f"Loaded settings: {self.settings}")
        
        # 初始化 LLM 客户端 (Initialize LLM Client)
        try:
            openai_config = self.settings.get('openai', {})
            if not openai_config.get('api_key'):
                 logger.warning("OpenAI API Key 未在配置或环境变量中找到。LLM 功能可能受限。")

            self.llm_client = OpenAIClient(
                api_key=openai_config.get('api_key'), # API Key (might be None initially if not set)
                model=openai_config.get('model'), # Model (should be validated by settings)
                base_url=openai_config.get('base_url'), # Optional Base URL
                default_temperature=openai_config.get('default_temperature', 0.7),
                default_max_tokens=openai_config.get('default_max_tokens', 150)
            )
        except (KeyError, ValueError) as e:
             logger.error(f"初始化 LLM 客户端失败: {e}。LLM 功能不可用。", exc_info=True)
             self.llm_client = None 

        # 初始化 TTS 合成器 (Initialize TTS Synthesizer)
        self.tts_synthesizer = SpeechSynthesizer(config=self.settings.get('tts', {}))

        # 初始化 Prompt 管理器 (Initialize Prompt Manager)
        self.prompt_manager = PromptManager(prompts_config=self.settings.get('prompts', {}))
        
        # 存储用户信息（用于模板渲染）(Store user info for templates)
        self.user_info = {
            "user_name": "User", # Default name
            "user_level": None, 
            "learning_goals": None,
        }
        
        # 当前使用的系统 Prompt 标识符 (Current system prompt identifier)
        self.current_prompt_identifier = DEFAULT_PROMPT_IDENTIFIER 
        # 加载初始的系统 Prompt (Load initial system prompt)
        self.system_prompt = None # Initialize system_prompt attribute
        self._update_system_prompt() # Call after user_info is defined

        # 会话管理 (Session Management)
        self.max_history = self.settings.get('session', {}).get('max_history', 10)
        self.session_history: deque[Dict[str, str]] = deque(maxlen=self.max_history)
        
        logger.info("AI 英语助手初始化完成 (AI English Assistant initialized).")
        if not self.llm_client or not self.llm_client.is_ready(): 
             logger.warning("LLM 客户端未能初始化或未配置API Key (Warning: LLM Client could not be initialized or API Key not configured).")
        if not self.tts_synthesizer.enabled:
             logger.warning("TTS 已禁用或初始化失败 (Warning: TTS is disabled or failed to initialize).")
        if not self.system_prompt or (self.system_prompt == DEFAULT_SYSTEM_PROMPT and self.current_prompt_identifier != DEFAULT_PROMPT_IDENTIFIER):
             logger.warning(f"初始 Prompt '{self.current_prompt_identifier}' 加载失败或回退到默认 (Warning: Initial prompt '{self.current_prompt_identifier}' failed to load or fell back to default).")


    def _update_system_prompt(self):
        """根据当前标识符和用户信息加载并渲染系统 Prompt (Loads and renders the system prompt based on the current identifier and user info)."""
        logger.debug(f"Updating system prompt. Current identifier: '{self.current_prompt_identifier}'")
        rendered_prompt = self.prompt_manager.render_prompt(
            self.current_prompt_identifier, 
            **self.user_info 
        )
        if rendered_prompt:
            self.system_prompt = rendered_prompt
            logger.info(f"System prompt '{self.current_prompt_identifier}' loaded and rendered.")
        else:
            logger.error(f"无法加载或渲染 Prompt '{self.current_prompt_identifier}'. 回退到默认 Prompt。")
            self.system_prompt = DEFAULT_SYSTEM_PROMPT

    def set_user_info(self, **kwargs: Any):
        """Updates user info and re-renders the current prompt."""
        updated = False
        for key, value in kwargs.items():
             if key in self.user_info:
                  if self.user_info[key] != value:
                       self.user_info[key] = value
                       updated = True
             else:
                  logger.warning(f"尝试设置无效的用户信息字段 '{key}'。")
         
        if updated:
             logger.info(f"用户信息已更新: {self.user_info}")
             self._update_system_prompt() # Re-render prompt with new info

    def start_new_session(self, keep_prompt: bool = True):
        """Clears the current session history.

        Args:
            keep_prompt: Whether to keep the current system prompt. 
                          If False, resets to default.
        """
        self.session_history.clear()
        if not keep_prompt:
             self.current_prompt_identifier = DEFAULT_PROMPT_IDENTIFIER
             self._update_system_prompt()
             logger.info(f"会话已重置，并切换回默认 Prompt '{self.current_prompt_identifier}' (Session reset, switched back to default prompt).")
        else:
            logger.info("新会话已开始，历史已清除 (New session started. History cleared).")

    def _add_to_history(self, role: str, content: str):
        """Adds a message to history, respecting max length."""
        # Avoid adding empty messages
        if content:
            self.session_history.append({"role": role, "content": content})
            logger.debug(f"Added to history - Role: {role}, Content: {content[:100]}...")
        else:
             logger.warning("Attempted to add empty content to history.")

    # !! 修改: 返回 Generator 用于流式处理 (Modification: Return Generator for streaming) !!
    def get_response_stream(self, user_input: str) -> Generator[str, None, None]:
        """Processes user input, gets LLM response as a stream, and manages history.

        Args:
            user_input: The text input from the user.

        Yields:
            str: Chunks of the LLM response text.
            
        Raises:
             RuntimeError: If LLM client is not ready or system prompt is missing.
        """
        if not self.llm_client or not self.llm_client.is_ready():
            logger.error("LLM 客户端不可用或未配置 API Key。(LLM client is not available or API Key not configured.)")
            # 在生成器中，通常通过引发异常来指示严重错误 (In generators, critical errors are often signaled by raising exceptions)
            raise RuntimeError("LLM client not ready.")
            
        if not self.system_prompt:
             logger.error("系统 Prompt 未加载。(System prompt not loaded.)")
             raise RuntimeError("System prompt not loaded.")

        # 先将用户消息添加到历史记录 (Add user message to history first)
        self._add_to_history("user", user_input)

        # 准备发送给 LLM API 的消息列表 (Prepare messages for LLM API)
        messages_for_llm = [{"role": "system", "content": self.system_prompt}] + list(self.session_history)
        logger.debug(f"Sending {len(messages_for_llm)} messages to LLM.")

        full_assistant_response = ""
        try:
            # 从 LLM 获取流式响应 (Get streaming response from LLM)
            stream_generator = self.llm_client.generate_completion_stream(
                messages=messages_for_llm
            )
            
            # 迭代并产生文本块，同时构建完整响应 (Iterate and yield chunks, building the full response)
            for chunk in stream_generator:
                yield chunk
                full_assistant_response += chunk
            
            # 流结束后，将完整的助手响应添加到历史记录 (After stream ends, add complete assistant response to history)
            # 确保在迭代完成后添加 (Ensure it's added after iteration completes)
            self._add_to_history("assistant", full_assistant_response)
            logger.info(f"Finished streaming response. Full length: {len(full_assistant_response)}")

        except Exception as e:
            # 异常可能在 generate_completion_stream 或迭代期间发生
            # (Exception could occur in generate_completion_stream or during iteration)
            logger.error(f"LLM 流式交互过程中出错: {e}", exc_info=True)
            # 让调用者知道发生了错误 (Let the caller know an error occurred)
            # 可以在这里 yield 一个错误消息或重新引发 (Could yield an error message or re-raise)
            yield f"[ERROR: {e}]"
            # 不应将部分或错误响应添加到历史记录 (Should not add partial or error response to history)
            # 如果需要，可以在这里从历史记录中移除最后的用户消息
            # if self.session_history and self.session_history[-1]["role"] == "user":
            #     self.session_history.pop()

    def switch_prompt(self, identifier: str) -> bool:
        """Switches to the specified system prompt.

        Args:
            identifier: The prompt identifier defined in config.toml.
            
        Returns:
            True if the prompt was found and loaded successfully, False otherwise.
        """
        logger.info(f"Attempting to switch prompt to '{identifier}'")
        # Check if identifier exists in config
        if identifier not in self.prompt_manager.prompts_config:
             logger.error(f"Prompt 标识符 '{identifier}' 未在配置中找到 (Prompt identifier '{identifier}' not found in config).")
             return False

        # Try to load and render new Prompt
        old_identifier = self.current_prompt_identifier
        self.current_prompt_identifier = identifier
        self._update_system_prompt() # This will load and render

        if self.system_prompt == DEFAULT_SYSTEM_PROMPT and identifier != DEFAULT_PROMPT_IDENTIFIER:
             logger.warning(f"切换到 Prompt '{identifier}' 失败，已回退到 '{old_identifier}' (Failed to switch to prompt '{identifier}', reverted to '{old_identifier}').")
             self.current_prompt_identifier = old_identifier
             self._update_system_prompt() # Re-render the old one
             return False
        else:
             logger.info(f"已成功切换到 Prompt: '{identifier}' (Switched to prompt: '{identifier}').")
             # Usually recommended to start a new session after switching prompts
             self.start_new_session(keep_prompt=True) 
             return True

# Example Usage (for testing within this file)
if __name__ == "__main__":
    print("--- Initializing Assistant --- ")
    assistant = Assistant()
    print("----------------------------\n")

    if not assistant.llm_client:
        print("Cannot run example interaction without a configured LLM client.")
    else:
        # Example Interaction
        print("Starting interactive session (type 'quit' to exit, 'new' for new session):")
        while True:
            user_text = input("You: ")
            if user_text.lower() == 'quit':
                break
            if user_text.lower() == 'new':
                 assistant.start_new_session()
                 print("Assistant: Okay, let's start fresh!")
                 continue
            if user_text.lower().startswith("prompt "):
                 p_name = user_text[7:].strip()
                 if assistant.switch_prompt(p_name):
                     print(f"Assistant: Switched to prompt '{p_name}'.")
                 else:
                      print(f"Assistant: Couldn't find prompt '{p_name}'.")
                 continue


            # Get response, disable TTS for console interaction
            response = assistant.get_response_stream(user_text)
            
            print(f"Assistant: {response}")
            
            # print("\nCurrent History:") # For debugging
            # for msg in assistant.session_history:
            #    print(f" - {msg['role']}: {msg['content']}")
            # print("----------")

        print("\nSession ended.") 