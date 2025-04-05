import os
from pathlib import Path
from typing import Dict, Any, Optional

class PromptManager:
    """Manages loading and rendering of prompt templates from files."""

    def __init__(self, prompts_config: Dict[str, str], base_prompt_dir: Optional[str] = None):
        """Initializes the PromptManager.

        Args:
            prompts_config: A dictionary mapping prompt identifiers (keys) to
                            relative file paths (values) as defined in config.toml [prompts].
            base_prompt_dir: The base directory where prompt files are stored. 
                             If None, it defaults to a 'prompts' directory relative 
                             to this file's parent directory.
        """
        self.prompts_config = prompts_config
        
        if base_prompt_dir:
            self.base_dir = Path(base_prompt_dir)
        else:
            # Default to ../prompts relative to this file (manager.py)
            self.base_dir = Path(__file__).parent.parent / "prompts" 
            
        if not self.base_dir.is_dir():
            print(f"Warning: Base prompt directory '{self.base_dir}' not found. Prompt loading will fail.")
            # You might want to raise an error here depending on requirements
            # raise FileNotFoundError(f"Base prompt directory '{self.base_dir}' not found.")

        print(f"PromptManager initialized. Base directory: '{self.base_dir}'")
        print(f"Registered prompt configurations: {list(prompts_config.keys())}")


    def get_prompt_path(self, identifier: str) -> Optional[Path]:
        """Gets the full path to a prompt file based on its identifier."""
        relative_path_str = self.prompts_config.get(identifier)
        if not relative_path_str:
            print(f"Error: Prompt identifier '{identifier}' not found in configuration.")
            return None
            
        # Ensure the relative path doesn't try to escape the base directory (security)
        # Path.joinpath handles this reasonably well, but explicit checks can be added.
        full_path = self.base_dir.joinpath(relative_path_str).resolve()

        # Double-check it's still within the intended base directory
        if self.base_dir.resolve() not in full_path.parents:
             print(f"Error: Prompt path '{relative_path_str}' for identifier '{identifier}' attempts to access files outside the base prompt directory '{self.base_dir}'.")
             return None

        if not full_path.is_file():
            print(f"Error: Prompt file not found at resolved path: '{full_path}' (from identifier: '{identifier}', relative: '{relative_path_str}')")
            return None
            
        return full_path

    def load_prompt_template(self, identifier: str) -> Optional[str]:
        """Loads the raw content (template) of a prompt file."""
        file_path = self.get_prompt_path(identifier)
        if not file_path:
            return None # Error already printed by get_prompt_path

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except IOError as e:
            print(f"Error reading prompt file '{file_path}': {e}")
            return None
        except Exception as e: # Catch potential encoding errors too
             print(f"An unexpected error occurred while reading prompt file '{file_path}': {e}")
             return None


    def render_prompt(self, identifier: str, **kwargs: Any) -> Optional[str]:
        """Loads a prompt template and renders it using provided keyword arguments.

        Args:
            identifier: The identifier of the prompt template to load (must be in config).
            **kwargs: Keyword arguments to fill in the placeholders in the template.
                      Placeholders are defined like {variable_name}.

        Returns:
            The rendered prompt string, or None if loading or rendering fails.
        """
        template = self.load_prompt_template(identifier)
        if template is None:
            return None # Error already printed

        try:
            # Basic templating using f-string logic via format()
            # This requires placeholders like {variable_name} in the text file.
            rendered_prompt = template.format(**kwargs)
            return rendered_prompt
        except KeyError as e:
            print(f"Error rendering prompt '{identifier}': Missing template variable {e}. Provided arguments: {list(kwargs.keys())}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during prompt rendering for '{identifier}': {e}")
            return None

# Example Usage (for testing within this file)
if __name__ == "__main__":
    # Simulate the config loaded from settings
    dummy_prompts_config = {
        "default_tutor": "en/english_tutor.txt",
        "general_chat_zh": "zh/general_assistant.txt",
        "greeting_template": "en/examples/greeting.txt" # Example with template
    }
    
    # Create a dummy prompt file for testing templating
    dummy_prompts_dir = Path("./temp_prompts_for_test")
    dummy_prompts_dir.mkdir(exist_ok=True)
    (dummy_prompts_dir / "en").mkdir(exist_ok=True)
    (dummy_prompts_dir / "en" / "examples").mkdir(exist_ok=True)
    
    greeting_file = dummy_prompts_dir / "en/examples/greeting.txt"
    with open(greeting_file, "w", encoding="utf-8") as f:
        f.write("""--- Role ---
You are a helpful greeter.

--- Task ---
Generate a short, friendly greeting for {user_name}.
Mention that the current time is approximately {current_time}.

--- Constraints ---
- Be polite.
- Keep it under 2 sentences.
""")

    # Initialize the manager using the dummy dir
    manager = PromptManager(prompts_config=dummy_prompts_config, base_prompt_dir=str(dummy_prompts_dir))

    print("\n--- Testing Loading ---")
    raw_tutor_prompt = manager.load_prompt_template("default_tutor")
    # This will fail unless you also copy the actual prompts into the dummy dir for the test
    if raw_tutor_prompt:
       print(f"Loaded 'default_tutor':\n{raw_tutor_prompt[:100]}...")
    else:
        print("Failed to load 'default_tutor' (expected if actual file not in dummy dir).")


    print("\n--- Testing Rendering ---")
    rendered_greeting = manager.render_prompt("greeting_template", user_name="Alice", current_time="evening")
    if rendered_greeting:
        print(f"Rendered 'greeting_template':\n{rendered_greeting}")
    else:
        print("Failed to render 'greeting_template'.")
        
    print("\n--- Testing Rendering Failure (Missing Variable) ---")
    failed_render = manager.render_prompt("greeting_template", user_name="Bob") # Missing current_time
    if not failed_render:
        print("Correctly failed to render due to missing variable.")

    print("\n--- Testing Non-Existent Identifier ---")
    non_existent = manager.load_prompt_template("non_existent_prompt")
    if not non_existent:
        print("Correctly failed to load non-existent prompt.")

    # Clean up the dummy directory and file
    import shutil
    try:
        greeting_file.unlink()
        (dummy_prompts_dir / "en" / "examples").rmdir()
        (dummy_prompts_dir / "en").rmdir()
        dummy_prompts_dir.rmdir()
        print("\nCleaned up temporary test files.")
    except Exception as e:
         print(f"Warning: Could not fully clean up test files: {e}") 