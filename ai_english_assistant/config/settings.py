import toml
import os
from typing import Any, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def find_config_file(filename: str = "config.toml") -> Optional[Path]:
    """Search for the config file upwards from the current dir."""
    current_dir = Path(__file__).parent
    while current_dir != current_dir.parent:
        potential_path = current_dir / filename
        if potential_path.is_file():
            return potential_path
        # Check in a potential parent config directory as well
        parent_config_dir = current_dir.parent / "config"
        if parent_config_dir.is_dir():
             potential_path_in_config = parent_config_dir / filename
             if potential_path_in_config.is_file():
                 return potential_path_in_config
        current_dir = current_dir.parent
    return None

def load_config() -> Dict[str, Any]:
    """Loads configuration from the TOML file.

    Returns:
        A dictionary containing the loaded configuration.

    Raises:
        FileNotFoundError: If the config file cannot be found.
        toml.TomlDecodeError: If the config file is not valid TOML.
        ValueError: If essential config sections are missing.
    """
    config_path = find_config_file()
    if not config_path:
        # Fallback check relative to potential project root if __file__ is deep
        project_root_check = Path.cwd()
        potential_path = project_root_check / "config" / "config.toml"
        if potential_path.is_file():
             config_path = potential_path
        else:
            raise FileNotFoundError("Could not find config.toml in parent directories or ./config/.")

    print(f"Loading configuration from: {config_path}")
    try:
        config = toml.load(config_path)
    except toml.TomlDecodeError as e:
        print(f"Error decoding TOML file at {config_path}: {e}")
        raise

    # --- Environment Variable Overrides & Defaults ---
    # OpenAI API Key: Prioritize environment variable
    if "openai" in config:
        if not config["openai"].get("api_key"): # 如果 config["openai"] 中没有 "api_key" 键，则使用环境变量 OPENAI_API_KEY
            config["openai"]["api_key"] = os.getenv("OPENAI_API_KEY")
        if not config["openai"].get("base_url"): # 如果 config["openai"] 中没有 "base_url" 键，则使用环境变量 OPENAI_BASE_URL   
            config["openai"]["base_url"] = os.getenv("OPENAI_BASE_URL")
        if not config["openai"].get("model"): # 如果 config["openai"] 中没有 "model" 键，则使用环境变量 OPENAI_MODEL
            config["openai"]["model"] = os.getenv("OPENAI_MODEL")
        # env_api_key = os.getenv("OPENAI_API_KEY")
        # if env_api_key:
            # config["openai"]["api_key"] = env_api_key
            # print("Using OpenAI API key from environment variable.")
        # elif not config["openai"].get("api_key") or config["openai"].get("api_key") == "YOUR_OPENAI_API_KEY_HERE":
            #  print("Warning: OpenAI API key not found in config or environment variables.")
             # Depending on strictness, you might raise an error here or allow operation without it initially
             # raise ValueError("OpenAI API key must be set in config.toml or via OPENAI_API_KEY environment variable.")
    else:
        raise ValueError("Missing [openai] section in config file.")

    # Ensure other essential sections/defaults exist
    if "tts" not in config:
        config["tts"] = {"enabled": False} # Default to disabled if section missing
        print("Warning: Missing [tts] section in config. TTS defaulted to disabled.")

    if "prompts" not in config:
        config["prompts"] = {} # Default to empty prompts if section missing

    if "session" not in config:
         config["session"] = {"max_history": 10} # Default history if section missing
         print("Warning: Missing [session] section in config. Defaulting max_history=10.")
    elif "max_history" not in config["session"]:
         config["session"]["max_history"] = 10 # Default if key missing
         print("Warning: Missing 'max_history' in [session]. Defaulting to 10.")


    # Validate specific required fields (optional but recommended)
    if not config["openai"].get("model"):
         raise ValueError("Missing 'model' under [openai] section in config file.")

    return config

# Load settings once when the module is imported
try:
    settings = load_config()
except (FileNotFoundError, ValueError, toml.TomlDecodeError) as e:
    print(f"CRITICAL ERROR loading configuration: {e}")
    # Handle critical error appropriately - maybe exit or use default fallback dict
    settings = {
        "openai": {"api_key": None, "model": "unknown", "default_temperature": 0.7, "default_max_tokens": 150},
        "tts": {"enabled": False},
        "prompts": {},
        "session": {"max_history": 5}
    }
    print("Proceeding with fallback default settings.")

# You can now import 'settings' from this module elsewhere
# from ai_english_assistant.config.settings import settings 