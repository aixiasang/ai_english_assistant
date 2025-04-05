from typing import Optional, Any, Tuple, Union
import os
import time # For unique filenames
import logging
from pathlib import Path
import io
import tempfile
import numpy as np
import re # <-- Import regex module

# --- Engine Imports (Attempting all, will check availability later) --- 
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None 

try:
    from piper.voice import PiperVoice
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    PiperVoice = None
    sd = None
    sf = None

try:
    from gtts import gTTS, gTTSError
except ImportError:
    gTTS = None

# Import a reliable playback library separately
try:
    import simpleaudio as sa 
except ImportError:
     sa = None 
# Also try playsound, needed for MP3
try:
    from playsound import playsound
except ImportError:
    playsound = None

# Try Coqui TTS
try:
    from TTS.api import TTS as CoquiTTS # Rename to avoid conflict
except ImportError:
     CoquiTTS = None

# --- End Engine Imports ---

logger = logging.getLogger(__name__) # Get logger for this module

# Type hint for engine union (Only includes engines that store a persistent object)
EngineType = Union[pyttsx3.Engine, PiperVoice, CoquiTTS, None] 

class SpeechSynthesizer:
    """处理文本到语音的合成和播放，默认使用 gTTS，支持 pyttsx3, Piper, Coqui TTS (XTTS) 作为备选。(Handles TTS using gTTS by default, supports pyttsx3, Piper TTS as alternatives)."""

    def __init__(self, config: dict):
        """初始化 TTS 引擎，根据配置选择。(Initializes TTS engine based on config).

        Args:
            config: 包含 TTS 配置的字典 (Dictionary with TTS config).
                    需要 'enabled' (bool).
                    可选 'engine' ('gtts', 'pyttsx3', 'piper', 'coqui', default 'gtts').
                    Piper: 需要 'piper_model_path', 可选 'piper_config_path'.
                    Coqui: 需要 'coqui_model_name', 可选 'coqui_speaker_wav'.
                    gTTS: 可选 'gtts_lang'.
        """
        self.enabled = config.get('enabled', False)
        self.engine_type: Optional[str] = None
        self.engine: EngineType = None 
        self.config = config
        self.output_dir = Path("tts_output") # Define output directory
        self.output_dir.mkdir(exist_ok=True) # Create directory
        self.piper_sample_rate: Optional[int] = None # Store piper model sample rate
        self.gtts_lang = config.get('gtts_lang', 'en') # Language for gTTS
        self.coqui_speaker_wav = config.get('coqui_speaker_wav') # For XTTS voice cloning
        self.coqui_language = config.get('coqui_language', 'en') # Target language for Coqui synthesis

        # Check playback availability early
        self.playback_available = sa is not None or playsound is not None
        if not self.playback_available:
             logger.warning("No playback library found (simpleaudio or playsound). Speak functionality will be limited.")
        elif not sa:
             logger.warning("simpleaudio not found. WAV playback (used by Coqui/Piper) will not work.")
        elif not playsound:
             logger.warning("playsound not found. MP3 playback (used by gTTS) will not work.")

        if not self.enabled:
            logger.info("TTS is disabled in the configuration.")
            return

        # --- 选择并初始化引擎 (Select and initialize engine) --- 
        self.engine_type = config.get('engine', 'gtts').lower()
        logger.info(f"Attempting to initialize TTS engine: '{self.engine_type}'")

        if self.engine_type == 'gtts':
            if gTTS and self.playback_available:
                logger.info(f"gTTS engine selected. Language: '{self.gtts_lang}'. Requires internet and playback library.")
                self.engine = "gTTS_ready" # type: ignore 
            elif not gTTS:
                logger.error("gTTS engine selected but library not found. Install with `pip install gTTS`.")
                self.enabled = False
        elif self.engine_type == 'pyttsx3':
            if pyttsx3:
                try:
                    self.engine = pyttsx3.init()
                    logger.info("pyttsx3 engine initialized successfully.")
                except Exception as e:
                    logger.error(f"Error initializing pyttsx3 engine: {e}", exc_info=True)
                    self.enabled = False
            else:
                logger.error("pyttsx3 engine selected but library not found. Install with `pip install pyttsx3`.")
                self.enabled = False
        elif self.engine_type == 'piper':
            if PiperVoice and sd and sf:
                model_path = config.get('piper_model_path')
                config_path = config.get('piper_config_path') # Optional
                if not model_path:
                    logger.error("Piper engine selected but 'piper_model_path' not found in TTS config.")
                    self.enabled = False
                else:
                    model_path = str(Path(model_path).resolve()) # Ensure absolute path
                    if config_path:
                        config_path = str(Path(config_path).resolve())
                    if not Path(model_path).is_file():
                        logger.error(f"Piper model file not found at: '{model_path}'")
                        self.enabled = False
                    elif config_path and not Path(config_path).is_file():
                        logger.error(f"Piper config file not found at: '{config_path}'")
                        self.enabled = False
                    else:
                        try:
                            logger.info(f"Loading Piper voice model from: {model_path}")
                            self.engine = PiperVoice.load(model_path, config_path=config_path, use_cuda=False)
                            self.piper_sample_rate = self.engine.config.sample_rate
                            logger.info(f"Piper engine initialized successfully. Sample rate: {self.piper_sample_rate} Hz")
                        except Exception as e:
                            logger.error(f"Error loading Piper voice model: {e}", exc_info=True)
                            self.enabled = False
            else:
                logger.error("Piper engine selected but libraries not found. Install with `pip install piper-tts sounddevice soundfile onnxruntime`.")
                self.enabled = False
        elif self.engine_type == 'coqui':
            if CoquiTTS and self.playback_available:
                model_name = config.get('coqui_model_name')
                if not model_name:
                    logger.error("Coqui engine selected but 'coqui_model_name' not found in TTS config (e.g., 'tts_models/multilingual/multi-dataset/xtts_v2').")
                    self.enabled = False
                else:
                    try:
                        # Check if a speaker WAV is provided for cloning
                        if self.coqui_speaker_wav and Path(self.coqui_speaker_wav).is_file():
                            logger.info(f"Initializing Coqui TTS model '{model_name}' with speaker cloning from: {self.coqui_speaker_wav}")
                        else:
                            logger.info(f"Initializing Coqui TTS model '{model_name}' (using default speaker if not cloning).")
                            if self.coqui_speaker_wav: # Log warning if path provided but invalid
                                logger.warning(f"Coqui speaker WAV path provided but not found: {self.coqui_speaker_wav}. Using default speaker.")
                                self.coqui_speaker_wav = None # Clear invalid path
                        
                        # Determine if GPU should be used (requires PyTorch with CUDA)
                        # Could add a config option for this, defaulting to False for broader compatibility
                        use_gpu = config.get('coqui_use_gpu', False)
                        logger.debug(f"Coqui TTS use_gpu set to: {use_gpu}")
                        
                        self.engine = CoquiTTS(model_name=model_name, gpu=use_gpu)
                        logger.info(f"Coqui TTS model '{model_name}' loaded successfully.")
                        
                    except Exception as e:
                        logger.error(f"Error initializing Coqui TTS model '{model_name}': {e}", exc_info=True)
                        logger.error("Ensure you have installed Coqui TTS correctly (pip install TTS) and have necessary dependencies (PyTorch, possibly build tools).")
                        self.enabled = False
            elif not CoquiTTS:
                logger.error("Coqui engine selected but library not found. Install `pip install TTS torch`. Check Coqui TTS documentation for detailed setup.")
                self.enabled = False
            else:
                logger.error("Coqui engine selected but no playback library found. Install `pip install simpleaudio` or `pip install playsound`.")
                self.enabled = False
        else:
            logger.error(f"Unsupported TTS engine specified in config: '{self.engine_type}'. Supported: 'gtts', 'pyttsx3', 'piper', 'coqui'.")
            self.enabled = False
        
        if not self.enabled:
            logger.warning("TTS initialization failed or engine disabled.")

    def _generate_output_path(self, text: str, extension: str = "wav") -> str:
        """生成唯一的输出文件路径 (Generates a unique output file path)."""
        timestamp = int(time.time() * 1000)
        safe_prefix = ''.join(c for c in text[:20] if c.isalnum() or c in ' -_').rstrip() or "audio"
        filename = f"{safe_prefix}_{timestamp}.{extension}"
        return str(self.output_dir / filename)

    def _preprocess_text_for_gtts(self, text: str) -> str:
        """Removes special characters and emojis problematic for gTTS, replacing them with spaces.
           Keeps alphanumeric characters (including Unicode letters like Chinese), whitespace,
           and basic punctuation (. , ? !).
        """
        # Pattern to match characters NOT in the allowed list OR emojis
        # Allowed: word chars (letters, numbers, underscore, Unicode letters), whitespace, ., ,, ?, !
        # Note: \w includes underscore. If underscore is problematic, it needs separate handling.
        # We use + to replace sequences of unwanted chars with a single space.
        pattern = r"[^\w\s.,?!]+|[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U00002600-\U000026FF]+"
        processed_text = re.sub(pattern, ' ', text)
        # Optional: Clean up multiple spaces resulting from replacements
        processed_text = re.sub(r'\s{2,}', ' ', processed_text).strip()
        if text != processed_text:
            logger.debug(f"Preprocessed gTTS text: '{processed_text[:100]}...' (Original: '{text[:100]}...')")
        return processed_text

    def synthesize_to_file(self, text: str) -> Optional[str]:
        """将给定文本合成为音频文件并保存 (Synthesizes text to audio file and saves it).

        Args:
            text: 要合成的文本 (Text to synthesize).

        Returns:
            生成文件的路径，如果失败则为 None (Path to the generated file, or None on failure).
        """
        if not self.enabled:
            logger.warning("TTS synthesis skipped: TTS is disabled.")
            return None
        if not self.engine and self.engine_type not in ['gtts']:
            logger.warning(f"TTS synthesis skipped: Engine '{self.engine_type}' not initialized.")
            return None
        if not text:
            logger.warning("TTS synthesis skipped: No text provided.")
            return None

        output_path: Optional[str] = None
        try:
            if self.engine_type == 'gtts':
                if not gTTS:
                    logger.error("gTTS library not available for synthesis.")
                    return None
                output_path = self._generate_output_path(text, "mp3")
                # Preprocess text specifically for gTTS
                processed_text = self._preprocess_text_for_gtts(text)
                if not processed_text:
                    logger.warning("TTS synthesis skipped: Text became empty after preprocessing.")
                    return None
                logger.info(f"Synthesizing (gtts) text: '{processed_text[:50]}...' to file: {output_path}")
                tts = gTTS(text=processed_text, lang=self.gtts_lang)
                tts.save(output_path)
                logger.info(f"Audio successfully saved (gTTS) to {output_path}")
                return output_path
            elif self.engine_type == 'pyttsx3' and isinstance(self.engine, pyttsx3.Engine):
                output_path = self._generate_output_path(text, "wav")
                logger.info(f"Synthesizing (pyttsx3) text: '{text[:50]}...' to file: {output_path}")
                self.engine.save_to_file(text, output_path)
                self.engine.runAndWait() 
                logger.info(f"Audio successfully saved (pyttsx3) to {output_path}")
                return output_path
            elif self.engine_type == 'piper' and isinstance(self.engine, PiperVoice):
                output_path = self._generate_output_path(text, "wav")
                logger.info(f"Synthesizing (piper) text: '{text[:50]}...' to file: {output_path}")
                audio_bytes = b"".join(self.engine.synthesize(text))
                if not audio_bytes:
                    raise ValueError("Piper synthesis returned empty audio bytes.")
                
                with sf.SoundFile(output_path, mode='wb', samplerate=self.piper_sample_rate, channels=1, format='WAV') as audio_file:
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_file.write(audio_data)
                logger.info(f"Audio successfully saved (Piper) to {output_path}")
                return output_path
            elif self.engine_type == 'coqui' and isinstance(self.engine, CoquiTTS):
                output_path = self._generate_output_path(text, "wav")
                logger.info(f"Synthesizing (coqui) text: '{text[:50]}...' to file: {output_path}")
                # Use tts_to_file with speaker_wav if provided
                self.engine.tts_to_file(
                    text=text, 
                    speaker_wav=self.coqui_speaker_wav, 
                    language=self.coqui_language, 
                    file_path=output_path
                )
                logger.info(f"Audio successfully saved (Coqui) to {output_path}")
                return output_path
            else:
                logger.error(f"Synthesize_to_file: Unsupported or uninitialized engine type ('{self.engine_type}').")
                return None
        except gTTSError as e:
            logger.error(f"gTTS API error during synthesis: {e}")
            if output_path and os.path.exists(output_path):
                try: os.remove(output_path)
                except OSError as ose: logger.warning(f"Could not remove Gtts error file {output_path}: {ose}")
            return None
        except Exception as e:
            logger.error(f"Error during TTS synthesis ('{self.engine_type}') to file '{output_path or 'N/A'}': {e}", exc_info=True)
            if output_path and os.path.exists(output_path):
                try: os.remove(output_path)
                except OSError as ose: logger.warning(f"Could not remove potentially incomplete TTS file {output_path}: {ose}")
            return None

    def _play_audio_file(self, file_path: str):
        """Plays an audio file using simpleaudio (for WAV) or playsound (for MP3)."""
        file_path_lower = file_path.lower()
        try:
            if file_path_lower.endswith(".wav"):
                if sa:
                    logger.debug(f"Playing WAV '{file_path}' using simpleaudio")
                    wave_obj = sa.WaveObject.from_wave_file(file_path)
                    play_obj = wave_obj.play()
                    play_obj.wait_done()
                    logger.debug("simpleaudio playback finished.")
                else:
                    logger.error(f"Cannot play WAV file '{file_path}': simpleaudio library not found.")
            elif file_path_lower.endswith(".mp3"):
                if playsound:
                    logger.debug(f"Playing MP3 '{file_path}' using playsound")
                    # playsound might have issues with paths containing spaces or special chars
                    # Consider quoting or using normpath if issues arise.
                    playsound(file_path)
                    logger.debug("playsound playback finished.")
                else:
                    logger.error(f"Cannot play MP3 file '{file_path}': playsound library not found.")
            else:
                logger.error(f"Unsupported audio file format for playback: {file_path}")
        except Exception as e:
            logger.error(f"Error playing audio file {file_path}: {e}", exc_info=True)

    def speak(self, text: str):
        """直接合成并播放文本 (Synthesizes and speaks the text directly).

        Args:
            text: 要朗读的文本 (Text to speak).
        """
        if not self.enabled:
            logger.warning("TTS speak skipped: Disabled.")
            return
        if not self.engine and self.engine_type not in ['gtts']:
            logger.warning(f"TTS speak skipped: Engine '{self.engine_type}' not init.")
            return
        if not text:
            logger.warning("TTS speak skipped: No text.")
            return

        logger.info(f"Speaking ('{self.engine_type}'): '{text[:50]}...'")
        temp_filename: Optional[str] = None # For gTTS temp file cleanup
        try:
            if self.engine_type == 'gtts':
                if not gTTS:
                    logger.error("gTTS library not available for speaking.")
                    return
                # Preprocess text specifically for gTTS
                processed_text = self._preprocess_text_for_gtts(text)
                if not processed_text:
                    logger.warning("TTS speak skipped: Text became empty after preprocessing.")
                    return
                
                temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                temp_filename = temp_file.name
                temp_file.close() # Close the file handle so gTTS can write to it
                
                logger.debug(f"Synthesizing gTTS to temporary file: {temp_filename}")
                tts = gTTS(text=processed_text, lang=self.gtts_lang)
                tts.save(temp_filename)
                
                logger.debug(f"Playing temporary gTTS file: {temp_filename}")
                self._play_audio_file(temp_filename)
            elif self.engine_type == 'pyttsx3' and isinstance(self.engine, pyttsx3.Engine):
                self.engine.say(text)
                self.engine.runAndWait() 
            elif self.engine_type == 'piper' and isinstance(self.engine, PiperVoice):
                audio_bytes = b"".join(self.engine.synthesize(text))
                if not audio_bytes:
                    raise ValueError("Piper synthesis returned empty audio bytes for speak.")
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                logger.debug(f"Playing Piper audio (Sample Rate: {self.piper_sample_rate} Hz, Duration: {len(audio_data)/self.piper_sample_rate:.2f}s)")
                sd.play(audio_data, samplerate=self.piper_sample_rate)
                sd.wait()
                logger.debug("Piper audio playback finished.")
            elif self.engine_type == 'coqui' and isinstance(self.engine, CoquiTTS):
                # Synthesize to a temporary WAV file, then play it
                if not self.playback_available:
                    logger.error(f"Playback library (simpleaudio/playsound) not available for '{self.engine_type}'.")
                    return
                
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_filename = temp_file.name
                temp_file.close()
                
                logger.debug(f"Synthesizing Coqui TTS to temporary file: {temp_filename}")
                self.engine.tts_to_file(
                    text=text, 
                    speaker_wav=self.coqui_speaker_wav, 
                    language=self.coqui_language, 
                    file_path=temp_filename
                )
                logger.debug(f"Playing temporary Coqui TTS file: {temp_filename}")
                self._play_audio_file(temp_filename)
            else:
                logger.error(f"Speak: Unsupported or uninitialized engine type ('{self.engine_type}').")
        except gTTSError as e:
            logger.error(f"gTTS API error during speak: {e}")
        except Exception as e:
            logger.error(f"Error during TTS speak ('{self.engine_type}'): {e}", exc_info=True)
        finally:
            if temp_filename and os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                    logger.debug(f"Removed temporary TTS file: {temp_filename}")
                except OSError as e:
                    logger.warning(f"Could not remove temporary TTS file {temp_filename}: {e}")

# --- 测试部分更新 --- (Test section update) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- SpeechSynthesizer Test --- ")

    # --- Test gTTS (Default) --- 
    print("\n--- Testing gTTS (Default Engine) --- ")
    config_gtts = {"enabled": True, "engine": "gtts", "gtts_lang": "en"}
    synth_gtts = SpeechSynthesizer(config=config_gtts)
    if synth_gtts.enabled:
        print("Speaking via gTTS (requires internet and playback library)...")
        synth_gtts.speak("Testing the Google Text to Speech engine. Seven, eight, nine.")
        print("Saving via gTTS...")
        file_gtts = synth_gtts.synthesize_to_file("Saving this output generated by gTTS.")
        if file_gtts: print(f"gTTS saved to: {file_gtts} (MP3 format)")
    else: print("gTTS engine unavailable. Install with `pip install gTTS` and `simpleaudio` or `playsound`.")

    # --- Test pyttsx3 (Alternative) --- 
    print("\n--- Testing pyttsx3 (Alternative) --- ")
    config_pyttsx3 = {"enabled": True, "engine": "pyttsx3"}
    synth_pyttsx3 = SpeechSynthesizer(config=config_pyttsx3)
    if synth_pyttsx3.enabled:
        synth_pyttsx3.speak("Testing the pyttsx3 engine. One, two, three.")
        file_pyttsx3 = synth_pyttsx3.synthesize_to_file("Saving this via pyttsx3.")
        if file_pyttsx3: print(f"pyttsx3 saved to: {file_pyttsx3}")
    else: print("pyttsx3 engine unavailable. Install `pip install pyttsx3`")

    # --- Test Piper (Alternative) --- 
    print("\n--- Testing Piper TTS (Alternative) --- ")
    piper_model = "path/to/your/downloaded/en_US-lessac-medium.onnx" # <--- CHANGE THIS
    piper_config = "path/to/your/downloaded/en_US-lessac-medium.onnx.json" # <--- CHANGE THIS
    config_piper = {
        "enabled": True, 
        "engine": "piper",
        "piper_model_path": piper_model,
        "piper_config_path": piper_config
    }
    if "path/to/your" not in piper_model and Path(piper_model).exists():
        synth_piper = SpeechSynthesizer(config=config_piper)
        if synth_piper.enabled:
            synth_piper.speak("Testing the Piper Text to Speech engine. Four, five, six.")
            file_piper = synth_piper.synthesize_to_file("Saving output generated by Piper.")
            if file_piper: print(f"Piper saved to: {file_piper}")
        else: print("Piper engine failed to initialize. Check paths & dependencies (piper-tts, onnxruntime, sounddevice, soundfile).")
    else: print("Skipping Piper test: Update paths in __main__ block.")

    # --- Test gTTS Japanese (Alternative Language) --- 
    print("\n--- Testing gTTS Japanese --- ")
    config_gtts_ja = {"enabled": True, "engine": "gtts", "gtts_lang": "ja"}
    synth_gtts_ja = SpeechSynthesizer(config=config_gtts_ja)
    if synth_gtts_ja.enabled:
        print("Speaking Japanese via gTTS (requires internet and playback library)...")
        # Japanese text: "こんにちは、世界！これは日本語のテストです。" (Konnichiwa, sekai! Kore wa Nihongo no tesuto desu.) - Hello, world! This is a Japanese test.
        japanese_text = "こんにちは、世界！これは日本語のテストです。"
        synth_gtts_ja.speak(japanese_text)
        print("Saving Japanese via gTTS...")
        file_gtts_ja = synth_gtts_ja.synthesize_to_file(japanese_text)
        if file_gtts_ja: print(f"gTTS Japanese saved to: {file_gtts_ja}")
    else: print("gTTS engine unavailable or failed to initialize for Japanese.")

    # --- Test Coqui TTS (Alternative) --- 
    print("\n--- Testing Coqui TTS / XTTS (Alternative) --- ")
    # !! Requires model download first (XTTS models are large) !!
    #    Run in Python: from TTS.api import TTS; TTS().download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    #    Or download manually from Hugging Face
    coqui_model = "tts_models/multilingual/multi-dataset/xtts_v2" # Default XTTSv2 model name/path
    # !! Optionally, provide a path to a clean WAV file (10-30s) for voice cloning !!
    speaker_wav_path = "path/to/your/speaker_voice.wav" # <--- CHANGE THIS (Optional)
    
    config_coqui = {
        "enabled": True, 
        "engine": "coqui",
        "coqui_model_name": coqui_model,
        # "coqui_speaker_wav": speaker_wav_path, # Uncomment and set path for cloning
        "coqui_language": "en", # Target language
        # "coqui_use_gpu": False # Set to True if you have CUDA setup
    }
    
    # Basic check if speaker path seems valid IF provided
    speaker_path_ok = not speaker_wav_path or "path/to/your" not in speaker_wav_path and Path(speaker_wav_path).exists()
    # You might need a more robust check depending on how you handle model paths
    model_path_assumed_exists = True # Assume TTS library handles model download/caching if not a full path
    
    if speaker_path_ok and model_path_assumed_exists:
        synth_coqui = SpeechSynthesizer(config=config_coqui)
        if synth_coqui.enabled:
            print("Speaking via Coqui TTS... (First time might download model)")
            synth_coqui.speak("Testing the Coqui Text to Speech engine, which supports voice cloning.")
            print("Saving via Coqui TTS...")
            file_coqui = synth_coqui.synthesize_to_file("Saving output generated by Coqui TTS.")
            if file_coqui: print(f"Coqui TTS saved to: {file_coqui}")
        else: print("Coqui TTS engine failed to initialize. Check install (TTS, torch), dependencies, and model name/paths.")
    else:
        print("Skipping Coqui TTS test: Check speaker_wav_path if provided, or ensure model is downloadable/available.") 