#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""独立测试 Coqui TTS (XTTSv2) 功能的脚本 (Standalone script to test Coqui TTS (XTTSv2) functionality)."""

import logging
import os
import tempfile
from pathlib import Path
import time
import pickle

# --- 尝试导入必要的库 (Try importing necessary libraries) ---

try:
    from TTS.api import TTS as CoquiTTS
except ImportError:
    print("错误: 未找到 Coqui TTS 库。请确保已在虚拟环境中安装。")
    print("Error: Coqui TTS library not found. Ensure it's installed in your virtual environment.")
    print("运行: pip install TTS torch")
    CoquiTTS = None

# 尝试导入播放库 (Try importing playback libraries)
try:
    import simpleaudio as sa 
except ImportError:
    sa = None
    try:
        from playsound import playsound
    except ImportError:
        playsound = None

# 尝试导入 PyTorch 和必要的配置类 (Try importing PyTorch and necessary config class)
try:
    import torch
    # Note: The exact path might change in future TTS versions
    from TTS.tts.configs.xtts_config import XttsConfig 
    from TTS.tts.models.xtts import XttsAudioConfig
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.speaker_encoder.configs.speaker_encoder_config import SpeakerEncoderConfig
    from TTS.vocoder.configs.hifigan_config import HifiganConfig
except ImportError as e:
    # Don't prevent script from running, but log a warning
    print(f"[警告] 无法导入 torch 或 TTS 配置类: {e}")
    print("[Warning] Could not import torch or TTS config classes.")
    print("         The safe loading context manager cannot be applied, which might lead to UnpicklingError on PyTorch 2.6+.")
    torch = None
    XttsConfig = None
    XttsAudioConfig = None
    BaseDatasetConfig = None
    SpeakerEncoderConfig = None
    HifiganConfig = None

# --- End Imports ---

# --- 配置 (Configuration) ---
# !! 根据你的设置修改这些值 !! (Modify these values based on your setup) !!

# XTTSv2 模型名称或路径 (XTTSv2 model name or path)
# 如果是第一次运行，库会尝试下载此模型 (Library will attempt download if not found locally)
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# [可选] 用于声音克隆的 WAV 文件路径 (Optional path to WAV for voice cloning)
# 需要一个 10-30 秒的清晰单声道 WAV 文件 (Needs a clean 10-30s mono WAV)
SPEAKER_WAV_PATH = "demos/马云/mayun_zh.wav"# "path/to/your/speaker_voice.wav"  # <-- 如果要克隆声音，请取消注释并修改路径

# 合成目标语言 (Target language for synthesis)
# 查看 XTTSv2 支持的语言 (See XTTSv2 supported languages)
TARGET_LANGUAGE = "zh-cn" # 例如: "en", "es", "fr", "zh-cn", "ja"

# 是否使用 GPU (如果你的 PyTorch 支持 CUDA) (Use GPU if PyTorch supports CUDA)
USE_GPU = False

# 测试用的文本 (Text to synthesize for testing)
TEXT_TO_SYNTHESIZE = "Hello there! This is a test of the Coqui Text to Speech engine using the XTTS version 2 model. It should support multiple languages and voice cloning."

# 输出目录 (Output directory)
OUTPUT_DIR = Path("./tts_output_coqui_test")

# --- End Configuration ---

def setup_test_logging():
    """为测试脚本设置基础日志记录 (Setup basic logging for the test script)."""
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # 降低 Coqui TTS 内部的日志级别以减少干扰 (Lower Coqui TTS internal logging)
    logging.getLogger("TTS").setLevel(logging.WARNING)
    return logging.getLogger(__name__)

logger = setup_test_logging()

def play_audio_file(file_path: str):
    """使用 simpleaudio 或 playsound 播放音频文件 (Play audio using simpleaudio or playsound)."""
    playback_available = sa is not None or playsound is not None
    if not playback_available:
        logger.error("No playback library (simpleaudio or playsound) found. Cannot play audio.")
        return
    
    logger.info(f"Attempting to play: {file_path}")
    try:
        if sa:
            wave_obj = sa.WaveObject.from_wave_file(file_path)
            play_obj = wave_obj.play()
            play_obj.wait_done()
            logger.info("Playback finished (simpleaudio).")
        elif playsound:
            playsound(file_path)
            logger.info("Playback finished (playsound).")
    except Exception as e:
        logger.error(f"Error playing audio file {file_path}: {e}", exc_info=True)

def main():
    """主测试函数 (Main test function)."""
    logger.info("--- Coqui TTS (XTTSv2) Standalone Test --- ")

    if CoquiTTS is None:
        logger.critical("Coqui TTS library (TTS) is not installed. Exiting.")
        return

    # 检查 speaker wav 文件 (如果提供了路径) (Check speaker wav if path provided)
    actual_speaker_wav = None
    if SPEAKER_WAV_PATH:
        speaker_path = Path(SPEAKER_WAV_PATH)
        if speaker_path.is_file():
            actual_speaker_wav = str(speaker_path.resolve())
            logger.info(f"Using speaker WAV for cloning: {actual_speaker_wav}")
        else:
            logger.warning(f"Speaker WAV file not found at: {SPEAKER_WAV_PATH}. Using default voice.")
    else:
        logger.info("No speaker WAV provided. Using default voice for the model.")

    # 初始化 TTS 模型 (Initialize TTS model)
    tts_engine: Optional[CoquiTTS] = None
    try:
        logger.info(f"Loading Coqui TTS model: {MODEL_NAME} (GPU: {USE_GPU})... (This may take time and download data)")
        start_time = time.time()

        # --- 修复: 应用 safe_globals 上下文管理器 --- 
        # --- Fix: Apply safe_globals context manager --- 
        if torch and XttsConfig and XttsAudioConfig and BaseDatasetConfig and SpeakerEncoderConfig and HifiganConfig:
            logger.info("Applying torch.serialization.safe_globals context for model loading.")
            # 明确允许加载所需的配置类
            # Explicitly allow loading the required config classes
            with torch.serialization.safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, SpeakerEncoderConfig, HifiganConfig]):
                tts_engine = CoquiTTS(model_name=MODEL_NAME, gpu=USE_GPU)
        else:
            logger.warning("Torch or TTS Config classes not imported, attempting model load without safe_globals context.")
            # Fallback: 尝试直接加载，如果 PyTorch 版本 < 2.6 或用户忽略警告，可能会工作 (Try loading directly, might work if PyTorch < 2.6 or user ignores warning)
            tts_engine = CoquiTTS(model_name=MODEL_NAME, gpu=USE_GPU)
        # --- 修复结束 --- 

        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds.")
    except ImportError as ie:
         #捕获可能由缺少 torch 或 XttsConfig 引起的导入错误
         logger.critical(f"Failed to initialize Coqui TTS model due to ImportError: {ie}", exc_info=True)
         logger.critical("This might be related to the safe_globals fix failing because torch or TTS Config classes could not be imported earlier.")
         return
    except Exception as e:
        # 捕获其他初始化错误，包括之前的 UnpicklingError
        logger.critical(f"Failed to initialize Coqui TTS model: {e}", exc_info=True)
        if isinstance(e, pickle.UnpicklingError) or "UnpicklingError" in str(e):
             logger.critical(f"This looks like the PyTorch 2.6+ UnpicklingError. The safe_globals fix was attempted but might have failed (ensure all necessary Config classes were allowlisted).")
        logger.critical("Ensure TTS, torch, and potentially build tools are installed correctly in your Python 3.9-3.11 environment.")
        return

    if tts_engine is None:
        logger.critical("TTS engine object is None after initialization attempt. Exiting.")
        return
        
    # --- 测试 1: 合成到文件 (Test 1: Synthesize to File) ---
    logger.info("\n--- Test 1: Synthesize to File ---")
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_filename = f"coqui_test_output_{int(time.time())}.wav"
    output_path = OUTPUT_DIR / output_filename
    try:
        logger.info(f"Synthesizing text to: {output_path}")
        logger.info(f"Text: {TEXT_TO_SYNTHESIZE}")
        start_time = time.time()
        tts_engine.tts_to_file(
            text=TEXT_TO_SYNTHESIZE,
            speaker_wav=actual_speaker_wav,
            language=TARGET_LANGUAGE,
            file_path=str(output_path)
        )
        synth_time = time.time() - start_time
        logger.info(f"Successfully synthesized to {output_path} in {synth_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error during synthesis to file: {e}", exc_info=True)

    # --- 测试 2: 合成并播放 (Test 2: Synthesize and Play) ---
    logger.info("\n--- Test 2: Synthesize and Play --- ")
    temp_filename: Optional[str] = None
    try:
        # 合成到临时文件 (Synthesize to temp file)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            temp_filename = fp.name
        logger.info(f"Synthesizing to temporary file for playback: {temp_filename}")
        start_time = time.time()
        tts_engine.tts_to_file(
            text=TEXT_TO_SYNTHESIZE,
            speaker_wav=actual_speaker_wav,
            language=TARGET_LANGUAGE,
            file_path=temp_filename
        )
        synth_time = time.time() - start_time
        logger.info(f"Synthesized temp file in {synth_time:.2f} seconds.")

        # 播放临时文件 (Play the temp file)
        play_audio_file(temp_filename)

    except Exception as e:
        logger.error(f"Error during synthesis or playback: {e}", exc_info=True)
    finally:
        # 清理临时文件 (Clean up temp file)
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
                logger.debug(f"Removed temporary file: {temp_filename}")
            except OSError as e:
                logger.warning(f"Could not remove temporary file {temp_filename}: {e}")

    logger.info("--- Coqui TTS Test Finished --- ")

if __name__ == "__main__":
    main() 