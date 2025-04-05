# ✨ AI 英语学习助手 ✨

一个基于 Python 的综合应用程序，旨在通过与 AI 助手的对话来帮助用户练习和提高英语技能。该应用利用现代 LLM 技术进行自然交互，并具备文本转语音 (TTS) 功能和会话管理。

## 🚀 主要功能

- **智能对话练习**: 与针对语言学习优化的 AI 导师进行互动对话。
- **多种角色选择**: 可选择不同的导师人设（友好、严格、商务导向等）。
- **文本转语音 (TTS) 支持**: 🗣️ 可选的语音输出，支持多种 TTS 引擎。
- **会话管理**: 💾 保存、加载和恢复学习对话。
- **跨平台兼容**: 💻 可在 Windows、macOS 和 Linux 上运行。
- **自定义提示 (Prompts)**: 🔧 可定制 AI 的行为以适应不同的学习目标。
- **多语言支持**: 主要为英语学习设计，但也为其他语言（如中文、日语）提供了基础提示。

## 💡 使用方法

### 启动应用

在项目根目录下，通过命令行运行：

```bash
python -m ai_english_assistant.main
```

### 可用命令

在交互会话中，你可以使用以下命令：

-   `quit` - 退出应用程序
-   `new` - 开始一个新的会话
-   `tts on/off` - 打开/关闭语音输出
-   `prompt <名称>` - 切换到 `config.toml` 中定义的其他角色/提示
-   `save` - 保存当前会话
-   `load <会话ID>` - 加载之前保存的会话
-   `sessions` - 列出所有已保存的会话

### 示例对话

```
Starting interactive session.
Type your message, or:
  'quit' - to exit the application.
  'new'  - to start a new conversation session.
  'tts on'/'tts off' - to toggle speech output for this session.
  'prompt <name>' - to switch to a prompt defined in config.toml.
  'save' - to save the current conversation session.
  'load <session_id>' - to load a previously saved session.
  'sessions' - to list all available saved sessions.
------------------------------------------------

You: 你好，可以帮我练习英语吗？
Assistant: Hi there! Absolutely, I'd be happy to help you practice your English. 
Is there anything specific you'd like to work on today? We could focus on 
conversation, grammar, vocabulary, or pronunciation.

You: 我想练习商务英语，准备面试。
Assistant: That's a great goal! Job interviews in English can be challenging, 
but with some practice, you'll feel more confident. Let's start by discussing 
common interview questions and appropriate responses...

You: save
Assistant: Session saved to sessions/default_tutor_20240406_103045.json
```
## ⚙️ 配置详解

应用程序的主要配置通过 `config/config.toml` 文件进行。关键部分包括：

### [openai]

```toml
[openai]
# api_key = "YOUR_OPENAI_API_KEY_HERE" # 优先使用 OPENAI_API_KEY 环境变量
# base_url = "http://localhost:8000/v1"   # 可选，用于本地模型或兼容接口
model = "gpt-3.5-turbo"              # 默认使用的模型
default_temperature = 0.7              # 回复的创造性 (0.0 - 1.0)
default_max_tokens = 4096              # 单次回复的最大长度
```

### [tts] - 文本转语音

```toml
[tts]
enabled = true               # 是否启用 TTS
engine = "gtts"              # 使用的引擎: gtts, pyttsx3, piper, coqui

# gTTS 特定设置
gtts_lang = "en"             # gTTS 使用的语言 (需要联网)

# pyttsx3 特定设置 (离线, 基础)
# 通常无需额外配置，使用系统自带语音库

# Piper 特定设置 (离线, 需下载模型)
# piper_model_path = "path/to/model.onnx"
# piper_config_path = "path/to/model.onnx.json"

# Coqui TTS 特定设置 (离线, 高质量, 需下载模型)
# coqui_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
# coqui_language = "en"
# coqui_use_gpu = false
```
**TTS 注意事项**: ⚠️
- `gtts` 需要网络连接。
- `pyttsx3` 是离线的，但音质和可用声音依赖于你的操作系统。
- `piper` 和 `coqui` 提供更高质量的离线语音，但需要手动下载和配置模型文件，并安装额外的依赖库 (`pip install TTS` 或特定于 Piper 的库)。安装过程可能比较复杂。
- **后续优化计划** 🚀: 当前 TTS 实现存在一些局限性。**计划在近期集成一个更先进的开源 TTS 仓库（例如基于 Coqui XTTS 或类似的高质量模型）**，旨在提供配置更简单、效果更自然、媲美真人的语音合成体验，让英语练习更加身临其境！敬请期待！

### [prompts] - 角色定义

```toml
[prompts]
# 标识符 = "相对于 prompts/ 目录的文件路径"
default_tutor = "en/persona_lively_female.txt"
tutor_strict = "en/tutor_strict_grammar.txt"
tutor_casual = "en/tutor_casual_conversation.txt"
general_chat_zh = "zh/general_assistant.txt"
```

### [session] - 会话设置

```toml
[session]
max_history = 100    # 保存在上下文中的最大对话轮数
save_path = "sessions/" # 会话保存的目录
```

## 🧩 扩展应用

### 添加新角色 (Prompt)

1.  在 `prompts/` 下相应的语言目录（例如 `en/`）中创建一个新的 `.txt` 文件。
2.  在 `config.toml` 的 `[prompts]` 部分添加一个新的条目，格式为 `名称 = "路径/文件名.txt"`。
3.  在应用中使用 `prompt <名称>` 命令即可切换到新角色。

### 添加新 TTS 引擎

可以通过在 `tts/speech_synthesizer.py` 中扩展 `SpeechSynthesizer` 类并实现新的引擎逻辑来添加支持。

## 🐛 问题排查

### 常见问题

-   **API 密钥问题**: 确保你的 OpenAI API 密钥在 `config.toml` 或环境变量中正确设置。
-   **TTS 不工作**: 检查你选择的 TTS 引擎所需的依赖是否已正确安装，并确认 `config.toml` 中的路径（如果需要）是否正确。
-   **会话加载错误**: 确认提供的会话 ID 存在于 `sessions/` 目录下，并且文件格式正确。

### 日志文件

应用程序的详细运行日志保存在 `logs/` 目录下的 `assistant.log` 文件中。如果遇到问题，请检查此文件获取详细信息。

## 📄 许可证

本项目采用 MIT 许可证 - 详情请参阅 `LICENSE` 文件 (如果存在)。

## 🙏 致谢

-   OpenAI 提供的强大语言模型 API。
-   项目中使用的各个开源 TTS 库。 