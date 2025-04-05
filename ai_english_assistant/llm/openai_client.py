import os
import time
from typing import Any, Dict, List, Optional, Generator
import logging

# 实际导入 OpenAI 库 (Import the actual OpenAI library)
from openai import OpenAI, OpenAIError, Stream, APIConnectionError, RateLimitError, APITimeoutError
from openai.types.chat import ChatCompletionChunk

logger = logging.getLogger(__name__)

class OpenAIClient:
    """封装与 OpenAI API 的交互，支持自定义 base_url 和流式响应。(Encapsulates interaction with the OpenAI API, supporting custom base_url and streaming responses)."""

    def __init__(self,
                 api_key: Optional[str],
                 model: str,
                 base_url: Optional[str] = None,
                 default_temperature: float = 0.7,
                 default_max_tokens: int = 150,
                 max_retries: int = 3,
                 retry_delay: float = 2.0):
        """初始化 OpenAI 客户端 (Initializes the OpenAI client).

        Args:
            api_key: OpenAI API 密钥 (可以为 None，但会导致无法调用 API). (OpenAI API key (can be None, but API calls will fail)).
            model: 默认使用的模型。(Default model to use).
            base_url: 可选的基础 URL，用于兼容其他 OpenAI 接口。(Optional base URL for compatibility).
            default_temperature: 默认的创造性/随机性因子。(Default creativity/randomness factor).
            default_max_tokens: 默认生成的最大 token 数。(Default maximum number of tokens to generate).
            max_retries: 连接失败时的最大重试次数。(Maximum number of retries on connection failure).
            retry_delay: 重试之间的延迟（秒）。(Delay between retries in seconds).
        """
        self.api_key = api_key
        self.default_model = model
        self.base_url = base_url
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client: Optional[OpenAI] = None
        
        # 初始化客户端，如果API密钥可用 (Initialize client if API key is available)
        self._initialize_client()
        
    def _initialize_client(self) -> bool:
        """初始化OpenAI客户端实例 (Initialize the OpenAI client instance).
        
        Returns:
            成功初始化返回True，否则返回False (True if initialization successful, False otherwise).
        """
        if not self.api_key:
            logger.warning("未提供 OpenAI API Key。客户端未初始化。(Warning: OpenAI API Key not provided. Client not initialized.)")
            return False
            
        try:
            # 根据是否有 base_url 来初始化客户端
            # (Initialize client based on presence of base_url)
            if self.base_url:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                logger.info(f"OpenAI Client initialized for model '{self.default_model}' with custom base_url: '{self.base_url}'")
            else:
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI Client initialized for model '{self.default_model}' using default base URL.")
            return True

        except OpenAIError as e:
             logger.error(f"初始化 OpenAI 客户端时出错: {e}", exc_info=True)
             self.client = None # Ensure client is None if init fails
             return False
        except Exception as e:
             logger.error(f"初始化 OpenAI 客户端时发生未知错误: {e}", exc_info=True)
             self.client = None
             return False

    def is_ready(self) -> bool:
        """检查客户端是否已初始化并准备好进行 API 调用 (Checks if the client is initialized and ready for API calls)."""
        return self.client is not None

    def _handle_connection_error(self, error: Exception) -> bool:
        """处理连接错误，如果可能的话重新初始化客户端 (Handle connection error, re-initialize client if possible).
        
        Args:
            error: 发生的错误 (The error that occurred).
            
        Returns:
            如果成功重新连接返回True，否则返回False (True if reconnection successful, False otherwise).
        """
        if isinstance(error, (APIConnectionError, APITimeoutError)):
            logger.warning(f"OpenAI API 连接错误: {error}。尝试重新初始化客户端。")
            return self._initialize_client()
        return False

    def generate_completion(
        self,
        messages: List[Dict[str, str]], # 现在 messages 是必需的，包含 system prompt
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> Optional[str]: # 返回 Optional[str] 以表示可能的失败
        """使用指定的模型和参数生成文本补全 (Generates text completion using the specified model and parameters).

        Args:
            messages: 包含系统消息和对话历史的消息列表。(List of messages including system message and history).
            model: 覆盖默认模型。(Override the default model).
            temperature: 覆盖默认 temperature。(Override the default temperature).
            max_tokens: 覆盖默认 max_tokens。(Override the default max tokens).
            **kwargs: 传递给 OpenAI API 调用的其他参数。(Additional arguments passed to the OpenAI API call).

        Returns:
            生成的文本内容，如果出错则为 None。(The generated text content, or None on error).
        """
        if not self.is_ready():
            logger.error("错误: OpenAI 客户端未初始化或未准备好。(Error: OpenAI client not initialized or ready.)")
            return None

        _model = model or self.default_model
        _temperature = temperature or self.default_temperature
        _max_tokens = max_tokens or self.default_max_tokens

        retries = 0
        while retries <= self.max_retries:
            try:
                # 使用 chat.completions.create API
                logger.info(f"Calling OpenAI API: model='{_model}', temp='{_temperature}', max_tokens='{_max_tokens}'")
                response = self.client.chat.completions.create(
                    model=_model,
                    messages=messages,
                    temperature=_temperature,
                    max_tokens=_max_tokens,
                    **kwargs
                )

                # 检查是否有有效的响应内容 (Check for valid response content)
                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    return response.choices[0].message.content.strip()
                else:
                     logger.warning(f"警告: OpenAI API 响应格式无效或内容为空。Response: {response}")
                     return None # Indicate unexpected response format

            except RateLimitError as e:
                logger.warning(f"OpenAI API 速率限制错误: {e}。已尝试 {retries+1}/{self.max_retries+1} 次")
                if retries < self.max_retries:
                    # 对于速率限制，使用指数退避等待更长时间
                    wait_time = self.retry_delay * (2 ** retries)
                    logger.info(f"等待 {wait_time:.2f} 秒后重试...")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    logger.error(f"达到最大重试次数 ({self.max_retries})。放弃请求。")
                    return None
                
            except (APIConnectionError, APITimeoutError) as e:
                logger.warning(f"OpenAI API 连接错误: {e}。已尝试 {retries+1}/{self.max_retries+1} 次")
                if retries < self.max_retries:
                    # 尝试重新初始化客户端
                    self._handle_connection_error(e)
                    wait_time = self.retry_delay * (1.5 ** retries)
                    logger.info(f"等待 {wait_time:.2f} 秒后重试...")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    logger.error(f"达到最大重试次数 ({self.max_retries})。放弃请求。")
                    return None
                    
            except OpenAIError as e:
                # 处理特定的 OpenAI API 错误 (Handle specific OpenAI API errors)
                logger.error(f"错误: 调用 OpenAI API 时出错: {e}")
                # 可以根据 e.status_code 等进行更详细的处理
                # (Could add more detailed handling based on e.status_code etc.)
                return None
                
            except Exception as e:
                # 处理其他意外错误 (Handle other unexpected errors)
                logger.error(f"错误: 调用 generate_completion 时发生未知错误: {e}", exc_info=True)
                return None

    # !! 修改: 实现流式响应 (Modification: Implement streaming response) !!
    def generate_completion_stream(
        self,
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> Generator[str, None, None]: # 返回一个字符串生成器 (Returns a generator of strings)
        """使用指定的模型和参数生成文本补全，并以流式方式返回。(Generates text completion using the specified model and parameters, returned as a stream).

        Args:
            messages: 包含系统消息和对话历史的消息列表。(List of messages including system message and history).
            model: 覆盖默认模型。(Override the default model).
            temperature: 覆盖默认 temperature。(Override the default temperature).
            max_tokens: 覆盖默认 max_tokens。(Override the default max tokens).
            **kwargs: 传递给 OpenAI API 调用的其他参数。(Additional arguments passed to the OpenAI API call).

        Yields:
            str: 从 API 返回的文本块。(Chunks of text content from the API).
            
        Raises:
            # 异常会在迭代生成器时抛出 (Exceptions will be raised during iteration)
        """
        if not self.is_ready():
            logger.error("OpenAI 客户端未初始化或未准备好，无法生成流式响应。(OpenAI client not initialized or ready, cannot generate stream.)")
            # 对于生成器，我们不能直接返回 None，可以选择返回一个空的生成器或立即引发异常
            # (For generators, we can't return None directly. Options: return empty generator or raise immediately)
            # raise RuntimeError("OpenAI client not ready.") 
            yield "[ERROR: OpenAI client not ready]"
            return # 返回空生成器 (Return an empty generator)

        _model = model or self.default_model
        _temperature = temperature or self.default_temperature
        _max_tokens = max_tokens or self.default_max_tokens

        retries = 0
        while retries <= self.max_retries:
            try:
                # 使用 stream=True 调用 API (Call API with stream=True)
                logger.info(f"Calling OpenAI API (stream): model='{_model}', temp='{_temperature}', max_tokens='{_max_tokens}'")
                stream: Stream[ChatCompletionChunk] = self.client.chat.completions.create(
                    model=_model,
                    messages=messages,
                    temperature=_temperature,
                    max_tokens=_max_tokens,
                    stream=True,
                    **kwargs
                )
                
                # 迭代流式响应块 (Iterate over the stream chunks)
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        # 提取并产生文本内容 (Extract and yield the text content)
                        yield chunk.choices[0].delta.content
                # 流式响应完成后退出重试循环
                return
                    
            except RateLimitError as e:
                logger.warning(f"OpenAI API 速率限制错误 (stream): {e}。已尝试 {retries+1}/{self.max_retries+1} 次")
                if retries < self.max_retries:
                    # 对于速率限制，使用指数退避等待更长时间
                    wait_time = self.retry_delay * (2 ** retries)
                    logger.info(f"等待 {wait_time:.2f} 秒后重试...")
                    yield f"[API rate limit exceeded. Retrying in {wait_time:.1f}s...]"
                    time.sleep(wait_time)
                    retries += 1
                else:
                    logger.error(f"达到最大重试次数 ({self.max_retries})。放弃流式请求。")
                    yield f"[ERROR: Maximum retries ({self.max_retries}) exceeded due to rate limits]"
                    return
                
            except (APIConnectionError, APITimeoutError) as e:
                logger.warning(f"OpenAI API 连接错误 (stream): {e}。已尝试 {retries+1}/{self.max_retries+1} 次")
                if retries < self.max_retries:
                    # 尝试重新初始化客户端
                    reconnected = self._handle_connection_error(e)
                    wait_time = self.retry_delay * (1.5 ** retries)
                    logger.info(f"等待 {wait_time:.2f} 秒后重试...")
                    yield f"[Connection error. {'' if reconnected else 'Failed to reconnect. '}Retrying in {wait_time:.1f}s...]"
                    time.sleep(wait_time)
                    retries += 1
                else:
                    logger.error(f"达到最大重试次数 ({self.max_retries})。放弃流式请求。")
                    yield f"[ERROR: Maximum retries ({self.max_retries}) exceeded due to connection issues]"
                    return
                    
            except OpenAIError as e:
                logger.error(f"调用 OpenAI API (stream) 时出错: {e}", exc_info=True)
                yield f"[ERROR: OpenAI API error: {str(e)}]"
                return
                
            except Exception as e:
                logger.error(f"调用 generate_completion_stream 时发生未知错误: {e}", exc_info=True)
                yield f"[ERROR: Unexpected error: {str(e)}]"
                return

# 保持 __main__ 部分用于基本测试 (Keep __main__ block for basic testing)
if __name__ == "__main__":
    # Setup basic logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("--- OpenAIClient Stream Test --- (Requires OPENAI_API_KEY env var)")
    api_key_from_env = os.getenv("OPENAI_API_KEY")
    base_url_from_env = os.getenv("OPENAI_BASE_URL") # Optional: for testing compatible endpoints
    
    if not api_key_from_env:
        logger.warning("Skipping test: Set OPENAI_API_KEY environment variable to run.")
    else:
        try:
            client = OpenAIClient(api_key=api_key_from_env, model="gpt-3.5-turbo", base_url=base_url_from_env)
            
            if client.is_ready():
                test_messages = [
                    {"role": "system", "content": "You are a helpful assistant that explains things simply."},
                    {"role": "user", "content": "Explain the concept of photosynthesis in one sentence."}
                ]
                
                print("\n--- Generating Stream Response --- ")
                full_response = ""
                try:
                    stream_generator = client.generate_completion_stream(messages=test_messages)
                    for chunk in stream_generator:
                        print(chunk, end='', flush=True) # Print chunks as they arrive
                        full_response += chunk
                    print() # Newline after stream finishes
                    logger.info(f"Full streamed response received: {full_response}")
                except Exception as stream_err:
                     logger.error(f"Error during stream processing: {stream_err}")
                     
            else:
                 logger.error("Client is not ready for testing.")

        except Exception as ex:
            logger.error(f"An error occurred during testing setup or execution: {ex}", exc_info=True) 