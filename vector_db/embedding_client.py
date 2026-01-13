import time
import requests
from typing import List, Callable, Optional
from loguru import logger
from config.settings import settings
from config.embedding_config import EmbeddingConfig
import threading


class _OpenAIEmbeddingAPI:
    """OpenAI Embedding API 客户端 - 内部单例实现"""

    def __init__(
        self,
        base_url: str = None,
        token: str = None,
        model: str = None,
        max_batch_size: int = 100,
    ):
        # 初始化embedding配置
        embedding_config = EmbeddingConfig()
        provider_name, model_name = embedding_config.get_default_model()
        model_info = embedding_config.get_model_info(provider_name, model_name)

        # 使用新配置系统获取配置,允许通过参数覆盖
        self.base_url = (
            base_url or (model_info.get("api_base_url") if model_info else "")
        ).rstrip("/")
        self.token = token or (model_info.get("api_key") if model_info else "")
        self.model = model or f"{provider_name},{model_name}"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        self.max_batch_size = max_batch_size
        self.request_timeout = 120
        self.embedding_dim = None
        self._initialized = False
        self._lock = threading.Lock()

        logger.info(
            f"OpenAIEmbeddingAPI created (model: {self.model}, batch_size: {self.max_batch_size})"
        )

    def _lazy_init(self):
        """延迟初始化,在第一次实际调用时初始化"""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            logger.debug("First call, detecting embedding dimension...")
            # 在第一次调用时获取维度,而不是在初始化时
            if self.embedding_dim is None:
                self.embedding_dim = self._get_actual_embedding_dimension()
            self._initialized = True
            logger.info(
                f"OpenAIEmbeddingAPI initialized, dimension: {self.embedding_dim}"
            )

    def set_batch_size(self, batch_size: int):
        """动态设置批处理大小"""
        self._lazy_init()
        self.max_batch_size = batch_size
        logger.info(f"Embedding batch size set to: {self.max_batch_size}")

    def test_connection(self):
        """
        测试API连接(手动调用) - 通过实际调用embedding接口来测试

        不再依赖/health端点，而是直接测试/v1/embeddings端点
        这样可以同时验证连接性和获取embedding维度

        Returns:
            dict: {"success": bool, "message": str, "dimension": int}
        """
        try:
            logger.info("Testing embedding API connection...")

            # 使用一个简单的测试文本
            test_text = "测试连接"

            # 直接调用embedding接口进行测试
            test_response = self._encode_single_batch([test_text], get_dimension=True)

            # 检查返回结果
            if test_response and len(test_response) > 0:
                dimension = len(test_response[0])
                logger.info(
                    f"API connection successful, embedding dimension: {dimension}"
                )
                return {
                    "success": True,
                    "message": f"API连接成功，embedding维度: {dimension}",
                    "dimension": dimension,
                }
            else:
                logger.error("API returned empty result")
                return {
                    "success": False,
                    "message": "API返回了空结果，请检查模型配置",
                    "dimension": None,
                }

        except requests.exceptions.ConnectionError as e:
            # 连接错误（无法连接到服务器）
            error_msg = f"无法连接到embedding服务 ({self.base_url}): {str(e)}"
            logger.error(error_msg)
            return {"success": False, "message": error_msg, "dimension": None}
        except requests.exceptions.Timeout as e:
            # 超时错误
            error_msg = f"连接超时 (timeout={self.request_timeout}s)，请检查服务是否正常运行: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "message": error_msg, "dimension": None}
        except requests.exceptions.HTTPError as e:
            # HTTP错误（4xx, 5xx）
            error_msg = f"HTTP错误: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "message": error_msg, "dimension": None}
        except Exception as e:
            # 其他未预期的错误
            error_msg = f"测试连接时发生错误: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "message": error_msg, "dimension": None}

    def _get_actual_embedding_dimension(self) -> int:
        """通过实际调用API获取embedding维度"""
        try:
            logger.debug("Detecting embedding dimension...")
            test_response = self._encode_single_batch(["测试文本"], get_dimension=True)
            if test_response and len(test_response) > 0:
                actual_dim = len(test_response[0])
                logger.info(f"Detected embedding dimension: {actual_dim}")
                return actual_dim
            else:
                logger.warning(
                    "Could not detect embedding dimension, using default 1024"
                )
                return 1024
        except Exception as e:
            logger.warning(
                f"Failed to detect embedding dimension: {e}, using default 1024"
            )
            return 1024

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """批量编码文本为向量"""
        self._lazy_init()
        if not texts:
            return []

        all_embeddings = []
        total_batches = (len(texts) + self.max_batch_size - 1) // self.max_batch_size

        logger.info(
            f"Starting batch encoding: {len(texts)} texts, {total_batches} batches"
        )

        for i in range(0, len(texts), self.max_batch_size):
            batch_texts = texts[i : i + self.max_batch_size]
            batch_num = i // self.max_batch_size + 1

            logger.debug(
                f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)"
            )
            batch_embeddings = self._encode_single_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)

            if i + self.max_batch_size < len(texts):
                time.sleep(0.2)

        logger.info(f"Batch encoding complete: {len(all_embeddings)} vectors generated")
        return all_embeddings

    def encode_texts_with_progress(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[List[float]]:
        """批量编码文本为向量（带进度回调的新方法）"""
        self._lazy_init()
        if not texts:
            return []

        all_embeddings = []
        total_batches = (len(texts) + self.max_batch_size - 1) // self.max_batch_size

        # 初始进度回调
        if progress_callback:
            progress_callback(
                0,
                total_batches,
                f"开始向量化 {len(texts)} 个文本，分为 {total_batches} 个批次",
            )

        logger.info(
            f"Starting batch encoding: {len(texts)} texts, {total_batches} batches"
        )

        for i in range(0, len(texts), self.max_batch_size):
            batch_texts = texts[i : i + self.max_batch_size]
            batch_num = i // self.max_batch_size + 1

            # 批次开始回调
            if progress_callback:
                progress_callback(
                    batch_num - 1,
                    total_batches,
                    f"正在处理第 {batch_num}/{total_batches} 批次 ({len(batch_texts)} 个文本)",
                )

            logger.debug(
                f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)"
            )

            start_time = time.time()
            batch_embeddings = self._encode_single_batch(batch_texts)
            end_time = time.time()

            all_embeddings.extend(batch_embeddings)

            # 批次完成回调
            if progress_callback:
                progress_callback(
                    batch_num,
                    total_batches,
                    f"第 {batch_num}/{total_batches} 批次完成 ({end_time - start_time:.2f}秒)",
                )

            if i + self.max_batch_size < len(texts):
                time.sleep(0.2)

        logger.info(f"Batch encoding complete: {len(all_embeddings)} vectors generated")

        # 最终完成回调
        if progress_callback:
            progress_callback(
                total_batches,
                total_batches,
                f"向量化完成，共处理 {len(all_embeddings)} 个向量",
            )

        return all_embeddings

    def _encode_single_batch(
        self, texts: List[str], get_dimension: bool = False
    ) -> List[List[float]]:
        """
        编码单个批次的文本

        Args:
            texts: 要编码的文本列表
            get_dimension: 是否用于获取维度（测试连接时使用），为True时失败会抛出异常

        Returns:
            向量列表

        Raises:
            当get_dimension=True且失败时抛出异常
        """
        payload = {"model": self.model, "input": texts, "encoding_format": "float"}

        max_retries = 5
        last_error = None

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/v1/embeddings",
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout,
                )
                end_time = time.time()

                if response.status_code == 200:
                    data = response.json()
                    embeddings = [item["embedding"] for item in data["data"]]
                    if not get_dimension:
                        logger.debug(f"Batch completed in {end_time - start_time:.2f}s")
                    return embeddings
                else:
                    error_msg = f"API请求失败: {response.status_code} - {response.text}"
                    last_error = Exception(error_msg)
                    if attempt == max_retries - 1:
                        # 最后一次重试失败，抛出异常
                        raise last_error
                    else:
                        logger.warning(
                            f"{error_msg}, retrying ({attempt + 1}/{max_retries})"
                        )
                        time.sleep(2**attempt)

            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt == max_retries - 1:
                    # 最后一次重试失败，抛出异常
                    raise requests.exceptions.Timeout("API请求超时")
                else:
                    logger.warning(
                        f"API request timeout, retrying ({attempt + 1}/{max_retries})"
                    )
                    time.sleep(2**attempt)
            except requests.exceptions.ConnectionError as e:
                # 连接错误通常不需要重试，直接抛出
                last_error = e
                raise
            except requests.exceptions.RequestException as e:
                # 其他requests相关错误
                last_error = e
                if attempt == max_retries - 1:
                    raise
                else:
                    logger.warning(
                        f"Request exception: {e}, retrying ({attempt + 1}/{max_retries})"
                    )
                    time.sleep(2**attempt)
            except Exception as e:
                # 其他未预期的异常（如JSON解析错误）
                last_error = e
                if attempt == max_retries - 1:
                    raise Exception(f"API请求异常: {e}")
                else:
                    logger.warning(
                        f"API exception: {e}, retrying ({attempt + 1}/{max_retries})"
                    )
                    time.sleep(2**attempt)

        # 如果是测试连接（get_dimension=True），失败时抛出异常
        if get_dimension and last_error:
            logger.error("Batch encoding failed")
            raise last_error

        # 正常向量化流程失败时返回零向量（容错处理）
        logger.error("Batch encoding failed, returning zero vectors")
        fallback_dim = getattr(self, "embedding_dim", 1024)
        return [[0.0] * fallback_dim for _ in texts]


# 模块级单例实例
_embedding_instance = None
_embedding_lock = threading.Lock()


class OpenAIEmbeddingAPI:
    """OpenAIEmbeddingAPI 的代理类，确保始终返回同一个实例"""

    def __new__(cls, *args, **kwargs):
        global _embedding_instance
        if _embedding_instance is None:
            with _embedding_lock:
                if _embedding_instance is None:
                    _embedding_instance = _OpenAIEmbeddingAPI(*args, **kwargs)
        return _embedding_instance
