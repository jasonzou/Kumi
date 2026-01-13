import time
import traceback
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from vector_db.factory import VectorDBFactory
from vector_db.embedding_client import OpenAIEmbeddingAPI
from config.settings import settings
from config.embedding_config import EmbeddingConfig


class SimilarityCalculator:
    """相似度计算引擎"""

    def __init__(self, db_type: str = None):
        """初始化相似度计算引擎"""

        # 初始化向量数据库客户端
        self.db_client = VectorDBFactory.create_client(db_type)

        # 初始化embedding API
        self.embedding_api = OpenAIEmbeddingAPI()

        # 初始化embedding配置
        self.embedding_config = EmbeddingConfig()

        logger.info("Similarity calculator initialized")
        logger.info(f"Database type: {settings.vector_db_type}")

    def test_connection(self) -> Dict[str, Any]:
        """测试连接状态"""
        try:
            # 测试数据库连接
            collections = self.db_client.list_collections()
            db_status = "connected"
            collections_count = len(collections)
        except Exception as e:
            db_status = f"failed: {str(e)}"
            collections_count = 0

        # 测试Embedding API连接
        try:
            test_embedding = self.embedding_api.encode_texts(["test"])
            embedding_status = "connected" if test_embedding else "failed"
        except Exception as e:
            embedding_status = f"failed: {str(e)}"

        # 获取默认模型信息
        provider_name, model_name = self.embedding_config.get_default_model()
        model_info = self.embedding_config.get_model_info(provider_name, model_name)

        return {
            "vector_db": {
                "type": settings.vector_db_type,
                "status": db_status,
                "collections_count": collections_count,
            },
            "embedding_api": {
                "status": embedding_status,
                "base_url": model_info.get("api_base_url", "") if model_info else "",
                "model": f"{provider_name},{model_name}",
            },
        }

    def get_collections(self) -> List[str]:
        """获取所有collection列表"""
        return self.db_client.list_collections()

    def get_collection_fields(self, collection_name: str) -> List[str]:
        """获取collection的字段列表"""
        return self.db_client.get_collection_fields(collection_name)

    def get_collection_data(
        self, collection_name: str, limit: int = 1000
    ) -> List[Dict]:
        """获取collection的数据"""
        return self.db_client.get_all_data(collection_name, limit)

    def calculate_cosine_similarity(
        self, vector1: List[float], vector2: List[float]
    ) -> float:
        """计算两个向量的余弦相似度"""
        try:
            v1 = np.array(vector1)
            v2 = np.array(vector2)

            # 检查向量维度是否匹配
            if v1.shape != v2.shape:
                raise ValueError(f"向量维度不匹配: {v1.shape} vs {v2.shape}")

            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0

            similarity = dot_product / (norm_v1 * norm_v2)

            # 处理浮点数精度误差，确保结果在[-1, 1]范围内
            similarity = max(-0.0, min(1.0, similarity))

            return float(similarity)

        except ValueError:
            # 重新抛出维度不匹配等值错误
            raise
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            raise ValueError(f"计算相似度失败: {str(e)}")

    def calculate_similarity_matrix(
        self,
        x_collection: str,
        y_collection: str,
        x_max_items: int = 100,
        y_max_items: int = 100,
    ) -> Dict[str, Any]:
        """计算两个collection之间的相似度矩阵"""
        try:
            start_time = time.time()

            logger.info(f"Starting similarity matrix calculation")
            logger.info(f"X Collection: {x_collection} (max items: {x_max_items})")
            logger.info(f"Y Collection: {y_collection} (max items: {y_max_items})")

            # 获取数据
            logger.info("Fetching data...")
            x_data = self.get_collection_data(x_collection, x_max_items)
            y_data = self.get_collection_data(y_collection, y_max_items)

            if not x_data or not y_data:
                raise ValueError("数据为空")

            logger.info(f"X data count: {len(x_data)}")
            logger.info(f"Y data count: {len(y_data)}")

            # 提取向量
            x_vectors = []
            y_vectors = []

            for item in x_data:
                if "dense_vector" in item:
                    x_vectors.append(item["dense_vector"])
                elif "embedding" in item:
                    x_vectors.append(item["embedding"])
                else:
                    raise ValueError("X数据中缺少向量字段")

            for item in y_data:
                if "dense_vector" in item:
                    y_vectors.append(item["dense_vector"])
                elif "embedding" in item:
                    y_vectors.append(item["embedding"])
                else:
                    raise ValueError("Y数据中缺少向量字段")

            # 检查向量维度是否一致
            if x_vectors and y_vectors:
                x_dim = len(x_vectors[0])
                y_dim = len(y_vectors[0])
                if x_dim != y_dim:
                    raise ValueError(
                        f"向量维度不匹配: {x_collection} 的向量维度为 {x_dim}, "
                        f"{y_collection} 的向量维度为 {y_dim}。"
                        f"无法计算不同维度向量之间的相似度。"
                    )

            # 计算相似度矩阵
            logger.info("Computing similarity matrix...")
            similarity_matrix = []
            total_comparisons = len(y_data) * len(x_data)
            completed = 0

            for i, y_vector in enumerate(y_vectors):
                row = []
                for j, x_vector in enumerate(x_vectors):
                    similarity = self.calculate_cosine_similarity(y_vector, x_vector)
                    row.append(similarity)
                    completed += 1

                    if completed % 1000 == 0:
                        progress = (completed / total_comparisons) * 100
                        logger.info(
                            f"Progress: {progress:.1f}% ({completed}/{total_comparisons})"
                        )

                similarity_matrix.append(row)

            # 计算统计信息
            flat_similarities = [sim for row in similarity_matrix for sim in row]
            stats = {
                "total_pairs": len(flat_similarities),
                "avg_similarity": float(np.mean(flat_similarities)),
                "min_similarity": float(np.min(flat_similarities)),
                "max_similarity": float(np.max(flat_similarities)),
                "std_similarity": float(np.std(flat_similarities)),
                "high_similarity_count": sum(
                    1 for sim in flat_similarities if sim > 0.8
                ),
                "medium_similarity_count": sum(
                    1 for sim in flat_similarities if 0.5 <= sim <= 0.8
                ),
                "low_similarity_count": sum(
                    1 for sim in flat_similarities if sim < 0.5
                ),
                "compute_time": (time.time() - start_time) * 1000,
            }

            end_time = time.time()
            logger.info(
                f"Similarity matrix calculation complete in {end_time - start_time:.2f}s"
            )

            # 处理数据
            x_data_processed = []
            for i, item in enumerate(x_data):
                item_copy = {
                    k: v
                    for k, v in item.items()
                    if k not in ["dense_vector", "embedding"]
                }
                item_copy["order_id"] = i
                x_data_processed.append(item_copy)

            y_data_processed = []
            for i, item in enumerate(y_data):
                item_copy = {
                    k: v
                    for k, v in item.items()
                    if k not in ["dense_vector", "embedding"]
                }
                item_copy["order_id"] = i
                y_data_processed.append(item_copy)

            # 获取可用字段列表
            x_available_fields = [
                k for k in x_data_processed[0].keys() if k not in ["id"]
            ]
            y_available_fields = [
                k for k in y_data_processed[0].keys() if k not in ["id"]
            ]

            return {
                "matrix": similarity_matrix,
                "x_data": x_data_processed,
                "y_data": y_data_processed,
                "x_available_fields": x_available_fields,
                "y_available_fields": y_available_fields,
                "stats": stats,
                "metadata": {
                    "x_collection": x_collection,
                    "y_collection": y_collection,
                    "x_max_items": x_max_items,
                    "y_max_items": y_max_items,
                    "calculation_time": end_time - start_time,
                },
            }

        except Exception as e:
            logger.error(f"Similarity matrix calculation failed: {e}")
            traceback.print_exc()
            raise

    def find_similar_items(
        self, collection_name: str, query_text: str, field_name: str, top_k: int = 10
    ) -> List[Dict]:
        """查找与查询文本最相似的项目"""
        try:
            # 将查询文本向量化
            query_vectors = self.embedding_api.encode_texts([query_text])
            if not query_vectors:
                raise ValueError("查询文本向量化失败")

            query_vector = query_vectors[0]

            # 在数据库中搜索
            search_results = self.db_client.query_by_vector(
                collection_name=collection_name, query_vector=query_vector, top_k=top_k
            )

            # 格式化结果
            results = []
            for result in search_results:
                formatted_result = {
                    "id": result.get("id", ""),
                    "text": result.get(field_name, result.get("document", "")),
                    "similarity": result.get("similarity", 0.0),
                    "distance": result.get("distance", 0.0),
                    "metadata": result.get("metadata", {}),
                }
                results.append(formatted_result)

            return results

        except Exception as e:
            logger.error(f"Similar item search failed: {e}")
            raise

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """获取collection统计信息"""
        return self.db_client.get_collection_stats(collection_name)

    def validate_request_data(
        self, data: Dict[str, Any], required_fields: List[str]
    ) -> Optional[str]:
        """验证请求数据"""
        for field in required_fields:
            if field not in data:
                return f"缺少必需字段: {field}"
            if not data[field]:
                return f"字段 {field} 不能为空"
        return None
