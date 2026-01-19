try:
    from pymilvus import MilvusClient, DataType
except ImportError:
    MilvusClient = None
    DataType = None

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import uuid
import numpy as np
from loguru import logger
from .base import VectorDBInterface
from .embedding_client import OpenAIEmbeddingAPI
from config.settings import settings


class MilvusDBClient(VectorDBInterface):
    """Milvus向量数据库客户端实现"""

    DEFAULT_VECTOR_DIM = 1024
    SYSTEM_FIELDS = {"id", "embedding", "dense_vector", "document"}

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
    ):
        if MilvusClient is None:
            raise ImportError("pymilvus is not installed. Run: pip install pymilvus")

        self.host = host or settings.milvus_host
        self.port = port or settings.milvus_port
        self.user = user or settings.milvus_user
        self.password = password or settings.milvus_password

        uri = f"http://{self.host}:{self.port}"
        self.client = MilvusClient(uri=uri, token=f"{self.user}:{self.password}")
        self.embedding_api = OpenAIEmbeddingAPI()
        self.vector_dim = self.embedding_api.embedding_dim or self.DEFAULT_VECTOR_DIM

        logger.info(f"Milvus client initialized at {uri}")
        logger.info(f"Embedding dimension: {self.vector_dim}")

    def _calculate_cosine_similarity(
        self, vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]
    ) -> float:
        try:
            arr1 = np.array(vec1, dtype=np.float32)
            arr2 = np.array(vec2, dtype=np.float32)

            dot_product = np.dot(arr1, arr2)
            norm_v1 = np.linalg.norm(arr1)
            norm_v2 = np.linalg.norm(arr2)

            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0

            cosine_similarity = dot_product / (norm_v1 * norm_v2)
            return float(np.clip(cosine_similarity, -1.0, 1.0))

        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0

    def create_collection(self, collection_name: str, **kwargs) -> bool:
        try:
            if self.has_collection(collection_name):
                logger.warning(f"Collection '{collection_name}' already exists, deleting...")
                self.delete_collection(collection_name)

            dimension = kwargs.get("dimension", self.vector_dim)

            self.client.create_collection(
                collection_name=collection_name,
                dimension=dimension,
                metric_type="COSINE",
                auto_id=False,
                id_type="string",
                max_length=256,
            )

            logger.info(f"Collection '{collection_name}' created successfully with dimension {dimension}")
            return True

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        try:
            self.client.drop_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    def list_collections(self) -> List[str]:
        try:
            collections = self.client.list_collections()
            return collections
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def has_collection(self, collection_name: str) -> bool:
        try:
            return self.client.has_collection(collection_name=collection_name)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False

    def insert_data(self, collection_name: str, data: List[Dict[str, Any]]) -> bool:
        try:
            if not data:
                logger.warning("No data to insert")
                return True

            insert_data = []

            for item in data:
                record: Dict[str, Any] = {}

                item_id = item.get("id", str(uuid.uuid4()))
                record["id"] = str(item_id)

                if "embedding" in item:
                    vector = item["embedding"]
                elif "dense_vector" in item:
                    vector = item["dense_vector"]
                else:
                    raise ValueError("数据中缺少embedding或dense_vector字段")

                if isinstance(vector, np.ndarray):
                    vector = vector.tolist()
                record["vector"] = vector

                if "document" in item:
                    record["document"] = str(item["document"])
                else:
                    doc_parts = []
                    for key, value in item.items():
                        if key not in self.SYSTEM_FIELDS and isinstance(value, str):
                            doc_parts.append(f"{key}: {value}")
                    record["document"] = " | ".join(doc_parts)

                metadata: Dict[str, Any] = {}
                for key, value in item.items():
                    if key not in self.SYSTEM_FIELDS and key not in ["vector"]:
                        if isinstance(value, (str, int, float, bool)):
                            metadata[key] = value
                        elif value is None:
                            metadata[key] = ""
                        else:
                            metadata[key] = str(value)

                record["metadata"] = metadata
                insert_data.append(record)

            self.client.insert(
                collection_name=collection_name,
                data=insert_data
            )

            logger.info(f"Inserted {len(data)} records into '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to insert data: {e}")
            return False

    def query_by_vector(
        self, collection_name: str, query_vector: List[float], top_k: int = 10, **kwargs
    ) -> List[Dict[str, Any]]:
        try:
            search_vector: List[float] = query_vector
            if isinstance(query_vector, np.ndarray):
                search_vector = query_vector.tolist()

            results = self.client.search(
                collection_name=collection_name,
                data=[search_vector],
                limit=top_k,
                output_fields=["id", "document", "metadata", "vector"],
            )

            formatted_results: List[Dict[str, Any]] = []

            if results and len(results) > 0:
                for hit in results[0]:
                    entity = hit.get("entity", {})
                    result: Dict[str, Any] = {
                        "id": hit.get("id", ""),
                        "document": entity.get("document", ""),
                        "metadata": entity.get("metadata", {}),
                        "distance": hit.get("distance", 0.0),
                        "similarity": 0.0,
                        "cosine_similarity": 0.0,
                    }

                    # COSINE metric: Milvus returns similarity [0,1] where 1=most similar
                    distance = hit.get("distance", 0.0)

                    if 0 <= distance <= 1:
                        result["cosine_similarity"] = distance
                        result["similarity"] = distance
                    else:
                        result["cosine_similarity"] = 1 - distance
                        result["similarity"] = 1 - distance

                    doc_vector = entity.get("vector")
                    if doc_vector:
                        result["cosine_similarity"] = self._calculate_cosine_similarity(
                            search_vector, doc_vector
                        )

                    formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"Vector query failed: {e}")
            return []

    def query_by_ids(self, collection_name: str, ids: List[str]) -> List[Dict[str, Any]]:
        try:
            str_ids = [str(id) for id in ids]

            results = self.client.get(
                collection_name=collection_name,
                ids=str_ids,
                output_fields=["id", "document", "metadata", "vector"]
            )

            formatted_results: List[Dict[str, Any]] = []

            for item in results:
                result: Dict[str, Any] = {
                    "id": item.get("id", ""),
                    "document": item.get("document", ""),
                    "metadata": item.get("metadata", {}),
                    "embedding": item.get("vector", []),
                }
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"ID query failed: {e}")
            return []

    def get_all_data(self, collection_name: str, limit: int = 1000) -> List[Dict[str, Any]]:
        try:
            results = self.client.query(
                collection_name=collection_name,
                filter="",
                output_fields=["id", "document", "metadata", "vector"],
                limit=limit
            )

            formatted_results: List[Dict[str, Any]] = []

            for item in results:
                result: Dict[str, Any] = {
                    "id": item.get("id", ""),
                    "document": item.get("document", ""),
                    "embedding": item.get("vector", []),
                    "dense_vector": item.get("vector", []),
                }

                metadata = item.get("metadata", {})
                if isinstance(metadata, dict):
                    for key, value in metadata.items():
                        if key not in result:
                            result[key] = value

                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to get all data: {e}")
            return []

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        try:
            stats = self.client.get_collection_stats(collection_name=collection_name)
            row_count = stats.get("row_count", 0)

            dimension = self.vector_dim
            desc: Dict[str, Any] = {}
            try:
                desc = self.client.describe_collection(collection_name=collection_name)
                for field in desc.get("fields", []):
                    if DataType and field.get("type") == DataType.FLOAT_VECTOR:
                        dimension = field.get("params", {}).get("dim", self.vector_dim)
                        break
            except Exception:
                pass

            return {
                "row_count": row_count,
                "name": collection_name,
                "dimension": dimension,
                "metadata": {
                    "created_at": desc.get("created_timestamp", ""),
                    "description": f"Milvus collection: {collection_name}",
                },
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e), "row_count": 0, "name": collection_name}

    def get_collection_fields(self, collection_name: str) -> List[str]:
        try:
            sample_data = self.get_all_data(collection_name, limit=1)
            if sample_data:
                fields = list(sample_data[0].keys())
                filtered_fields = [
                    f for f in fields if f not in ["id", "embedding", "dense_vector", "vector"]
                ]
                return filtered_fields
            else:
                return ["document"]

        except Exception as e:
            logger.error(f"Failed to get fields: {e}")
            return []

    def update_data(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Milvus不支持直接更新，通过删除+插入实现"""
        try:
            existing_data = self.query_by_ids(collection_name, ids)

            if not existing_data:
                logger.warning(f"No existing data found for ids: {ids}")
                return False

            update_data: List[Dict[str, Any]] = []

            for i, item_id in enumerate(ids):
                existing: Optional[Dict[str, Any]] = None
                for data in existing_data:
                    if data.get("id") == str(item_id):
                        existing = data
                        break

                if existing is None:
                    logger.warning(f"ID {item_id} not found, skipping")
                    continue

                record: Dict[str, Any] = {"id": str(item_id)}

                if embeddings is not None and i < len(embeddings):
                    vector = embeddings[i]
                    if isinstance(vector, np.ndarray):
                        vector = vector.tolist()
                    record["vector"] = vector
                else:
                    record["vector"] = existing.get("embedding", [])

                if documents is not None and i < len(documents):
                    record["document"] = documents[i]
                else:
                    record["document"] = existing.get("document", "")

                if metadatas is not None and i < len(metadatas):
                    metadata: Dict[str, Any] = {}
                    for key, value in metadatas[i].items():
                        if isinstance(value, (str, int, float, bool)):
                            metadata[key] = value
                        elif value is None:
                            metadata[key] = ""
                        else:
                            metadata[key] = str(value)
                    record["metadata"] = metadata
                else:
                    record["metadata"] = existing.get("metadata", {})

                update_data.append(record)

            if not update_data:
                logger.warning("No data to update")
                return False

            self.delete_by_ids(collection_name, ids)

            self.client.insert(
                collection_name=collection_name,
                data=update_data
            )

            logger.info(f"Updated {len(update_data)} records in '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to update data: {e}")
            return False

    def delete_by_ids(self, collection_name: str, ids: List[str]) -> bool:
        try:
            str_ids = [str(id) for id in ids]

            self.client.delete(
                collection_name=collection_name,
                ids=str_ids
            )

            logger.info(f"Deleted {len(ids)} records from '{collection_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to delete data: {e}")
            return False
