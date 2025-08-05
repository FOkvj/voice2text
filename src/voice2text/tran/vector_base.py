# ============================================================================
# 向量数据库接口和实现模块
# ============================================================================

import asyncio
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np


class VectorDBType(Enum):
    """向量数据库类型枚举"""
    CHROMADB = "chromadb"
    MILVUS = "milvus"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"

# if __name__ == "__main__":
#     print(VectorDBType.CHROMADB == VectorDBType.CHROMADB)


@dataclass
class VectorDBConfig:
    """向量数据库配置基类"""
    db_type: VectorDBType
    collection_name: str = "voice_prints"
    dimension: int = 192  # ECAPA-TDNN embedding dimension
    distance_metric: str = "cosine"  # cosine, euclidean, dot_product
    connection_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChromaDBConfig(VectorDBConfig):
    """ChromaDB配置"""
    persist_directory: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MilvusConfig(VectorDBConfig):
    """Milvus配置"""
    host: str = "localhost"
    port: int = 19530
    username: Optional[str] = None
    password: Optional[str] = None
    secure: bool = False
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"
    nlist: int = 128


@dataclass
class VoicePrintRecord:
    """声纹记录数据结构"""
    id: str
    speaker_id: str
    embedding: np.ndarray
    sample_number: int
    audio_duration: float
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'speaker_id': self.speaker_id,
            'embedding': self.embedding.tolist(),
            'sample_number': self.sample_number,
            'audio_duration': self.audio_duration,
            'created_at': self.created_at,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoicePrintRecord':
        """从字典创建记录"""
        return cls(
            id=data['id'],
            speaker_id=data['speaker_id'],
            embedding=np.array(data['embedding']),
            sample_number=data['sample_number'],
            audio_duration=data['audio_duration'],
            created_at=datetime.fromisoformat(data['created_at']),
            metadata=data.get('metadata', {})
        )


class VectorDatabaseInterface(ABC):
    """向量数据库抽象接口"""

    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def connect(self) -> None:
        """连接到数据库"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """断开数据库连接"""
        pass

    @abstractmethod
    async def create_collection(self) -> bool:
        """创建集合/表"""
        pass

    @abstractmethod
    async def collection_exists(self) -> bool:
        """检查集合是否存在"""
        pass

    @abstractmethod
    async def insert_vector(self, record: VoicePrintRecord) -> bool:
        """插入向量记录"""
        pass

    @abstractmethod
    async def batch_insert_vectors(self, records: List[VoicePrintRecord]) -> bool:
        """批量插入向量记录"""
        pass

    @abstractmethod
    async def search_similar_vectors(self,
                                     query_vector: np.ndarray,
                                     top_k: int = 10,
                                     threshold: Optional[float] = None,
                                     filters: Optional[Dict[str, Any]] = None) -> List[Tuple[VoicePrintRecord, float]]:
        """搜索相似向量"""
        pass

    @abstractmethod
    async def get_vector_by_id(self, vector_id: str) -> Optional[VoicePrintRecord]:
        """根据ID获取向量记录"""
        pass

    @abstractmethod
    async def get_vectors_by_speaker(self, speaker_id: str) -> List[VoicePrintRecord]:
        """根据说话人ID获取所有向量记录"""
        pass

    @abstractmethod
    async def update_vector(self, vector_id: str, updates: Dict[str, Any]) -> bool:
        """更新向量记录"""
        pass

    @abstractmethod
    async def delete_vector(self, vector_id: str) -> bool:
        """删除向量记录"""
        pass

    @abstractmethod
    async def delete_vectors_by_speaker(self, speaker_id: str) -> int:
        """删除说话人的所有向量记录"""
        pass

    @abstractmethod
    async def count_vectors(self, speaker_id: Optional[str] = None) -> int:
        """统计向量数量"""
        pass

    @abstractmethod
    async def list_speakers(self) -> Dict[str, List]:
        """列出所有说话人ID"""
        pass

    @abstractmethod
    async def clear_collection(self) -> bool:
        """清空集合"""
        pass

    # 上下文管理器支持
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()


class ChromaDBImplementation(VectorDatabaseInterface):
    """ChromaDB实现"""

    def __init__(self, config: ChromaDBConfig):
        super().__init__(config)
        self.chroma_config = config
        self.client = None
        self.collection = None

    async def connect(self) -> None:
        """连接到ChromaDB"""
        try:
            import chromadb
            from chromadb.config import Settings

            # 构建ChromaDB客户端
            if self.chroma_config.host and self.chroma_config.port:
                # 远程连接
                self.client = chromadb.HttpClient(
                    host=self.chroma_config.host,
                    port=self.chroma_config.port,
                    settings=Settings(**self.chroma_config.settings)
                )
            else:
                # 本地持久化
                persist_dir = self.chroma_config.persist_directory or "./chroma_db"
                settings = Settings(persist_directory=persist_dir, **self.chroma_config.settings)
                self.client = chromadb.PersistentClient(settings=settings)

            self.logger.info("Connected to ChromaDB successfully")

            # 创建或获取集合
            await self.create_collection()

        except ImportError:
            raise ImportError("ChromaDB is not installed. Please install with: pip install chromadb")
        except Exception as e:
            self.logger.error(f"Failed to connect to ChromaDB: {e}")
            raise

    async def disconnect(self) -> None:
        """断开ChromaDB连接"""
        if self.client:
            # ChromaDB通常不需要显式关闭连接
            self.client = None
            self.collection = None
            self.logger.info("Disconnected from ChromaDB")

    async def create_collection(self) -> bool:
        """创建ChromaDB集合"""
        try:
            # ChromaDB的distance函数映射
            distance_mapping = {
                "cosine": "cosine",
                "euclidean": "l2",
                "dot_product": "ip"
            }

            distance_function = distance_mapping.get(
                self.config.distance_metric, "cosine"
            )

            # 创建或获取集合
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={
                    "hnsw:space": distance_function,
                    "dimension": self.config.dimension
                }
            )

            self.logger.info(f"Collection '{self.config.collection_name}' ready")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create collection: {e}")
            return False

    async def collection_exists(self) -> bool:
        """检查集合是否存在"""
        try:
            collections = self.client.list_collections()
            return any(col.name == self.config.collection_name for col in collections)
        except Exception as e:
            self.logger.error(f"Failed to check collection existence: {e}")
            return False

    async def insert_vector(self, record: VoicePrintRecord) -> bool:
        """插入向量记录到ChromaDB"""
        try:
            # 准备元数据
            metadata = {
                'speaker_id': record.speaker_id,
                'sample_number': record.sample_number,
                'audio_duration': record.audio_duration,
                'created_at': record.created_at,
                **record.metadata
            }

            # 插入到ChromaDB
            self.collection.add(
                ids=[record.id],
                embeddings=[record.embedding.tolist()],
                metadatas=[metadata]
            )

            self.logger.debug(f"Inserted vector record {record.id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to insert vector {record.id}: {e}")
            return False

    async def batch_insert_vectors(self, records: List[VoicePrintRecord]) -> bool:
        """批量插入向量记录"""
        try:
            if not records:
                return True

            ids = []
            embeddings = []
            metadatas = []

            for record in records:
                ids.append(record.id)
                embeddings.append(record.embedding.tolist())

                metadata = {
                    'speaker_id': record.speaker_id,
                    'sample_number': record.sample_number,
                    'audio_duration': record.audio_duration,
                    'created_at': record.created_at.isoformat(),
                    **record.metadata
                }
                metadatas.append(metadata)

            # 批量插入
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )

            self.logger.info(f"Batch inserted {len(records)} vector records")
            return True

        except Exception as e:
            self.logger.error(f"Failed to batch insert vectors: {e}")
            return False

    async def search_similar_vectors(self,
                                     query_vector: np.ndarray,
                                     top_k: int = 10,
                                     threshold: Optional[float] = None,
                                     filters: Optional[Dict[str, Any]] = None) -> List[Tuple[VoicePrintRecord, float]]:
        """搜索相似向量"""
        try:
            # 准备where条件
            where_clause = {}
            if filters:
                where_clause.update(filters)

            # 执行查询
            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=top_k,
                where=where_clause if where_clause else None
            )

            # 处理结果
            similar_vectors = []
            if results['ids'] and results['ids'][0]:
                for i, vector_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i]
                    metadata = results['metadatas'][0][i]

                    similarity = 1 - distance

                    # 应用阈值过滤
                    # if threshold is None or similarity >= threshold:
                    record = VoicePrintRecord(
                        id=vector_id,
                        speaker_id=metadata['speaker_id'],
                        embedding=None,
                        sample_number=metadata['sample_number'],
                        audio_duration=metadata['audio_duration'],
                        created_at=datetime.fromisoformat(metadata['created_at']),
                        metadata={k: v for k, v in metadata.items()
                                  if k not in ['speaker_id', 'sample_number', 'audio_duration', 'created_at']}
                    )
                    similar_vectors.append((record, similarity))

            return similar_vectors

        except Exception as e:
            self.logger.error(f"Failed to search similar vectors: {e}")
            return []

    async def get_vector_by_id(self, vector_id: str) -> Optional[VoicePrintRecord]:
        """根据ID获取向量记录"""
        try:
            results = self.collection.get(ids=[vector_id], include=['embeddings', 'metadatas'])

            if results['ids'] and results['ids'][0] == vector_id:
                metadata = results['metadatas'][0]
                embedding = np.array(results['embeddings'][0])

                return VoicePrintRecord(
                    id=vector_id,
                    speaker_id=metadata['speaker_id'],
                    embedding=embedding,
                    sample_number=metadata['sample_number'],
                    audio_duration=metadata['audio_duration'],
                    created_at=datetime.fromisoformat(metadata['created_at']),
                    metadata={k: v for k, v in metadata.items()
                              if k not in ['speaker_id', 'sample_number', 'audio_duration', 'created_at']}
                )

            return None

        except Exception as e:
            self.logger.error(f"Failed to get vector {vector_id}: {e}")
            return None

    async def get_vectors_by_speaker(self, speaker_id: str) -> List[VoicePrintRecord]:
        """根据说话人ID获取所有向量记录"""
        try:
            results = self.collection.get(
                where={"speaker_id": speaker_id},
                include=['embeddings', 'metadatas']
            )

            records = []
            if results['ids']:
                for i, vector_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]
                    embedding = np.array(results['embeddings'][i])

                    record = VoicePrintRecord(
                        id=vector_id,
                        speaker_id=metadata['speaker_id'],
                        embedding=embedding,
                        sample_number=metadata['sample_number'],
                        audio_duration=metadata['audio_duration'],
                        created_at=datetime.fromisoformat(metadata['created_at']),
                        metadata={k: v for k, v in metadata.items()
                                  if k not in ['speaker_id', 'sample_number', 'audio_duration', 'created_at']}
                    )
                    records.append(record)

            return records

        except Exception as e:
            self.logger.error(f"Failed to get vectors for speaker {speaker_id}: {e}")
            return []

    async def update_vector(self, vector_id: str, updates: Dict[str, Any]) -> bool:
        """更新向量记录"""
        try:
            # ChromaDB的更新需要先获取现有记录，然后重新插入
            existing_record = await self.get_vector_by_id(vector_id)
            if not existing_record:
                return False

            # 应用更新
            if 'speaker_id' in updates:
                existing_record.speaker_id = updates['speaker_id']
            if 'sample_number' in updates:
                existing_record.sample_number = updates['sample_number']
            if 'audio_duration' in updates:
                existing_record.audio_duration = updates['audio_duration']
            if 'metadata' in updates:
                existing_record.metadata.update(updates['metadata'])

            # 删除旧记录并插入新记录
            self.collection.delete(ids=[vector_id])
            return await self.insert_vector(existing_record)

        except Exception as e:
            self.logger.error(f"Failed to update vector {vector_id}: {e}")
            return False

    async def delete_vector(self, vector_id: str) -> bool:
        """删除向量记录"""
        try:
            self.collection.delete(ids=[vector_id])
            self.logger.debug(f"Deleted vector {vector_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete vector {vector_id}: {e}")
            return False

    async def delete_vectors_by_speaker(self, speaker_id: str) -> int:
        """删除说话人的所有向量记录"""
        try:
            # 先获取该说话人的所有记录
            records = await self.get_vectors_by_speaker(speaker_id)

            if records:
                ids_to_delete = [record.id for record in records]
                self.collection.delete(ids=ids_to_delete)
                self.logger.info(f"Deleted {len(ids_to_delete)} vectors for speaker {speaker_id}")
                return len(ids_to_delete)

            return 0

        except Exception as e:
            self.logger.error(f"Failed to delete vectors for speaker {speaker_id}: {e}")
            return 0

    async def count_vectors(self, speaker_id: Optional[str] = None) -> int:
        """统计向量数量"""
        try:
            if speaker_id:
                results = self.collection.get(where={"speaker_id": speaker_id})
                return len(results['ids']) if results['ids'] else 0
            else:
                return self.collection.count()

        except Exception as e:
            self.logger.error(f"Failed to count vectors: {e}")
            return 0

    async def list_speakers(self) -> Dict[str, List]:
        """列出所有说话人ID"""
        try:
            return self.collection.get(include=['metadatas'])
        except Exception as e:
            self.logger.error(f"Failed to list speakers: {e}")
            return []

    async def clear_collection(self) -> bool:
        """清空集合"""
        try:
            # ChromaDB清空集合的方式是删除并重新创建
            self.client.delete_collection(name=self.config.collection_name)
            await self.create_collection()
            self.logger.info(f"Cleared collection {self.config.collection_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to clear collection: {e}")
            return False


class MilvusImplementation(VectorDatabaseInterface):
    """Milvus向量数据库实现"""

    def __init__(self, config: MilvusConfig):
        super().__init__(config)
        self.milvus_config = config
        self.collection = None
        self._connected = False

    async def connect(self) -> None:
        """连接到Milvus"""
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

            # 连接到Milvus
            connections.connect(
                alias="default",
                host=self.milvus_config.host,
                port=self.milvus_config.port,
                user=self.milvus_config.username,
                password=self.milvus_config.password,
                secure=self.milvus_config.secure
            )

            self._connected = True
            self.logger.info(f"Connected to Milvus at {self.milvus_config.host}:{self.milvus_config.port}")

            # 创建或获取集合
            await self.create_collection()

        except ImportError:
            raise ImportError("pymilvus is not installed. Please install with: pip install pymilvus")
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus: {e}")
            raise

    async def disconnect(self) -> None:
        """断开Milvus连接"""
        try:
            if self._connected:
                from pymilvus import connections
                connections.disconnect("default")
                self.collection = None
                self._connected = False
                self.logger.info("Disconnected from Milvus")
        except Exception as e:
            self.logger.error(f"Failed to disconnect from Milvus: {e}")

    async def create_collection(self) -> bool:
        """创建Milvus集合"""
        try:
            from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility

            # 如果集合已存在，直接获取
            if utility.has_collection(self.config.collection_name):
                self.collection = Collection(self.config.collection_name)
                self.logger.info(f"Collection '{self.config.collection_name}' already exists")

                # 加载集合到内存
                if not self.collection.has_index():
                    await self._create_index()

                self.collection.load()
                return True

            # 定义字段模式
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="speaker_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.config.dimension),
                FieldSchema(name="sample_number", dtype=DataType.INT32),
                FieldSchema(name="audio_duration", dtype=DataType.FLOAT),
                FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=1000)
            ]

            # 创建集合模式
            schema = CollectionSchema(
                fields=fields,
                description=f"Voice print collection: {self.config.collection_name}"
            )

            # 创建集合
            self.collection = Collection(
                name=self.config.collection_name,
                schema=schema
            )

            self.logger.info(f"Created collection '{self.config.collection_name}'")

            # 创建索引
            await self._create_index()

            # 加载集合到内存
            self.collection.load()

            return True

        except Exception as e:
            self.logger.error(f"Failed to create collection: {e}")
            return False

    async def _create_index(self) -> None:
        """创建向量索引"""
        try:
            # 距离度量映射
            metric_mapping = {
                "cosine": "COSINE",
                "euclidean": "L2",
                "dot_product": "IP"
            }

            metric_type = metric_mapping.get(self.config.distance_metric, "COSINE")

            # 索引参数
            index_params = {
                "metric_type": metric_type,
                "index_type": self.milvus_config.index_type,
                "params": {"nlist": self.milvus_config.nlist}
            }

            # 创建向量字段索引
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )

            # 为标量字段创建索引以提高查询性能
            self.collection.create_index(
                field_name="speaker_id",
                index_params={"index_type": "TRIE"}
            )

            self.logger.info("Created indexes successfully")

        except Exception as e:
            self.logger.error(f"Failed to create index: {e}")
            raise

    async def collection_exists(self) -> bool:
        """检查集合是否存在"""
        try:
            from pymilvus import utility
            return utility.has_collection(self.config.collection_name)
        except Exception as e:
            self.logger.error(f"Failed to check collection existence: {e}")
            return False

    async def insert_vector(self, record: VoicePrintRecord) -> bool:
        """插入向量记录到Milvus"""
        try:
            # 准备数据
            data = [
                [record.id],
                [record.speaker_id],
                [record.embedding.tolist()],
                [record.sample_number],
                [record.audio_duration],
                [record.created_at.isoformat()],
                [json.dumps(record.metadata)]
            ]

            # 插入数据
            self.collection.insert(data)

            # 刷新以确保数据持久化
            self.collection.flush()

            self.logger.debug(f"Inserted vector record {record.id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to insert vector {record.id}: {e}")
            return False

    async def batch_insert_vectors(self, records: List[VoicePrintRecord]) -> bool:
        """批量插入向量记录"""
        try:
            if not records:
                return True

            # 准备批量数据
            ids = []
            speaker_ids = []
            embeddings = []
            sample_numbers = []
            audio_durations = []
            created_ats = []
            metadata_jsons = []

            for record in records:
                ids.append(record.id)
                speaker_ids.append(record.speaker_id)
                embeddings.append(record.embedding.tolist())
                sample_numbers.append(record.sample_number)
                audio_durations.append(record.audio_duration)
                created_ats.append(record.created_at.isoformat())
                metadata_jsons.append(json.dumps(record.metadata))

            # 批量插入
            data = [
                ids,
                speaker_ids,
                embeddings,
                sample_numbers,
                audio_durations,
                created_ats,
                metadata_jsons
            ]

            self.collection.insert(data)
            self.collection.flush()

            self.logger.info(f"Batch inserted {len(records)} vector records")
            return True

        except Exception as e:
            self.logger.error(f"Failed to batch insert vectors: {e}")
            return False

    async def search_similar_vectors(self,
                                     query_vector: np.ndarray,
                                     top_k: int = 10,
                                     threshold: Optional[float] = None,
                                     filters: Optional[Dict[str, Any]] = None) -> List[Tuple[VoicePrintRecord, float]]:
        """搜索相似向量"""
        try:
            # 准备搜索参数
            search_params = {
                "metric_type": self.milvus_config.metric_type,
                "params": {"nprobe": min(16, self.milvus_config.nlist)}
            }

            # 准备过滤表达式
            filter_expr = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        conditions.append(f'{key} == "{value}"')
                    else:
                        conditions.append(f'{key} == {value}')
                filter_expr = " and ".join(conditions)

            # 执行搜索
            search_results = self.collection.search(
                data=[query_vector.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["id", "speaker_id", "sample_number", "audio_duration", "created_at", "metadata_json"]
            )

            # 处理结果
            similar_vectors = []
            if search_results and len(search_results) > 0:
                for hit in search_results[0]:
                    # 计算相似度（Milvus返回的是距离）
                    distance = hit.distance
                    if self.config.distance_metric == "cosine":
                        similarity = 1 - distance
                    elif self.config.distance_metric == "euclidean":
                        # 对于欧几里得距离，转换为相似度
                        similarity = 1 / (1 + distance)
                    elif self.config.distance_metric == "dot_product":
                        similarity = distance  # 点积本身就是相似度
                    else:
                        similarity = 1 - distance

                    # 应用阈值过滤
                    if threshold is None or similarity >= threshold:
                        # 解析元数据
                        metadata_json = hit.entity.get("metadata_json", "{}")
                        try:
                            metadata = json.loads(metadata_json)
                        except json.JSONDecodeError:
                            metadata = {}

                        record = VoicePrintRecord(
                            id=hit.entity.get("id"),
                            speaker_id=hit.entity.get("speaker_id"),
                            embedding=None,  # 搜索时不返回embedding以节省内存
                            sample_number=hit.entity.get("sample_number"),
                            audio_duration=hit.entity.get("audio_duration"),
                            created_at=datetime.fromisoformat(hit.entity.get("created_at")),
                            metadata=metadata
                        )
                        similar_vectors.append((record, similarity))

            return similar_vectors

        except Exception as e:
            self.logger.error(f"Failed to search similar vectors: {e}")
            return []

    async def get_vector_by_id(self, vector_id: str) -> Optional[VoicePrintRecord]:
        """根据ID获取向量记录"""
        try:
            # 使用查询获取特定ID的记录
            results = self.collection.query(
                expr=f'id == "{vector_id}"',
                output_fields=["id", "speaker_id", "embedding", "sample_number", "audio_duration", "created_at",
                               "metadata_json"]
            )

            if results:
                result = results[0]

                # 解析元数据
                metadata_json = result.get("metadata_json", "{}")
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    metadata = {}

                return VoicePrintRecord(
                    id=result["id"],
                    speaker_id=result["speaker_id"],
                    embedding=np.array(result["embedding"]),
                    sample_number=result["sample_number"],
                    audio_duration=result["audio_duration"],
                    created_at=datetime.fromisoformat(result["created_at"]),
                    metadata=metadata
                )

            return None

        except Exception as e:
            self.logger.error(f"Failed to get vector {vector_id}: {e}")
            return None

    async def get_vectors_by_speaker(self, speaker_id: str) -> List[VoicePrintRecord]:
        """根据说话人ID获取所有向量记录"""
        try:
            results = self.collection.query(
                expr=f'speaker_id == "{speaker_id}"',
                output_fields=["id", "speaker_id", "embedding", "sample_number", "audio_duration", "created_at",
                               "metadata_json"]
            )

            records = []
            for result in results:
                # 解析元数据
                metadata_json = result.get("metadata_json", "{}")
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    metadata = {}

                record = VoicePrintRecord(
                    id=result["id"],
                    speaker_id=result["speaker_id"],
                    embedding=np.array(result["embedding"]),
                    sample_number=result["sample_number"],
                    audio_duration=result["audio_duration"],
                    created_at=datetime.fromisoformat(result["created_at"]),
                    metadata=metadata
                )
                records.append(record)

            return records

        except Exception as e:
            self.logger.error(f"Failed to get vectors for speaker {speaker_id}: {e}")
            return []

    async def update_vector(self, vector_id: str, updates: Dict[str, Any]) -> bool:
        """更新向量记录"""
        try:
            # Milvus不支持直接更新，需要删除后重新插入
            existing_record = await self.get_vector_by_id(vector_id)
            if not existing_record:
                self.logger.warning(f"Vector {vector_id} not found for update")
                return False

            # 应用更新
            if 'speaker_id' in updates:
                existing_record.speaker_id = updates['speaker_id']
            if 'sample_number' in updates:
                existing_record.sample_number = updates['sample_number']
            if 'audio_duration' in updates:
                existing_record.audio_duration = updates['audio_duration']
            if 'metadata' in updates:
                existing_record.metadata.update(updates['metadata'])

            # 删除旧记录并插入新记录
            await self.delete_vector(vector_id)
            return await self.insert_vector(existing_record)

        except Exception as e:
            self.logger.error(f"Failed to update vector {vector_id}: {e}")
            return False

    async def delete_vector(self, vector_id: str) -> bool:
        """删除向量记录"""
        try:
            # 使用表达式删除特定ID的记录
            self.collection.delete(expr=f'id == "{vector_id}"')
            self.collection.flush()

            self.logger.debug(f"Deleted vector {vector_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete vector {vector_id}: {e}")
            return False

    async def delete_vectors_by_speaker(self, speaker_id: str) -> int:
        """删除说话人的所有向量记录"""
        try:
            # 先统计要删除的记录数
            count_before = await self.count_vectors(speaker_id)

            if count_before > 0:
                # 删除该说话人的所有记录
                self.collection.delete(expr=f'speaker_id == "{speaker_id}"')
                self.collection.flush()

                self.logger.info(f"Deleted {count_before} vectors for speaker {speaker_id}")
                return count_before

            return 0

        except Exception as e:
            self.logger.error(f"Failed to delete vectors for speaker {speaker_id}: {e}")
            return 0

    async def count_vectors(self, speaker_id: Optional[str] = None) -> int:
        """统计向量数量"""
        try:
            if speaker_id:
                # 查询特定说话人的向量数量
                results = self.collection.query(
                    expr=f'speaker_id == "{speaker_id}"',
                    output_fields=["id"]
                )
                return len(results)
            else:
                # 获取集合总数量
                return self.collection.num_entities

        except Exception as e:
            self.logger.error(f"Failed to count vectors: {e}")
            return 0

    async def list_speakers(self) -> Dict[str, List]:
        """列出所有说话人ID及其元数据"""
        try:
            # 查询所有记录的说话人ID和元数据
            results = self.collection.query(
                expr="",  # 空表达式查询所有记录
                output_fields=["speaker_id", "sample_number", "audio_duration", "created_at", "metadata_json"]
            )

            speakers_dict = {}
            for result in results:
                speaker_id = result["speaker_id"]

                # 解析元数据
                metadata_json = result.get("metadata_json", "{}")
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    metadata = {}

                # 构建元数据字典
                meta_info = {
                    "speaker_id": speaker_id,
                    "sample_number": result["sample_number"],
                    "audio_duration": result["audio_duration"],
                    "created_at": result["created_at"],
                    **metadata
                }

                if speaker_id not in speakers_dict:
                    speakers_dict[speaker_id] = []
                speakers_dict[speaker_id].append(meta_info)

            return speakers_dict

        except Exception as e:
            self.logger.error(f"Failed to list speakers: {e}")
            return {}

    async def clear_collection(self) -> bool:
        """清空集合"""
        try:
            from pymilvus import utility

            # 删除集合
            utility.drop_collection(self.config.collection_name)

            # 重新创建集合
            success = await self.create_collection()

            if success:
                self.logger.info(f"Cleared collection {self.config.collection_name}")
                return True
            else:
                return False

        except Exception as e:
            self.logger.error(f"Failed to clear collection: {e}")
            return False

class VectorDatabaseFactory:
    """向量数据库工厂类"""

    _implementations = {
        VectorDBType.CHROMADB: ChromaDBImplementation,
        VectorDBType.MILVUS: MilvusImplementation,
    }

    @classmethod
    def create_database(cls, config: VectorDBConfig) -> VectorDatabaseInterface:
        """创建向量数据库实例"""
        db_type = config.db_type
        if db_type not in cls._implementations:
            raise ValueError(f"Unsupported vector database type: {db_type}")

        implementation_class = cls._implementations[db_type]
        return implementation_class(config)

    @classmethod
    def register_implementation(cls, db_type: VectorDBType, implementation_class: type):
        """注册新的向量数据库实现"""
        cls._implementations[db_type] = implementation_class

    @classmethod
    def list_supported_databases(cls) -> List[str]:
        """列出支持的数据库类型"""
        return [db_type.value for db_type in cls._implementations.keys()]


# ============================================================================
# 配置工厂扩展
# ============================================================================

class VectorDBConfigFactory:
    """向量数据库配置工厂"""

    @staticmethod
    def create_chromadb_config(
            collection_name: str = "voice_prints",
            persist_directory: Optional[str] = None,
            host: Optional[str] = None,
            port: Optional[int] = None,
            dimension: int = 192,
            distance_metric: str = "cosine"
    ) -> ChromaDBConfig:
        """创建ChromaDB配置"""
        return ChromaDBConfig(
            db_type=VectorDBType.CHROMADB.value,
            collection_name=collection_name,
            dimension=dimension,
            distance_metric=distance_metric,
            persist_directory=persist_directory,
            host=host,
            port=port
        )

    @staticmethod
    def create_milvus_config(
            collection_name: str = "voice_prints",
            host: str = "localhost",
            port: int = 19530,
            username: Optional[str] = None,
            password: Optional[str] = None,
            dimension: int = 192,
            distance_metric: str = "cosine"
    ) -> MilvusConfig:
        """创建Milvus配置"""
        return MilvusConfig(
            db_type=VectorDBType.MILVUS.value,
            collection_name=collection_name,
            dimension=dimension,
            distance_metric=distance_metric,
            host=host,
            port=port,
            username=username,
            password=password
        )


# ============================================================================
# 使用示例
# ============================================================================

async def example_usage():
    """使用示例"""

    # 创建ChromaDB配置
    config = VectorDBConfigFactory.create_chromadb_config(
        collection_name="test_voice_prints",
        persist_directory="./test_chroma_db"
    )

    # 创建向量数据库实例
    vector_db = VectorDatabaseFactory.create_database(VectorDBType.CHROMADB, config)

    async with vector_db:
        # 创建测试数据
        test_record = VoicePrintRecord(
            id=str(uuid.uuid4()),
            speaker_id="test_speaker_001",
            embedding=np.random.rand(192),
            sample_number=1,
            audio_duration=10.5,
            created_at=datetime.now(),
            metadata={"source": "test"}
        )

        # 插入向量
        success = await vector_db.insert_vector(test_record)
        print(f"Insert result: {success}")

        # 搜索相似向量
        query_vector = np.random.rand(192)
        similar_vectors = await vector_db.search_similar_vectors(
            query_vector=query_vector,
            top_k=5,
            threshold=0.7
        )
        print(f"Found {len(similar_vectors)} similar vectors")

        # 列出说话人
        speakers = await vector_db.list_speakers()
        print(f"Speakers: {speakers}")

        # 统计向量数量
        count = await vector_db.count_vectors()
        print(f"Total vectors: {count}")


# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milvus向量数据库使用示例 - 参照现有代码风格
"""

import asyncio
import uuid
import numpy as np
from datetime import datetime


async def milvus_example_usage():
    """Milvus使用示例 - 参照现有example_usage()风格"""

    # 创建Milvus配置
    config = VectorDBConfigFactory.create_milvus_config(
        collection_name="test_voice_prints_milvus",
        host="localhost",
        port=19530,
        username=None,  # 如果有认证需求，设置用户名
        password=None,  # 如果有认证需求，设置密码
        dimension=192
    )

    # 创建向量数据库实例
    vector_db = VectorDatabaseFactory.create_database(VectorDBType.MILVUS, config)

    async with vector_db:
        print("Connected to Milvus successfully")

        # 创建测试数据
        test_record = VoicePrintRecord(
            id=str(uuid.uuid4()),
            speaker_id="test_speaker_001",
            embedding=np.random.rand(192).astype(np.float32),
            sample_number=1,
            audio_duration=10.5,
            created_at=datetime.now(),
            metadata={"source": "test", "quality": "high"}
        )

        # 插入向量
        success = await vector_db.insert_vector(test_record)
        print(f"Insert result: {success}")

        # 批量插入更多测试数据
        batch_records = []
        for i in range(5):
            record = VoicePrintRecord(
                id=str(uuid.uuid4()),
                speaker_id=f"test_speaker_{i:03d}",
                embedding=np.random.rand(192).astype(np.float32),
                sample_number=1,
                audio_duration=np.random.uniform(5.0, 20.0),
                created_at=datetime.now(),
                metadata={"source": "batch_test", "batch_id": i}
            )
            batch_records.append(record)

        batch_success = await vector_db.batch_insert_vectors(batch_records)
        print(f"Batch insert result: {batch_success}")

        # 搜索相似向量
        query_vector = np.random.rand(192).astype(np.float32)
        similar_vectors = await vector_db.search_similar_vectors(
            query_vector=query_vector,
            top_k=5,
            threshold=0.7
        )
        print(f"Found {len(similar_vectors)} similar vectors")

        # 打印搜索结果
        for i, (record, similarity) in enumerate(similar_vectors):
            print(f"  #{i + 1}: {record.speaker_id} (similarity: {similarity:.4f})")

        # 根据ID获取向量
        retrieved_record = await vector_db.get_vector_by_id(test_record.id)
        if retrieved_record:
            print(f"Retrieved record: {retrieved_record.speaker_id}")
        else:
            print("Record not found")

        # 根据说话人获取所有向量
        speaker_records = await vector_db.get_vectors_by_speaker("test_speaker_001")
        print(f"Found {len(speaker_records)} records for test_speaker_001")

        # 列出说话人
        speakers = await vector_db.list_speakers()
        print(f"Speakers: {list(speakers.keys())}")

        # 统计向量数量
        count = await vector_db.count_vectors()
        print(f"Total vectors: {count}")

        # 统计特定说话人的向量数量
        speaker_count = await vector_db.count_vectors("test_speaker_001")
        print(f"Vectors for test_speaker_001: {speaker_count}")

        # 更新向量
        update_success = await vector_db.update_vector(
            test_record.id,
            {"metadata": {"source": "updated_test", "updated": True}}
        )
        print(f"Update result: {update_success}")

        # 删除向量
        delete_success = await vector_db.delete_vector(test_record.id)
        print(f"Delete result: {delete_success}")

        # 删除说话人的所有向量
        deleted_count = await vector_db.delete_vectors_by_speaker("test_speaker_002")
        print(f"Deleted {deleted_count} vectors for test_speaker_002")

        # 最终统计
        final_count = await vector_db.count_vectors()
        print(f"Final vector count: {final_count}")


if __name__ == "__main__":
    # 运行示例
    # asyncio.run(example_usage())
    asyncio.run(milvus_example_usage())