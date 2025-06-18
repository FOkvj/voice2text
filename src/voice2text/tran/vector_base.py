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
    db_type: str
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
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'speaker_id': self.speaker_id,
            'embedding': self.embedding.tolist(),
            'sample_number': self.sample_number,
            'audio_duration': self.audio_duration,
            'created_at': self.created_at.isoformat(),
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
    async def list_speakers(self) -> List[str]:
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
                'created_at': record.created_at.isoformat(),
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

    async def list_speakers(self) -> List[str]:
        """列出所有说话人ID"""
        try:
            results = self.collection.get(include=['metadatas'])
            speakers = set()

            if results['metadatas']:
                for metadata in results['metadatas']:
                    speakers.add(metadata['speaker_id'])

            return list(speakers)

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
    """Milvus实现 - 占位符，待实现"""

    def __init__(self, config: MilvusConfig):
        super().__init__(config)
        self.milvus_config = config
        self.connection = None

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

            self.logger.info("Connected to Milvus successfully")
            await self.create_collection()

        except ImportError:
            raise ImportError("pymilvus is not installed. Please install with: pip install pymilvus")
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus: {e}")
            raise

    async def disconnect(self) -> None:
        """断开Milvus连接"""
        try:
            from pymilvus import connections
            connections.disconnect("default")
            self.logger.info("Disconnected from Milvus")
        except Exception as e:
            self.logger.error(f"Failed to disconnect from Milvus: {e}")

    # 其他方法的实现...
    async def create_collection(self) -> bool:
        # TODO: 实现Milvus集合创建
        raise NotImplementedError("Milvus implementation is not complete yet")

    async def collection_exists(self) -> bool:
        raise NotImplementedError("Milvus implementation is not complete yet")

    async def insert_vector(self, record: VoicePrintRecord) -> bool:
        raise NotImplementedError("Milvus implementation is not complete yet")

    async def batch_insert_vectors(self, records: List[VoicePrintRecord]) -> bool:
        raise NotImplementedError("Milvus implementation is not complete yet")

    async def search_similar_vectors(self, query_vector: np.ndarray, top_k: int = 10,
                                     threshold: Optional[float] = None,
                                     filters: Optional[Dict[str, Any]] = None) -> List[Tuple[VoicePrintRecord, float]]:
        raise NotImplementedError("Milvus implementation is not complete yet")

    async def get_vector_by_id(self, vector_id: str) -> Optional[VoicePrintRecord]:
        raise NotImplementedError("Milvus implementation is not complete yet")

    async def get_vectors_by_speaker(self, speaker_id: str) -> List[VoicePrintRecord]:
        raise NotImplementedError("Milvus implementation is not complete yet")

    async def update_vector(self, vector_id: str, updates: Dict[str, Any]) -> bool:
        raise NotImplementedError("Milvus implementation is not complete yet")

    async def delete_vector(self, vector_id: str) -> bool:
        raise NotImplementedError("Milvus implementation is not complete yet")

    async def delete_vectors_by_speaker(self, speaker_id: str) -> int:
        raise NotImplementedError("Milvus implementation is not complete yet")

    async def count_vectors(self, speaker_id: Optional[str] = None) -> int:
        raise NotImplementedError("Milvus implementation is not complete yet")

    async def list_speakers(self) -> List[str]:
        raise NotImplementedError("Milvus implementation is not complete yet")

    async def clear_collection(self) -> bool:
        raise NotImplementedError("Milvus implementation is not complete yet")


class VectorDatabaseFactory:
    """向量数据库工厂类"""

    _implementations = {
        VectorDBType.CHROMADB: ChromaDBImplementation,
        VectorDBType.MILVUS: MilvusImplementation,
    }

    @classmethod
    def create_database(cls, db_type: VectorDBType, config: VectorDBConfig) -> VectorDatabaseInterface:
        """创建向量数据库实例"""
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


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())