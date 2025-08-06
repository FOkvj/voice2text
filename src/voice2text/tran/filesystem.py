# ============================================================================
# 改进的文件管理器 - 支持多种存储协议
# ============================================================================

import asyncio
import hashlib
import io
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Union, BinaryIO, List, Literal, Tuple

import aioboto3
import aiofiles
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field


# ============================================================================
# 存储类型枚举
# ============================================================================

class StorageType(Enum):
    """存储类型枚举"""
    LOCAL = "local"
    S3 = "s3"
    # 未来可以添加更多类型
    # AZURE = "azure"
    # GCS = "gcs"
    # OSS = "oss"


# ============================================================================
# 配置类
# ============================================================================

class StorageConfig(BaseModel):
    """存储配置基类"""
    storage_type: StorageType
    metadata_backend: str = "json"  # json, redis, dynamodb等
    enable_versioning: bool = False
    enable_encryption: bool = False

    class Config:
        use_enum_values = False


class LocalStorageConfig(StorageConfig):
    """本地存储配置"""
    base_dir: str
    temp_dir: Optional[str] = Field(default="/tmp/local_cache", description="临时文件目录")
    storage_type: StorageType = Field(default=StorageType.LOCAL, description="local存储类型")


class S3StorageConfig(StorageConfig):
    """S3存储配置"""
    bucket_name: str
    region_name: str = "us-east-1"
    endpoint_url: Optional[str] = None  # 用于S3兼容服务如MinIO
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    prefix: str = ""  # S3键前缀
    temp_dir: str = "/tmp/s3_cache"
    storage_type: StorageType = Field(default=StorageType.S3, description="s3存储类型")


# ============================================================================
# 文件元数据类
# ============================================================================

class FileMetadata(BaseModel):
    """文件元数据"""
    file_id: str
    filename: str
    category: str
    storage_path: str  # 相对路径或S3 key
    size: int
    hash: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    storage_type: StorageType = StorageType.LOCAL
    content_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = False
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'file_id': self.file_id,
            'filename': self.filename,
            'category': self.category,
            'storage_path': self.storage_path,
            'size': self.size,
            'hash': self.hash,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'storage_type': self.storage_type.value,
            'content_type': self.content_type,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileMetadata':
        """从字典创建"""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('updated_at'):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        data['storage_type'] = StorageType(data.get('storage_type', 'local'))
        return cls(**data)


# ============================================================================
# 存储接口
# ============================================================================

class FileStorageInterface(ABC):
    """文件存储抽象接口"""

    @abstractmethod
    async def connect(self) -> None:
        """连接到存储服务"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """断开存储服务连接"""
        pass

    @abstractmethod
    async def save(self,
                   data: Union[bytes, BinaryIO],
                   path: str,
                   content_type: Optional[str] = None) -> Dict[str, Any]:
        """保存文件"""
        pass

    @abstractmethod
    async def load(self, path: str) -> bytes:
        """加载文件内容"""
        pass

    @abstractmethod
    async def load_stream(self, path: str) -> BinaryIO:
        """加载文件并返回流对象"""
        pass

    @abstractmethod
    async def exists(self, path: str) -> bool:
        """检查文件是否存在"""
        pass

    @abstractmethod
    async def delete(self, path: str) -> bool:
        """删除文件"""
        pass

    @abstractmethod
    async def list_files(self, prefix: str = "") -> List[str]:
        """列出文件"""
        pass

    @abstractmethod
    async def get_file_info(self, path: str) -> Dict[str, Any]:
        """获取文件信息"""
        pass

    @abstractmethod
    async def get_temp_path(self, path: str) -> str:
        """获取用于处理的临时文件路径"""
        pass

    @abstractmethod
    async def copy(self, src_path: str, dst_path: str) -> bool:
        """复制文件"""
        pass

    @abstractmethod
    async def move(self, src_path: str, dst_path: str) -> bool:
        """移动文件"""
        pass


# ============================================================================
# 本地文件存储实现
# ============================================================================

class LocalFileStorage(FileStorageInterface):
    """本地文件存储实现"""

    def __init__(self, config: LocalStorageConfig):
        self.config = config
        self.base_dir = Path(config.base_dir)
        self.temp_dir = Path(config.temp_dir)

    async def connect(self) -> None:
        """创建必要的目录"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def disconnect(self) -> None:
        """本地存储无需断开连接"""
        pass

    async def save(self,
                   data: Union[bytes, BinaryIO],
                   path: str,
                   content_type: Optional[str] = None) -> Dict[str, Any]:
        """保存文件到本地"""
        full_path = self.base_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # 异步写入文件
        if isinstance(data, bytes):
            async with aiofiles.open(full_path, 'wb') as f:
                await f.write(data)
            size = len(data)
        else:
            # 如果是文件对象，先读取到内存
            if hasattr(data, 'read'):
                content = data.read()
                async with aiofiles.open(full_path, 'wb') as f:
                    await f.write(content)
                size = len(content)
            else:
                raise ValueError("Unsupported data type")

        # 计算文件哈希
        file_hash = await self._calculate_file_hash(full_path)

        return {
            'size': size,
            'hash': file_hash,
            'path': str(full_path),
            'content_type': content_type
        }

    async def load(self, path: str) -> bytes:
        """从本地加载文件"""
        full_path = self.base_dir / path

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        async with aiofiles.open(full_path, 'rb') as f:
            return await f.read()

    async def load_stream(self, path: str) -> BinaryIO:
        """从本地加载文件并返回流对象"""
        full_path = self.base_dir / path

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # 读取文件内容到内存并返回BytesIO流
        async with aiofiles.open(full_path, 'rb') as f:
            content = await f.read()

        return io.BytesIO(content)

    async def exists(self, path: str) -> bool:
        """检查文件是否存在"""
        full_path = self.base_dir / path
        return full_path.exists()

    async def delete(self, path: str) -> bool:
        """删除本地文件"""
        full_path = self.base_dir / path

        if full_path.exists():
            full_path.unlink()
            return True
        return False

    async def list_files(self, prefix: str = "") -> List[str]:
        """列出本地文件"""
        search_path = self.base_dir / prefix

        if not search_path.exists():
            return []

        files = []
        for file_path in search_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(self.base_dir)
                files.append(str(relative_path))

        return files

    async def get_file_info(self, path: str) -> Dict[str, Any]:
        """获取文件信息"""
        full_path = self.base_dir / path

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        stat = full_path.stat()
        file_hash = await self._calculate_file_hash(full_path)

        return {
            'size': stat.st_size,
            'created_at': datetime.fromtimestamp(stat.st_ctime),
            'updated_at': datetime.fromtimestamp(stat.st_mtime),
            'hash': file_hash
        }

    async def get_temp_path(self, path: str) -> str:
        """获取临时文件路径（本地存储直接返回原路径）"""
        full_path = self.base_dir / path
        return str(full_path)

    async def copy(self, src_path: str, dst_path: str) -> bool:
        """复制文件"""
        src_full = self.base_dir / src_path
        dst_full = self.base_dir / dst_path

        if not src_full.exists():
            return False

        dst_full.parent.mkdir(parents=True, exist_ok=True)

        # 异步复制
        async with aiofiles.open(src_full, 'rb') as src:
            async with aiofiles.open(dst_full, 'wb') as dst:
                await dst.write(await src.read())

        return True

    async def move(self, src_path: str, dst_path: str) -> bool:
        """移动文件"""
        src_full = self.base_dir / src_path
        dst_full = self.base_dir / dst_path

        if not src_full.exists():
            return False

        dst_full.parent.mkdir(parents=True, exist_ok=True)
        src_full.rename(dst_full)
        return True

    async def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希"""
        hash_md5 = hashlib.md5()

        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(8192):
                hash_md5.update(chunk)

        return hash_md5.hexdigest()


# ============================================================================
# S3文件存储实现
# ============================================================================

class S3FileStorage(FileStorageInterface):
    """S3文件存储实现"""

    def __init__(self, config: S3StorageConfig):
        self.config = config
        self.session = None
        self.s3_client = None
        self.temp_dir = Path(config.temp_dir)
        self.save_chunk_size = 1024 * 1024

    async def connect(self) -> None:
        """连接到S3"""
        # 创建临时目录
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # 创建S3会话
        self.session = aioboto3.Session(
            aws_access_key_id=self.config.access_key_id,
            aws_secret_access_key=self.config.secret_access_key,
            region_name=self.config.region_name
        )

        # 测试连接
        async with self.session.client(
                's3',
                endpoint_url=self.config.endpoint_url
        ) as s3:
            try:
                await s3.head_bucket(Bucket=self.config.bucket_name)
            except ClientError:
                # 尝试创建bucket
                await s3.create_bucket(Bucket=self.config.bucket_name)

    async def disconnect(self) -> None:
        """断开S3连接"""
        # aioboto3使用上下文管理器，无需显式断开
        pass

    def _get_s3_key(self, path: str) -> str:
        """获取S3键"""
        if self.config.prefix:
            return f"{self.config.prefix}/{path}"
        return path

    async def save(self,
                   data: Union[bytes, BinaryIO],
                   path: str,
                   content_type: Optional[str] = None) -> Dict[str, Any]:
        """保存文件到S3"""
        s3_key = self._get_s3_key(path)

        # 准备上传数据
        if isinstance(data, bytes):
            file_obj = io.BytesIO(data)
            size = len(data)
            # 计算哈希
            file_hash = hashlib.md5(data).hexdigest()
        else:
            # 如果是文件对象，读取内容
            content = data.read() if hasattr(data, 'read') else data
            file_obj = io.BytesIO(content)
            size = len(content)
            file_hash = hashlib.md5(content).hexdigest()

        # 上传到S3
        async with self.session.client('s3', endpoint_url=self.config.endpoint_url) as s3:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type

            await s3.put_object(
                Bucket=self.config.bucket_name,
                Key=s3_key,
                Body=file_obj.getvalue(),
                **extra_args
            )

        return {
            'size': size,
            'hash': file_hash,
            'path': s3_key,
            'content_type': content_type
        }

    async def load(self, path: str) -> bytes:
        """从S3加载文件"""
        s3_key = self._get_s3_key(path)

        async with self.session.client('s3', endpoint_url=self.config.endpoint_url) as s3:
            try:
                response = await s3.get_object(
                    Bucket=self.config.bucket_name,
                    Key=s3_key
                )
                return await response['Body'].read()
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    raise FileNotFoundError(f"File not found: {path}")
                raise

    async def load_stream(self, path: str) -> BinaryIO:
        """从S3加载文件并返回流对象"""
        s3_key = self._get_s3_key(path)

        async with self.session.client('s3', endpoint_url=self.config.endpoint_url) as s3:
            try:
                response = await s3.get_object(
                    Bucket=self.config.bucket_name,
                    Key=s3_key
                )
                content = io.BytesIO()
                body = response["Body"]

                # Stream the data in chunks
                while True:
                    chunk = await body.read(self.save_chunk_size)
                    if not chunk:
                        break
                    content.write(chunk)

                content.seek(0)
                return content
            except Exception as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    raise FileNotFoundError(f"File not found: {path}")
                raise

    async def exists(self, path: str) -> bool:
        """检查文件是否存在"""
        s3_key = self._get_s3_key(path)

        async with self.session.client('s3', endpoint_url=self.config.endpoint_url) as s3:
            try:
                await s3.head_object(
                    Bucket=self.config.bucket_name,
                    Key=s3_key
                )
                return True
            except ClientError:
                return False

    async def delete(self, path: str) -> bool:
        """删除S3文件"""
        s3_key = self._get_s3_key(path)

        async with self.session.client('s3', endpoint_url=self.config.endpoint_url) as s3:
            try:
                await s3.delete_object(
                    Bucket=self.config.bucket_name,
                    Key=s3_key
                )
                return True
            except ClientError:
                return False

    async def list_files(self, prefix: str = "") -> List[str]:
        """列出S3文件"""
        list_prefix = self._get_s3_key(prefix)
        files = []

        async with self.session.client('s3', endpoint_url=self.config.endpoint_url) as s3:
            paginator = s3.get_paginator('list_objects_v2')

            async for page in paginator.paginate(
                    Bucket=self.config.bucket_name,
                    Prefix=list_prefix
            ):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        # 移除配置的前缀
                        key = obj['Key']
                        if self.config.prefix and key.startswith(self.config.prefix + '/'):
                            key = key[len(self.config.prefix) + 1:]
                        files.append(key)

        return files

    async def get_file_info(self, path: str) -> Dict[str, Any]:
        """获取文件信息"""
        s3_key = self._get_s3_key(path)

        async with self.session.client('s3', endpoint_url=self.config.endpoint_url) as s3:
            try:
                response = await s3.head_object(
                    Bucket=self.config.bucket_name,
                    Key=s3_key
                )

                return {
                    'size': response['ContentLength'],
                    'created_at': response.get('LastModified'),
                    'updated_at': response.get('LastModified'),
                    'hash': response.get('ETag', '').strip('"'),
                    'content_type': response.get('ContentType')
                }
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    raise FileNotFoundError(f"File not found: {path}")
                raise

    async def get_temp_path(self, path: str) -> str:
        """下载S3文件到临时路径"""
        # 创建临时文件
        temp_file = self.temp_dir / f"s3_temp_{uuid.uuid4().hex}_{Path(path).name}"

        # 下载文件
        content = await self.load(path)

        # 写入临时文件
        async with aiofiles.open(temp_file, 'wb') as f:
            await f.write(content)

        return str(temp_file)

    async def copy(self, src_path: str, dst_path: str) -> bool:
        """复制S3文件"""
        src_key = self._get_s3_key(src_path)
        dst_key = self._get_s3_key(dst_path)

        async with self.session.client('s3', endpoint_url=self.config.endpoint_url) as s3:
            try:
                await s3.copy_object(
                    Bucket=self.config.bucket_name,
                    CopySource={'Bucket': self.config.bucket_name, 'Key': src_key},
                    Key=dst_key
                )
                return True
            except ClientError:
                return False

    async def move(self, src_path: str, dst_path: str) -> bool:
        """移动S3文件"""
        # S3没有原生的移动操作，需要复制后删除
        if await self.copy(src_path, dst_path):
            return await self.delete(src_path)
        return False


# ============================================================================
# 元数据存储接口
# ============================================================================

class MetadataStorageInterface(ABC):
    """元数据存储接口"""

    @abstractmethod
    async def save(self, metadata: Dict[str, FileMetadata]) -> None:
        """保存元数据"""
        pass

    @abstractmethod
    async def load(self) -> Dict[str, FileMetadata]:
        """加载元数据"""
        pass

    @abstractmethod
    async def get(self, file_id: str) -> Optional[FileMetadata]:
        """获取单个文件元数据"""
        pass

    @abstractmethod
    async def delete(self, file_id: str) -> bool:
        """删除元数据"""
        pass

    @abstractmethod
    async def update(self, file_id: str, updates: Dict[str, Any]) -> bool:
        """更新元数据"""
        pass


# ============================================================================
# JSON元数据存储实现
# ============================================================================

class JsonMetadataStorage(MetadataStorageInterface):
    """JSON文件元数据存储"""

    def __init__(self, storage: FileStorageInterface, metadata_path: str = "metadata.json"):
        self.storage = storage
        self.metadata_path = metadata_path
        self._cache: Dict[str, FileMetadata] = {}
        self._loaded = False

    async def save(self, metadata: Dict[str, FileMetadata]) -> None:
        """保存元数据到JSON"""
        # 转换为可序列化格式
        data = {}
        for file_id, meta in metadata.items():
            data[file_id] = meta.to_dict()

        # 保存到存储
        json_data = json.dumps(data, ensure_ascii=False, indent=2)
        await self.storage.save(
            json_data.encode('utf-8'),
            self.metadata_path,
            content_type='application/json'
        )

        self._cache = metadata.copy()

    async def load(self) -> Dict[str, FileMetadata]:
        """从JSON加载元数据"""
        if self._loaded:
            return self._cache

        try:
            content = await self.storage.load(self.metadata_path)
            data = json.loads(content.decode('utf-8'))

            self._cache = {}
            for file_id, meta_dict in data.items():
                self._cache[file_id] = FileMetadata.from_dict(meta_dict)

            self._loaded = True
            return self._cache

        except FileNotFoundError:
            self._cache = {}
            self._loaded = True
            return self._cache

    async def get(self, file_id: str) -> Optional[FileMetadata]:
        """获取单个文件元数据"""
        if not self._loaded:
            await self.load()
        return self._cache.get(file_id)

    async def delete(self, file_id: str) -> bool:
        """删除元数据"""
        if not self._loaded:
            await self.load()

        if file_id in self._cache:
            del self._cache[file_id]
            await self.save(self._cache)
            return True
        return False

    async def update(self, file_id: str, updates: Dict[str, Any]) -> bool:
        """更新元数据"""
        if not self._loaded:
            await self.load()

        if file_id in self._cache:
            meta = self._cache[file_id]
            for key, value in updates.items():
                if hasattr(meta, key):
                    setattr(meta, key, value)
            meta.updated_at = datetime.now()
            await self.save(self._cache)
            return True
        return False


# ============================================================================
# 改进的文件管理器
# ============================================================================

class ImprovedFileManager:
    """改进的文件管理器 - 支持多种存储协议"""

    def __init__(self,
                 storage: FileStorageInterface,
                 metadata_storage: Optional[MetadataStorageInterface] = None):
        """
        初始化文件管理器

        Args:
            storage: 文件存储接口实例
            metadata_storage: 元数据存储接口实例（可选）
        """
        self.storage = storage
        self.metadata_storage = metadata_storage or JsonMetadataStorage(storage)
        self.metadata: Dict[str, FileMetadata] = {}
        self._initialized = False

    async def initialize(self):
        """初始化管理器"""
        if self._initialized:
            return

        # 连接存储
        await self.storage.connect()

        # 加载元数据
        self.metadata = await self.metadata_storage.load()

        self._initialized = True
        print(f"FileManager initialized with {len(self.metadata)} files")

    async def save_file(self,
                        data: Union[bytes, BinaryIO],
                        filename: str,
                        category: str = "general",
                        metadata: Optional[Dict] = None,
                        content_type: Optional[str] = None) -> str:
        """
        保存文件

        Args:
            data: 文件数据或文件对象
            filename: 文件名
            category: 文件分类
            metadata: 附加元数据
            content_type: 内容类型

        Returns:
            文件ID
        """
        if not self._initialized:
            await self.initialize()

        # 生成文件ID和路径
        file_id = str(uuid.uuid4())
        file_path = f"{category}/{file_id}_{filename}"

        # 保存文件到存储
        save_result = await self.storage.save(data, file_path, content_type)

        # 创建元数据
        file_metadata = FileMetadata(
            file_id=file_id,
            filename=filename,
            category=category,
            storage_path=file_path,
            size=save_result['size'],
            hash=save_result['hash'],
            created_at=datetime.now(),
            storage_type=self.storage.config.storage_type,
            content_type=content_type,
            metadata=metadata or {}
        )

        # 保存元数据
        self.metadata[file_id] = file_metadata
        await self.metadata_storage.save(self.metadata)

        return file_id

    async def get_file_path(self, file_id: str) -> Optional[str]:
        """
        获取文件路径（对于需要本地路径的场景）

        对于S3等远程存储，会下载到临时目录
        """
        if not self._initialized:
            await self.initialize()

        if file_id not in self.metadata:
            return None

        file_meta = self.metadata[file_id]

        # 获取临时路径（本地存储直接返回，远程存储会下载）
        temp_path = await self.storage.get_temp_path(file_meta.storage_path)
        return temp_path

    async def load_file(self, file_id: str) -> Tuple[Optional[FileMetadata], Optional[bytes]]:
        """加载文件内容"""
        if not self._initialized:
            await self.initialize()

        if file_id not in self.metadata:
            return None, None

        file_meta = self.metadata[file_id]
        try:
            return file_meta, await self.storage.load(file_meta.storage_path)
        except FileNotFoundError:
            return None, None

    async def load_file_stream(self, file_id: str) -> Tuple[Optional[FileMetadata], Optional[BinaryIO]]:
        """
        加载文件并返回元数据和流对象的元组

        Args:
            file_id: 文件ID

        Returns:
            Tuple[FileMetadata, BinaryIO]: 文件元数据和二进制流对象的元组
            如果文件不存在，返回 (None, None)
        """
        if not self._initialized:
            await self.initialize()

        if file_id not in self.metadata:
            return None, None

        file_meta = self.metadata[file_id]
        try:
            stream = await self.storage.load_stream(file_meta.storage_path)
            return file_meta, stream
        except FileNotFoundError:
            return None, None

    async def delete_file(self, file_id: str) -> bool:
        """删除文件"""
        if not self._initialized:
            await self.initialize()

        if file_id not in self.metadata:
            return False

        file_meta = self.metadata[file_id]

        # 删除存储中的文件
        deleted = await self.storage.delete(file_meta.storage_path)

        if deleted:
            # 删除元数据
            await self.metadata_storage.delete(file_id)
            del self.metadata[file_id]

        return deleted

    def list_files(self, category: Optional[str] = None) -> List[Dict]:
        """列出文件"""
        files = []
        for file_id, meta in self.metadata.items():
            if category is None or meta.category == category:
                files.append({
                    'file_id': file_id,
                    'filename': meta.filename,
                    'category': meta.category,
                    'size': meta.size,
                    'created_at': meta.created_at.isoformat(),
                    'metadata': meta.metadata
                })
        return files

    def get_file_by_id(self, file_id: str) -> Optional[FileMetadata]:
        """通过文件ID获取文件元数据"""
        return self.metadata.get(file_id)

    async def copy_file(self, file_id: str, new_category: Optional[str] = None) -> Optional[str]:
        """复制文件"""
        if not self._initialized:
            await self.initialize()

        if file_id not in self.metadata:
            return None

        source_meta = self.metadata[file_id]

        # 生成新的文件ID和路径
        new_file_id = str(uuid.uuid4())
        new_category = new_category or source_meta.category
        new_path = f"{new_category}/{new_file_id}_{source_meta.filename}"

        # 复制文件
        success = await self.storage.copy(source_meta.storage_path, new_path)

        if success:
            # 创建新的元数据
            new_metadata = FileMetadata(
                file_id=new_file_id,
                filename=source_meta.filename,
                category=new_category,
                storage_path=new_path,
                size=source_meta.size,
                hash=source_meta.hash,
                created_at=datetime.now(),
                storage_type=source_meta.storage_type,
                content_type=source_meta.content_type,
                metadata=source_meta.metadata.copy()
            )

            self.metadata[new_file_id] = new_metadata
            await self.metadata_storage.save(self.metadata)

            return new_file_id

        return None

    async def move_file(self, file_id: str, new_category: str) -> bool:
        """移动文件到新分类"""
        if not self._initialized:
            await self.initialize()

        if file_id not in self.metadata:
            return False

        file_meta = self.metadata[file_id]

        # 生成新路径
        new_path = f"{new_category}/{file_id}_{file_meta.filename}"

        # 移动文件
        success = await self.storage.move(file_meta.storage_path, new_path)

        if success:
            # 更新元数据
            file_meta.category = new_category
            file_meta.storage_path = new_path
            file_meta.updated_at = datetime.now()
            await self.metadata_storage.save(self.metadata)

        return success

    async def cleanup_temp_files(self, max_age_hours: int = 24):
        """清理临时文件"""
        # 主要用于S3存储的本地缓存清理
        if hasattr(self.storage, 'temp_dir'):
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600

            temp_dir = Path(self.storage.temp_dir)
            if temp_dir.exists():
                for file_path in temp_dir.iterdir():
                    if file_path.is_file():
                        file_age = current_time - file_path.stat().st_mtime
                        if file_age > max_age_seconds:
                            file_path.unlink()
                            print(f"Cleaned up temp file: {file_path}")

    # 上下文管理器支持
    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.storage.disconnect()


# ============================================================================
# 存储工厂
# ============================================================================

class StorageFactory:
    """存储工厂类"""

    @staticmethod
    def create_storage(config: StorageConfig) -> FileStorageInterface:
        """根据配置创建存储实例"""
        if isinstance(config, LocalStorageConfig):
            return LocalFileStorage(config)
        elif isinstance(config, S3StorageConfig):
            return S3FileStorage(config)
        else:
            raise ValueError(f"Unsupported storage config type: {type(config)}")

    @staticmethod
    def create_local_storage(base_dir: str, **kwargs) -> LocalFileStorage:
        """创建本地存储"""
        config = LocalStorageConfig(
            storage_type=StorageType.LOCAL,
            base_dir=base_dir,
            **kwargs
        )
        return LocalFileStorage(config)

    @staticmethod
    def create_s3_storage(bucket_name: str,
                          region_name: str = "us-east-1",
                          **kwargs) -> S3FileStorage:
        """创建S3存储"""
        config = S3StorageConfig(
            storage_type=StorageType.S3,
            bucket_name=bucket_name,
            region_name=region_name,
            **kwargs
        )
        return S3FileStorage(config)



async def example_usage():
    """使用示例"""

    # 示例1：使用本地存储
    print("=== 本地存储示例 ===")
    local_config = LocalStorageConfig(
        storage_type=StorageType.LOCAL,
        base_dir="./test_storage",
        temp_dir="./test_storage/temp"
    )

    local_storage = StorageFactory.create_storage(local_config)

    async with ImprovedFileManager(local_storage) as file_manager:
        # 保存文件
        file_id = await file_manager.save_file(
            b"Hello, Local Storage!",
            "test.txt",
            category="documents",
            metadata={"author": "test"},
            content_type="text/plain"
        )
        print(f"Saved file with ID: {file_id}")

        # 获取文件
        # content = await file_manager.load_file(file_id)
        # print(f"File content: {content.decode('utf-8')}")

        # 列出文件
        files = file_manager.list_files()
        print(f"Files: {files}")


async def quick_test():
    """快速测试 MinIO 服务"""
    print("🧪 快速测试 MinIO 服务...")

    # MinIO 配置
    config = S3StorageConfig(
        storage_type=StorageType.S3,
        bucket_name="quicktest",
        endpoint_url="http://localhost:9000",
        access_key_id="admin",
        secret_access_key="minioadmin123",
        prefix="test"
    )

    storage = StorageFactory.create_storage(config)

    try:
        async with ImprovedFileManager(storage) as fm:
            print("✅ 连接成功")

            # 上传测试文件
            file_id = await fm.save_file(
                b"Hello MinIO from Python!",
                "hello.txt",
                "test"
            )
            print(f"✅ 文件上传成功: {file_id}")

            # 下载测试文件
            metadata, content = await fm.load_file(file_id)
            print(f"✅ 文件下载成功: {content.decode()}")

            # 删除测试文件
            # await fm.delete_file(file_id)
            # print("✅ 文件删除成功")

            print("🎉 MinIO 服务工作正常！")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("\n请检查:")
        print("1. MinIO 服务是否启动: podman ps")
        print("2. 端口是否正确: http://localhost:9000")
        print("3. 凭据是否正确: minioadmin/minioadmin123")

if __name__ == "__main__":
    # 运行示例
    # asyncio.run(example_usage())
    asyncio.run(quick_test())