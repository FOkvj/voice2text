"""
配置管理模块 - 从.env文件或环境变量加载配置
"""
import os
import logging
from typing import Any, Dict, Optional, List
from pathlib import Path


# 尝试导入dotenv，如果不存在则提供安装提示
try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv 未安装，请运行: pip install python-dotenv")
    print("或者添加到 requirements.txt 并重新安装依赖")
    
    # 定义一个空函数，以便在没有安装dotenv的情况下代码仍然可以运行
    def load_dotenv(*args, **kwargs):
        print("警告: python-dotenv 未安装，将只使用系统环境变量")
        return False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载.env文件
env_path = Path(__file__).parent.parent.parent.parent / '.env'
load_result = load_dotenv(dotenv_path=env_path)
if load_result:
    logger.info(f"已从 {env_path} 加载环境变量")
else:
    logger.warning(f"未找到 .env 文件或加载失败，将使用系统环境变量或默认值")


def get_env(key: str, default: Any = None, required: bool = False) -> Any:
    """
    从环境变量获取配置值
    
    Args:
        key: 环境变量名
        default: 默认值
        required: 是否必需，如果为True且环境变量不存在，则抛出异常
        
    Returns:
        环境变量值或默认值
    """
    value = os.getenv(key)
    if value is None:
        if required:
            raise ValueError(f"必需的环境变量 {key} 未设置")
        return default
    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """获取布尔类型的环境变量"""
    value = get_env(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'y', 't')


def get_env_int(key: str, default: int = 0) -> int:
    """获取整数类型的环境变量"""
    try:
        return int(get_env(key, default))
    except (ValueError, TypeError):
        logger.warning(f"环境变量 {key} 不是有效的整数，使用默认值 {default}")
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """获取浮点数类型的环境变量"""
    try:
        return float(get_env(key, default))
    except (ValueError, TypeError):
        logger.warning(f"环境变量 {key} 不是有效的浮点数，使用默认值 {default}")
        return default


def get_env_list(key: str, default: List = None, separator: str = ',') -> List:
    """获取列表类型的环境变量，使用分隔符分割"""
    if default is None:
        default = []
    value = get_env(key)
    if value is None:
        return default
    return [item.strip() for item in value.split(separator)]


def get_env_dict(key: str, default: Dict = None, separator: str = ',', kv_separator: str = '=') -> Dict:
    """获取字典类型的环境变量，使用分隔符分割键值对"""
    if default is None:
        default = {}
    value = get_env(key)
    if value is None:
        return default
    
    result = {}
    for item in value.split(separator):
        if kv_separator in item:
            k, v = item.split(kv_separator, 1)
            result[k.strip()] = v.strip()
    return result


# ASR配置
class ASRConfig:
    """ASR配置类"""
    
    @property
    def model_name(self) -> str:
        """ASR模型名称"""
        return get_env("ASR_MODEL_NAME", "paraformer-zh")
    
    @property
    def device(self) -> str:
        """ASR设备"""
        return get_env("ASR_DEVICE", "cpu")


# Whisper配置
class WhisperConfig:
    """Whisper配置类"""
    
    @property
    def model_name(self) -> str:
        """Whisper模型名称"""
        return get_env("WHISPER_MODEL_NAME", "large-v3-turbo")
    
    @property
    def device(self) -> str:
        """Whisper设备"""
        return get_env("WHISPER_DEVICE", "cpu")
    
    @property
    def use_auth_token(self) -> Optional[str]:
        """Hugging Face token"""
        return get_env("HF_TOKEN")


# 说话人识别配置
class SpeakerConfig:
    """说话人识别配置类"""
    
    @property
    def threshold(self) -> float:
        """说话人识别阈值"""
        return get_env_float("SPEAKER_THRESHOLD", 0.5)
    
    @property
    def device(self) -> str:
        """说话人识别设备"""
        return get_env("SPEAKER_DEVICE", "cpu")


# 向量数据库配置
class VectorDBConfig:
    """向量数据库配置类"""
    
    @property
    def db_type(self) -> str:
        """向量数据库类型"""
        from voice2text.tran.vector_base import VectorDBType
        db_type_str = get_env("VECTOR_DB_TYPE", "CHROMADB")
        try:
            return getattr(VectorDBType, db_type_str)
        except AttributeError:
            logger.warning(f"未知的向量数据库类型 {db_type_str}，使用默认值 CHROMADB")
            return VectorDBType.CHROMADB
    
    @property
    def persist_directory(self) -> str:
        """向量数据库持久化目录"""
        return get_env("VECTOR_DB_PERSIST_DIRECTORY", "./voice_vectors")
    
    @property
    def collection_name(self) -> str:
        """向量数据库集合名称"""
        return get_env("VECTOR_DB_COLLECTION_NAME", "voice_prints")


# S3存储配置
class S3Config:
    """S3存储配置类"""
    
    @property
    def storage_type(self) -> str:
        """存储类型"""
        from voice2text.tran.filesystem import StorageType
        storage_type_str = get_env("STORAGE_TYPE", "S3")
        try:
            return getattr(StorageType, storage_type_str)
        except AttributeError:
            logger.warning(f"未知的存储类型 {storage_type_str}，使用默认值 S3")
            return StorageType.S3
    
    @property
    def bucket_name(self) -> str:
        """S3桶名称"""
        return get_env("S3_BUCKET_NAME", "voice")
    
    @property
    def endpoint_url(self) -> str:
        """S3端点URL"""
        return get_env("S3_ENDPOINT_URL", "http://localhost:9000")
    
    @property
    def access_key_id(self) -> str:
        """S3访问密钥ID"""
        return get_env("S3_ACCESS_KEY_ID", "admin")
    
    @property
    def secret_access_key(self) -> str:
        """S3秘密访问密钥"""
        return get_env("S3_SECRET_ACCESS_KEY", "minioadmin123")
    
    @property
    def prefix(self) -> str:
        """S3前缀"""
        return get_env("S3_PREFIX", "stt")


# 服务配置
class ServiceConfig:
    """服务配置类"""
    
    @property
    def transcription_strategy(self) -> str:
        """转写策略"""
        from voice2text.tran.speech2text import TranscriptionStrategy
        strategy_str = get_env("TRANSCRIPTION_STRATEGY", "AUTO_SELECT")
        try:
            return getattr(TranscriptionStrategy, strategy_str)
        except AttributeError:
            logger.warning(f"未知的转写策略 {strategy_str}，使用默认值 AUTO_SELECT")
            return TranscriptionStrategy.AUTO_SELECT
    
    @property
    def language_model_mapping(self) -> Dict[str, str]:
        """语言模型映射"""
        default_mapping = {
            "auto": "whisper",
            "zh": "funasr",
            "en": "whisper",
            "ja": "whisper"
        }
        # 这里可以从环境变量加载更复杂的映射，但为简单起见，我们使用默认值
        return default_mapping
    
    @property
    def max_transcribe_concurrent(self) -> int:
        """最大并发转写任务数"""
        return get_env_int("MAX_TRANSCRIBE_CONCURRENT", 2)
    
    @property
    def max_speaker_concurrent(self) -> int:
        """最大并发说话人识别任务数"""
        return get_env_int("MAX_SPEAKER_CONCURRENT", 3)
    
    @property
    def task_timeout(self) -> float:
        """任务超时时间（秒）"""
        return get_env_float("TASK_TIMEOUT", 300.0)


# 服务器配置
class ServerConfig:
    """服务器配置类"""
    
    @property
    def host(self) -> str:
        """服务器主机"""
        return get_env("SERVER_HOST", "0.0.0.0")
    
    @property
    def port(self) -> int:
        """服务器端口"""
        return get_env_int("SERVER_PORT", 8765)
    
    @property
    def log_level(self) -> str:
        """日志级别"""
        return get_env("SERVER_LOG_LEVEL", "info")


# 全局配置实例
asr_config = ASRConfig()
whisper_config = WhisperConfig()
speaker_config = SpeakerConfig()
vector_db_config = VectorDBConfig()
s3_config = S3Config()
service_config = ServiceConfig()
server_config = ServerConfig()