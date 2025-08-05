# ============================================================================
# Voice2Text SDK - 更新的API实现，集成文件管理器
# ============================================================================

from typing import Generic, TypeVar, Optional, Any, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from voice2text.tran.speech2text import STTAsyncVoice2TextService

# ============================================================================
# 标准响应封装
# ============================================================================

T = TypeVar('T')


class ResponseCode(Enum):
    """标准响应码"""
    SUCCESS = 200
    CREATED = 201
    ACCEPTED = 202
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    CONFLICT = 409
    PAYLOAD_TOO_LARGE = 413
    UNPROCESSABLE_ENTITY = 422
    RATE_LIMITED = 429
    INTERNAL_ERROR = 500
    SERVICE_UNAVAILABLE = 503


@dataclass
class ApiResponse(Generic[T]):
    """标准API响应格式"""
    success: bool
    code: int
    message: str
    data: Optional[T] = None
    errors: Optional[List[str]] = None
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def success_response(cls, data: T, message: str = "操作成功", code: int = 200) -> 'ApiResponse[T]':
        """创建成功响应"""
        return cls(
            success=True,
            code=code,
            message=message,
            data=data
        )

    @classmethod
    def error_response(cls, message: str, code: int = 500, errors: Optional[List[str]] = None) -> 'ApiResponse[None]':
        """创建错误响应"""
        return cls(
            success=False,
            code=code,
            message=message,
            errors=errors or []
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "success": self.success,
            "code": self.code,
            "message": self.message,
            "timestamp": self.timestamp.isoformat()
        }

        if self.data is not None:
            result["data"] = self.data

        if self.errors:
            result["errors"] = self.errors

        if self.request_id:
            result["request_id"] = self.request_id

        return result

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


# ============================================================================
# 数据传输对象 (DTOs)
# ============================================================================

@dataclass
class TaskInfo:
    """任务信息DTO"""
    task_id: str
    status: str
    progress: float
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscribeRequest:
    """转写请求DTO"""
    audio_file_id: str  # 上传的文件ID
    threshold: Optional[float] = None
    auto_register_unknown: bool = True
    priority: int = 5
    batch_size_s: int = 300
    hotword: str = ""
    delete_after_processing: bool = True  # 新增：处理后是否删除文件


@dataclass
class VoiceprintRegisterRequest:
    """声纹注册请求DTO"""
    person_name: str
    audio_file_id: str  # 上传的文件ID
    delete_after_processing: bool = True  # 处理后是否删除文件


@dataclass
class TranscribeResult:
    """转写结果DTO"""
    transcript: str
    audio_duration: float
    auto_registered_speakers: Dict[str, Any]
    voiceprint_audio_samples: Dict[str, Any]
    output_file: Optional[str] = None


@dataclass
class VoicePrintInfo:
    """声纹信息DTO"""
    speaker_id: str
    sample_list: List[Dict]
    total_duration: float


@dataclass
class SpeakerStatistics:
    """说话人统计DTO"""
    total_speakers: int
    named_speakers: int
    unnamed_speakers: int
    total_samples: int
    total_duration: float
    average_samples_per_speaker: float


@dataclass
class SimilarVoice:
    """相似声纹DTO"""
    speaker_id: str
    speaker_name: str
    similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileUploadResult:
    """文件上传结果DTO"""
    file_id: str
    filename: str
    file_size: int
    file_type: str
    upload_time: datetime
    category: str


@dataclass
class ServiceStatus:
    """服务状态DTO"""
    service_name: str
    version: str
    status: str
    uptime: float
    total_tasks: int
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    current_load: float
