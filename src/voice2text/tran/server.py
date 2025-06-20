# ============================================================================
# Voice2Text SDK - 更新的API实现，集成文件管理器
# ============================================================================

from typing import Generic, TypeVar, Optional, Any, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from voice2text.tran.speech2text import VectorAsyncVoice2TextService

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


# ============================================================================
# FastAPI服务端实现
# ============================================================================

from fastapi import FastAPI, File, UploadFile, Query, Path as FastAPIPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uuid
import asyncio
import aiofiles
import io

from pathlib import Path


def parse_filename(audio_file_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """解析文件名获取地点和时间信息"""
    import os
    from datetime import datetime

    filename = os.path.basename(audio_file_path)
    name_without_ext = os.path.splitext(filename)[0]

    # 尝试解析格式: 地点_日期_时间_名称
    parts = name_without_ext.split('_')
    if len(parts) >= 3:
        location = parts[0]
        date_str = parts[1]
        time_str = parts[2]

        # 验证日期时间格式
        try:
            datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M%S")
            # 转换为标准格式
            formatted_date = f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:8]}"
            formatted_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
            return location, formatted_date, formatted_time
        except ValueError:
            pass

    print(f"警告: 文件名 {filename} 不符合标准格式")
    return None, None, None


class VoiceSDKServer:
    """Voice2Text SDK 服务端"""

    def __init__(self, voice_service: VectorAsyncVoice2TextService):
        """
        初始化服务端

        Args:
            voice_service: 语音服务实例
            file_manager: 文件管理器实例
        """
        self.voice_service = voice_service
        self.file_manager = voice_service.file_manager

        # 创建FastAPI应用
        self.app = FastAPI(
            title="Voice2Text SDK API",
            description="高性能语音转文字服务API",
            version="1.0.0"
        )

        # 配置CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 注册路由
        self._register_routes()

    def _register_routes(self):
        """注册所有API路由"""

        @self.app.get("/health", response_model=Dict)
        async def health_check():
            """健康检查"""
            stats = self.voice_service.get_service_stats()
            status = ServiceStatus(
                service_name="Voice2Text SDK",
                version="1.0.0",
                status="healthy",
                uptime=stats.get('uptime', 0),
                total_tasks=stats.get('total_tasks', 0),
                active_tasks=len(self.voice_service.list_active_tasks()),
                completed_tasks=stats.get('completed_tasks', 0),
                failed_tasks=stats.get('failed_tasks', 0),
                current_load=stats.get('current_load', 0)
            )

            return ApiResponse.success_response(status.__dict__).to_dict()

        @self.app.post("/api/v1/audio/upload")
        async def upload_audio_file(
                file: UploadFile = File(...),
                category: str = Query(default="transcribe", description="文件分类：transcribe(转写) 或 voiceprint(声纹)")
        ):
            """
            上传音频文件

            Args:
                file: 上传的音频文件
                category: 文件分类，支持 'transcribe'(转写) 或 'voiceprint'(声纹注册)
            """
            try:
                # 验证文件类型
                if not self._is_valid_audio_file(file.filename):
                    return ApiResponse.error_response(
                        "不支持的音频格式",
                        code=ResponseCode.UNPROCESSABLE_ENTITY.value
                    ).to_dict()

                # 验证分类
                if category not in ["transcribe", "voiceprint"]:
                    return ApiResponse.error_response(
                        "分类必须是 'transcribe' 或 'voiceprint'",
                        code=ResponseCode.BAD_REQUEST.value
                    ).to_dict()

                # 读取文件内容
                content = await file.read()

                # 使用文件管理器保存文件
                file_id = await self.file_manager.save_file(
                    data=content,
                    filename=file.filename,
                    category=category,
                    metadata={
                        "original_filename": file.filename,
                        "upload_source": "api"
                    },
                    content_type=file.content_type
                )

                # 创建上传结果
                result = FileUploadResult(
                    file_id=file_id,
                    filename=file.filename,
                    file_size=len(content),
                    file_type=file.content_type,
                    upload_time=datetime.now(),
                    category=category
                )

                return ApiResponse.success_response(
                    result.__dict__,
                    "文件上传成功",
                    code=ResponseCode.CREATED.value
                ).to_dict()

            except Exception as e:
                return ApiResponse.error_response(
                    f"文件上传失败: {str(e)}",
                    code=ResponseCode.INTERNAL_ERROR.value
                ).to_dict()

        @self.app.post("/api/v1/audio/transcribe")
        async def transcribe_audio(request: TranscribeRequest):
            """提交音频转写任务"""
            try:
                # 获取文件元数据
                meta, input_data = await self.file_manager.load_file(request.audio_file_id)
                if not meta:
                    return ApiResponse.error_response(
                        "文件不存在",
                        code=ResponseCode.NOT_FOUND.value
                    ).to_dict()

                # 验证文件分类
                if meta.category != "transcribe":
                    return ApiResponse.error_response(
                        "文件分类错误，该文件不是用于转写的音频文件",
                        code=ResponseCode.BAD_REQUEST.value
                    ).to_dict()

                # 加载文件数据
                if not input_data:
                    return ApiResponse.error_response(
                        "无法读取文件数据",
                        code=ResponseCode.INTERNAL_ERROR.value
                    ).to_dict()

                # 解析文件名信息
                file_location, file_date, file_time = parse_filename(meta.filename)

                # 提交异步任务
                task_id = await self.voice_service.transcribe_file_async(
                    audio_input=input_data,
                    threshold=request.threshold,
                    auto_register_unknown=request.auto_register_unknown,
                    priority=request.priority,
                    batch_size_s=request.batch_size_s,
                    hotword=request.hotword,
                    file_location=file_location,
                    file_date=file_date,
                    file_time=file_time,
                    metadata={
                        "source_file_id": request.audio_file_id,
                        "delete_after_processing": request.delete_after_processing
                    }
                )

                # 获取任务信息
                task_info = self.voice_service.task_manager.get_task_status(task_id)
                task_dto = TaskInfo(
                    task_id=task_id,
                    status=task_info.status.value,
                    progress=task_info.progress,
                    created_at=datetime.fromtimestamp(task_info.created_at),
                    metadata=task_info.metadata
                )

                return ApiResponse.success_response(
                    task_dto.__dict__,
                    "转写任务已提交",
                    code=ResponseCode.ACCEPTED.value
                ).to_dict()

            except Exception as e:
                return ApiResponse.error_response(
                    f"提交转写任务失败: {str(e)}",
                    code=ResponseCode.INTERNAL_ERROR.value
                ).to_dict()

        @self.app.get("/api/v1/tasks/{task_id}")
        async def get_task_status(task_id: str):
            """获取任务状态"""
            try:
                progress = await self.voice_service.get_task_progress(task_id)

                if "error" in progress:
                    return ApiResponse.error_response(
                        progress["error"],
                        code=ResponseCode.NOT_FOUND.value
                    ).to_dict()

                task_dto = TaskInfo(
                    task_id=progress["task_id"],
                    status=progress["status"],
                    progress=progress["progress"],
                    created_at=datetime.fromtimestamp(progress["created_at"]),
                    started_at=datetime.fromtimestamp(progress["started_at"]) if progress.get("started_at") else None,
                    completed_at=datetime.fromtimestamp(progress["completed_at"]) if progress.get(
                        "completed_at") else None,
                    metadata=progress.get("metadata", {})
                )

                return ApiResponse.success_response(task_dto.__dict__).to_dict()

            except Exception as e:
                return ApiResponse.error_response(
                    f"获取任务状态失败: {str(e)}",
                    code=ResponseCode.INTERNAL_ERROR.value
                ).to_dict()

        @self.app.get("/api/v1/tasks/{task_id}/result")
        async def get_task_result(task_id: str):
            """获取任务结果"""
            try:
                result = await self.voice_service.get_transcribe_result(task_id)

                # 检查是否需要删除源文件
                task_info = self.voice_service.task_manager.get_task_status(task_id)
                if task_info and task_info.metadata.get("delete_after_processing", True):
                    source_file_id = task_info.metadata.get("source_file_id")
                    if source_file_id:
                        try:
                            await self.file_manager.delete_file(source_file_id)
                            print(f"已删除源文件: {source_file_id}")
                        except Exception as e:
                            print(f"删除源文件失败: {e}")

                # 转换为标准DTO
                result_dto = TranscribeResult(
                    transcript=result["transcript"],
                    audio_duration=result["audio_duration"],
                    auto_registered_speakers=result["auto_registered_speakers"],
                    voiceprint_audio_samples=result["voiceprint_audio_samples"],
                    output_file=result.get("output_file")
                )

                return ApiResponse.success_response(result_dto.__dict__).to_dict()

            except Exception as e:
                return ApiResponse.error_response(
                    f"获取任务结果失败: {str(e)}",
                    code=ResponseCode.INTERNAL_ERROR.value
                ).to_dict()

        @self.app.post("/api/v1/voiceprints/register")
        async def register_voiceprint(request: VoiceprintRegisterRequest):
            """注册声纹"""
            try:
                # 获取文件元数据
                meta, input_data = await self.file_manager.load_file(request.audio_file_id)
                if not meta:
                    return ApiResponse.error_response(
                        "文件不存在",
                        code=ResponseCode.NOT_FOUND.value
                    ).to_dict()

                # 验证文件分类
                if meta.category != "voiceprint":
                    return ApiResponse.error_response(
                        "文件分类错误，该文件不是用于声纹注册的音频文件",
                        code=ResponseCode.BAD_REQUEST.value
                    ).to_dict()

                # 提交声纹注册任务
                person_name, speaker_id = await self.voice_service.register_voice_async(
                    request.person_name,
                    input_data
                )

                return ApiResponse.success_response(
                    {"speaker_id": speaker_id, "person_name": person_name},
                    "声纹注册任务已提交",
                    code=ResponseCode.ACCEPTED.value
                ).to_dict()

            except Exception as e:
                return ApiResponse.error_response(
                    f"注册声纹失败: {str(e)}",
                    code=ResponseCode.INTERNAL_ERROR.value
                ).to_dict()

        @self.app.get("/api/v1/voiceprints/list")
        async def list_voiceprints(include_unnamed: bool = True):
            """获取声纹列表"""
            try:
                voices = await self.voice_service.list_registered_voices_async(include_unnamed)
                all_voices = {}
                all_voices.update(voices["named_voice_prints"])
                all_voices.update(voices["unnamed_voice_prints"])
                # 转换为标准DTO
                voiceprint_list = []
                for speaker_id, samples in all_voices.items():
                    voiceprint = VoicePrintInfo(
                        speaker_id=speaker_id,
                        sample_list=samples,
                        total_duration=sum(s['audio_duration'] for s in samples)
                    )
                    voiceprint_list.append(voiceprint.__dict__)

                return ApiResponse.success_response(voiceprint_list).to_dict()

            except Exception as e:
                return ApiResponse.error_response(
                    f"获取声纹列表失败: {str(e)}",
                    code=ResponseCode.INTERNAL_ERROR.value
                ).to_dict()


        @self.app.put("/api/v1/speakers/{speaker_id}/rename")
        async def rename_speaker(speaker_id: str, new_name: str):
            """重命名说话人"""
            try:
                success = await self.voice_service.voice_print_manager.rename_voice_print(speaker_id, new_name)

                if success:
                    return ApiResponse.success_response(
                        {"new_name": new_name},
                        f"{speaker_id}已重命名为{new_name}",
                        code=ResponseCode.ACCEPTED.value
                    ).to_dict()
                return ApiResponse.error_response(
                    "重命名失败",
                )
            except Exception as e:
                return ApiResponse.error_response(
                    f"重命名说话人失败: {str(e)}",
                    code=ResponseCode.INTERNAL_ERROR.value
                ).to_dict()

        @self.app.delete("/api/v1/speakers/{speaker_id}/samples/{file_id}")
        async def delete_speaker_audio_sample(
                speaker_id: str = FastAPIPath(..., description="说话人ID"),
                file_id: str = FastAPIPath(..., description="音频文件ID")
        ):
            """删除特定说话人的音频样本"""
            try:

                # 直接调用删除方法
                success = await self.voice_service.voice_print_manager.delete_speaker_audio_sample(speaker_id,
                                                                                                  file_id)

                if success:
                    return ApiResponse.success_response(
                        {"deleted": True},
                        "音频样本删除成功"
                    ).to_dict()
                else:
                    return ApiResponse.error_response(
                        "音频样本不存在或删除失败",
                        code=ResponseCode.NOT_FOUND.value
                    ).to_dict()

            except Exception as e:
                return ApiResponse.error_response(
                    f"删除音频样本失败: {str(e)}",
                    code=ResponseCode.INTERNAL_ERROR.value
                ).to_dict()

        @self.app.delete("/api/v1/speakers/{speaker_id}")
        async def delete_speaker(speaker_id: str):
            """删除说话人"""
            try:
                success = await self.voice_service.voice_print_manager.delete_speaker(speaker_id)

                if not success:
                    return ApiResponse.error_response(
                        "说话人不存在或删除失败",
                        code=ResponseCode.NOT_FOUND.value
                    ).to_dict()
                return ApiResponse.success_response(
                    speaker_id,
                    "删除成功",
                    code=ResponseCode.ACCEPTED.value
                ).to_dict()

            except Exception as e:
                return ApiResponse.error_response(
                    f"删除说话人失败: {str(e)}",
                    code=ResponseCode.INTERNAL_ERROR.value
                ).to_dict()

        @self.app.get("/api/v1/statistics")
        async def get_statistics():
            """获取统计信息"""
            try:
                stats = await self.voice_service.get_speaker_statistics_async()

                statistics = SpeakerStatistics(
                    total_speakers=stats.get("total_speakers", 0),
                    named_speakers=stats.get("named_speakers", 0),
                    unnamed_speakers=stats.get("unnamed_speakers", 0),
                    total_samples=stats.get("total_samples", 0),
                    total_duration=stats.get("total_duration", 0.0),
                    average_samples_per_speaker=stats.get("average_samples_per_speaker", 0.0)
                )

                return ApiResponse.success_response(statistics.__dict__).to_dict()

            except Exception as e:
                return ApiResponse.error_response(
                    f"获取统计信息失败: {str(e)}",
                    code=ResponseCode.INTERNAL_ERROR.value
                ).to_dict()

        @self.app.get("/api/v1/files/list")
        async def list_files(category: Optional[str] = Query(None, description="文件分类过滤")):
            """列出已上传的文件"""
            try:
                files = self.file_manager.list_files(category=category)
                return ApiResponse.success_response(files).to_dict()

            except Exception as e:
                return ApiResponse.error_response(
                    f"获取文件列表失败: {str(e)}",
                    code=ResponseCode.INTERNAL_ERROR.value
                ).to_dict()

        @self.app.get("/api/v1/files/{file_id}/download")
        async def download_file(file_id: str = FastAPIPath(..., description="文件ID")):
            """下载文件"""
            try:
                # 获取文件元数据
                meta = self.file_manager.get_file_by_id(file_id)
                if not meta:
                    return ApiResponse.error_response(
                        "文件不存在",
                        code=ResponseCode.NOT_FOUND.value
                    ).to_dict()

                # 加载文件内容
                file_content = await self.file_manager.load_file(file_id)
                if not file_content:
                    return ApiResponse.error_response(
                        "无法读取文件内容",
                        code=ResponseCode.INTERNAL_ERROR.value
                    ).to_dict()

                # 创建流式响应
                def iter_content():
                    yield file_content

                return StreamingResponse(
                    iter_content(),
                    media_type=meta.content_type or "application/octet-stream",
                    headers={
                        "Content-Disposition": f"attachment; filename={meta.filename}",
                        "Content-Length": str(meta.size)
                    }
                )

            except Exception as e:
                return ApiResponse.error_response(
                    f"下载文件失败: {str(e)}",
                    code=ResponseCode.INTERNAL_ERROR.value
                ).to_dict()

        @self.app.delete("/api/v1/files/{file_id}")
        async def delete_file(file_id: str = FastAPIPath(..., description="文件ID")):
            """删除文件"""
            try:
                success = await self.file_manager.delete_file(file_id)
                if success:
                    return ApiResponse.success_response(
                        {"deleted": True},
                        "文件删除成功"
                    ).to_dict()
                else:
                    return ApiResponse.error_response(
                        "文件不存在或删除失败",
                        code=ResponseCode.NOT_FOUND.value
                    ).to_dict()

            except Exception as e:
                return ApiResponse.error_response(
                    f"删除文件失败: {str(e)}",
                    code=ResponseCode.INTERNAL_ERROR.value
                ).to_dict()

        @self.app.get("/api/v1/files/{file_id}/info")
        async def get_file_info(file_id: str = FastAPIPath(..., description="文件ID")):
            """获取文件信息"""
            try:
                meta = self.file_manager.get_file_by_id(file_id)
                if not meta:
                    return ApiResponse.error_response(
                        "文件不存在",
                        code=ResponseCode.NOT_FOUND.value
                    ).to_dict()

                file_info = {
                    "file_id": meta.file_id,
                    "filename": meta.filename,
                    "category": meta.category,
                    "size": meta.size,
                    "hash": meta.hash,
                    "created_at": meta.created_at.isoformat(),
                    "updated_at": meta.updated_at.isoformat() if meta.updated_at else None,
                    "storage_type": meta.storage_type.value,
                    "content_type": meta.content_type,
                    "metadata": meta.metadata
                }

                return ApiResponse.success_response(file_info).to_dict()

            except Exception as e:
                return ApiResponse.error_response(
                    f"获取文件信息失败: {str(e)}",
                    code=ResponseCode.INTERNAL_ERROR.value
                ).to_dict()

    def _is_valid_audio_file(self, filename: str) -> bool:
        """验证音频文件格式"""
        if not filename:
            return False

        valid_extensions = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg'}
        return Path(filename).suffix.lower() in valid_extensions


# ============================================================================
# SDK客户端实现
# ============================================================================

import httpx


class VoiceSDKClient:
    """Voice2Text SDK 客户端"""

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout

        # 设置请求头
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> ApiResponse:
        """发送HTTP请求并返回标准响应"""
        try:
            response = await self.client.request(method, endpoint, **kwargs)
            response_data = response.json()

            # 转换为标准ApiResponse
            return ApiResponse(
                success=response_data.get('success', False),
                code=response_data.get('code', response.status_code),
                message=response_data.get('message', ''),
                data=response_data.get('data'),
                errors=response_data.get('errors'),
                request_id=response_data.get('request_id'),
                timestamp=datetime.fromisoformat(response_data.get('timestamp', datetime.now().isoformat()))
            )

        except httpx.RequestError as e:
            return ApiResponse.error_response(f"请求失败: {str(e)}")
        except Exception as e:
            return ApiResponse.error_response(f"未知错误: {str(e)}")

    async def health_check(self) -> ApiResponse[ServiceStatus]:
        """健康检查"""
        return await self._make_request('GET', '/health')

    async def upload_audio_file(self, file_path: str, category: str = "transcribe") -> ApiResponse[FileUploadResult]:
        """
        上传音频文件

        Args:
            file_path: 本地文件路径
            category: 文件分类，'transcribe' 或 'voiceprint'
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return ApiResponse.error_response("文件不存在", code=ResponseCode.NOT_FOUND.value)

            async with aiofiles.open(file_path, 'rb') as f:
                file_content = await f.read()
                files = {'file': (path.name, file_content)}

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/api/v1/audio/upload",
                        files=files,
                        params={'category': category},
                        headers={'Authorization': f'Bearer {self.api_key}'} if self.api_key else {},
                        timeout=self.timeout
                    )

            response_data = response.json()
            return ApiResponse(
                success=response_data.get("success", False),
                code=response_data.get("code", response.status_code),
                message=response_data.get("message", ""),
                data=response_data.get("data"),
                errors=response_data.get("errors"),
            )
        except Exception as e:
            return ApiResponse.error_response(f"上传文件失败: {str(e)}")

    async def transcribe_file_direct(
            self,
            file_path: str,
            wait_for_completion: bool = True,
            delete_after_processing: bool = True,
            poll_interval: float = 1.0,
            timeout: float = 300.0,
            **transcribe_kwargs
    ) -> ApiResponse[Dict]:
        """
        直接转写文件：上传 -> 转写 -> 获取结果 一键完成

        Args:
            file_path: 本地音频文件路径
            wait_for_completion: 是否等待任务完成
            delete_after_processing: 处理后是否删除文件
            poll_interval: 轮询间隔(秒)
            timeout: 等待超时时间(秒)
            **transcribe_kwargs: 其他转写参数(threshold, auto_register_unknown等)

        Returns:
            如果wait_for_completion=True，返回转写结果
            如果wait_for_completion=False，返回任务信息
        """
        try:
            # 1. 上传文件
            upload_result = await self.upload_audio_file(file_path, category="transcribe")
            if not upload_result.success:
                return upload_result

            file_id = upload_result.data['file_id']

            try:
                # 2. 提交转写任务
                transcribe_task = await self.transcribe_audio(
                    file_id,
                    delete_after_processing=delete_after_processing,
                    **transcribe_kwargs
                )
                if not transcribe_task.success:
                    # 转写任务提交失败，清理上传的文件
                    await self.delete_file(file_id)
                    return transcribe_task

                task_id = transcribe_task.data['task_id']

                # 3. 如果不等待完成，直接返回任务信息
                if not wait_for_completion:
                    return ApiResponse.success_response(
                        {
                            "task_id": task_id,
                            "file_id": file_id,
                            "task_info": transcribe_task.data,
                            "message": "转写任务已提交，使用task_id查询进度"
                        },
                        "转写任务已提交"
                    )

                # 4. 等待任务完成并获取结果
                result = await self.wait_for_task_completion(task_id, poll_interval, timeout)

                if result.success:
                    if delete_after_processing:
                        # 任务成功且需要删除文件
                        await self.delete_file(file_id)
                    return ApiResponse.success_response(
                        {
                            "task_id": task_id,
                            "file_id": file_id,
                            "transcript": result.data['transcript'],
                            "audio_duration": result.data['audio_duration'],
                            "auto_registered_speakers": result.data['auto_registered_speakers'],
                            "voiceprint_audio_samples": result.data['voiceprint_audio_samples'],
                            "output_file": result.data.get('output_file')
                        },
                        "转写完成"
                    )
                else:
                    # 任务失败，手动清理文件（如果需要）
                    if not delete_after_processing:
                        await self.delete_file(file_id)
                    return result

            except Exception as e:
                # 出错时清理上传的文件
                try:
                    await self.delete_file(file_id)
                except:
                    pass
                return ApiResponse.error_response(f"转写过程中出错: {str(e)}")

        except Exception as e:
            return ApiResponse.error_response(f"转写文件失败: {str(e)}")

    async def register_voiceprint_direct(
            self,
            person_name: str,
            file_path: str,
    ) -> ApiResponse[Dict]:
        """
        直接注册声纹：上传 -> 注册 一键完成

        Args:
            person_name: 人员姓名
            file_path: 本地音频文件路径
            delete_after_processing: 处理后是否删除文件

        Returns:
            声纹注册结果
        """
        try:
            # 1. 上传文件
            upload_result = await self.upload_audio_file(file_path, category="voiceprint")
            if not upload_result.success:
                return upload_result

            file_id = upload_result.data['file_id']

            try:
                # 2. 注册声纹
                register_result = await self.register_voiceprint(
                    person_name,
                    file_id
                )

                if register_result.success:
                    return ApiResponse.success_response(
                        {
                            "speaker_id": register_result.data['speaker_id'],
                            "person_name": register_result.data['person_name'],
                            "file_id": file_id,
                            "original_filename": Path(file_path).name
                        },
                        "声纹注册成功"
                    )
                else:
                    # 注册失败，清理上传的文件
                    await self.delete_file(file_id)
                    return register_result

            except Exception as e:
                # 出错时清理上传的文件
                try:
                    await self.delete_file(file_id)
                except:
                    pass
                return ApiResponse.error_response(f"声纹注册过程中出错: {str(e)}")

        except Exception as e:
            return ApiResponse.error_response(f"声纹注册失败: {str(e)}")

    async def delete_speaker_audio_sample(self, speaker_id: str, file_id: str) -> ApiResponse[Dict]:
        """
        删除特定说话人的音频样本

        Args:
            speaker_id: 说话人ID
            file_id: 音频文件ID
        """
        return await self._make_request('DELETE', f'/api/v1/speakers/{speaker_id}/samples/{file_id}')

    async def transcribe_audio(self, audio_file_id: str, delete_after_processing: bool = True, **kwargs) -> ApiResponse[
        TaskInfo]:
        """
        提交音频转写任务

        Args:
            audio_file_id: 上传文件返回的file_id
            delete_after_processing: 处理后是否删除文件
            **kwargs: 其他转写参数
        """
        request_data = TranscribeRequest(
            audio_file_id=audio_file_id,
            delete_after_processing=delete_after_processing,
            **kwargs
        ).__dict__
        return await self._make_request('POST', '/api/v1/audio/transcribe', json=request_data)

    async def register_voiceprint(self, person_name: str, audio_file_id: str) -> \
    ApiResponse[Dict]:
        """
        注册声纹

        Args:
            person_name: 人员姓名
            audio_file_id: 上传文件返回的file_id
            delete_after_processing: 处理后是否删除文件
        """
        request_data = VoiceprintRegisterRequest(
            person_name=person_name,
            audio_file_id=audio_file_id
        ).__dict__
        return await self._make_request('POST', '/api/v1/voiceprints/register', json=request_data)

    async def get_task_status(self, task_id: str) -> ApiResponse[TaskInfo]:
        """获取任务状态"""
        return await self._make_request('GET', f'/api/v1/tasks/{task_id}')

    async def get_task_result(self, task_id: str) -> ApiResponse[TranscribeResult]:
        """获取任务结果"""
        return await self._make_request('GET', f'/api/v1/tasks/{task_id}/result')

    async def list_voiceprints(self, include_unnamed: bool = True) -> ApiResponse[List[VoicePrintInfo]]:
        """获取声纹列表"""
        params = {'include_unnamed': include_unnamed}
        return await self._make_request('GET', '/api/v1/voiceprints/list', params=params)


    async def rename_speaker(self, speaker_id: str, new_name: str) -> ApiResponse[Dict]:
        """重命名说话人"""
        params = {'new_name': new_name}
        return await self._make_request('PUT', f'/api/v1/speakers/{speaker_id}/rename', params=params)

    async def delete_speaker(self, speaker_id: str) -> ApiResponse[Dict]:
        """删除说话人"""
        return await self._make_request('DELETE', f'/api/v1/speakers/{speaker_id}')

    async def get_statistics(self) -> ApiResponse[SpeakerStatistics]:
        """获取统计信息"""
        return await self._make_request('GET', '/api/v1/statistics')

    async def list_files(self, category: Optional[str] = None) -> ApiResponse[List[Dict]]:
        """列出文件"""
        params = {'category': category} if category else {}
        return await self._make_request('GET', '/api/v1/files/list', params=params)

    async def download_file(self, file_id: str, save_path: str) -> ApiResponse[Dict]:
        """
        下载文件

        Args:
            file_id: 文件ID
            save_path: 保存路径
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/files/{file_id}/download",
                    headers={'Authorization': f'Bearer {self.api_key}'} if self.api_key else {},
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    async with aiofiles.open(save_path, 'wb') as f:
                        async for chunk in response.aiter_bytes():
                            await f.write(chunk)

                    return ApiResponse.success_response(
                        {"saved_path": save_path, "size": len(response.content)},
                        "文件下载成功"
                    )
                else:
                    error_data = response.json()
                    return ApiResponse.error_response(
                        error_data.get('message', '下载失败'),
                        code=response.status_code
                    )

        except Exception as e:
            return ApiResponse.error_response(f"下载文件失败: {str(e)}")

    async def delete_file(self, file_id: str) -> ApiResponse[Dict]:
        """删除文件"""
        return await self._make_request('DELETE', f'/api/v1/files/{file_id}')

    async def get_file_info(self, file_id: str) -> ApiResponse[Dict]:
        """获取文件信息"""
        return await self._make_request('GET', f'/api/v1/files/{file_id}/info')

    async def wait_for_task_completion(self, task_id: str, poll_interval: float = 1.0, timeout: float = 300.0) -> \
    ApiResponse[TranscribeResult]:
        """等待任务完成并返回结果"""
        import time
        start_time = time.time()
        while time.time() - start_time < timeout:
            status_response = await self.get_task_status(task_id)

            if not status_response.success:
                return status_response

            task_status = status_response.data.get('status')

            if task_status == 'completed':
                return await self.get_task_result(task_id)
            elif task_status == 'failed':
                return ApiResponse.error_response("任务执行失败")
            elif task_status == 'cancelled':
                return ApiResponse.error_response("任务已取消")

            await asyncio.sleep(poll_interval)

        return ApiResponse.error_response("等待任务完成超时")


# ============================================================================
# 使用示例
# ============================================================================

async def example_usage():
    """SDK使用示例"""

    # 客户端使用示例
    async with VoiceSDKClient("http://localhost:8765") as client:
        # 1. 健康检查
        health = await client.health_check()
        print(f"服务状态: {health.message}")

        # 2. 上传音频文件用于转写
        upload_result = await client.upload_audio_file(
            "../../data/刘星家_20231212_122300_家有儿女吃饭.mp3",
            category="transcribe"
        )
        if upload_result.success:
            file_id = upload_result.data['file_id']
            print(f"转写文件上传成功，文件ID: {file_id}")

            # 3. 提交转写任务（处理后保留文件）
            transcribe_task = await client.transcribe_audio(
                file_id,
                delete_after_processing=False
            )
            if transcribe_task.success:
                task_id = transcribe_task.data['task_id']
                print(f"转写任务已提交: {task_id}")

                # 4. 等待任务完成
                result = await client.wait_for_task_completion(task_id)
                if result.success:
                    print(f"转写结果: {result.data['transcript']}")
                else:
                    print(f"转写失败: {result.message}")

        # 5. 上传音频文件用于声纹注册
        voiceprint_upload = await client.upload_audio_file(
            "../../data/sample_voice.wav",
            category="voiceprint"
        )
        if voiceprint_upload.success:
            voiceprint_file_id = voiceprint_upload.data['file_id']
            print(f"声纹文件上传成功，文件ID: {voiceprint_file_id}")

            # 6. 注册声纹
            register_result = await client.register_voiceprint(
                "张三",
                voiceprint_file_id,
                delete_after_processing=True
            )
            if register_result.success:
                print(f"声纹注册任务已提交: {register_result.data['task_id']}")

        # 7. 获取文件列表
        files = await client.list_files()
        if files.success:
            print(f"文件列表: {len(files.data)} 个文件")
            for file_info in files.data:
                print(f"  - {file_info['filename']} ({file_info['category']})")

        # 8. 获取声纹列表
        voiceprints = await client.list_voiceprints()
        if voiceprints.success:
            print(f"已注册声纹数量: {len(voiceprints.data)}")

        # 9. 下载文件示例
        if upload_result.success:
            download_result = await client.download_file(
                file_id,
                "./downloaded_file.mp3"
            )
            if download_result.success:
                print(f"文件下载成功: {download_result.data['saved_path']}")



# ============================================================================
# 使用示例
# ============================================================================

async def example_usage():
    """SDK使用示例"""

    # 客户端使用示例
    async with VoiceSDKClient("http://localhost:8765") as client:
        # 1. 健康检查
        health = await client.health_check()
        print(f"服务状态: {health.message}")
        await client.rename_speaker("Speaker_68dc353b", "夏东海2")
        # 注册声纹
        #
        # upload_voice = await client.upload_audio_file("../../data/sample/刘星.mp3", category="voiceprint")
        # if upload_voice.success:
        #     register_result = await client.register_voiceprint("刘星", upload_voice.data['file_id'])
        #
        #
        # # 2. 上传音频文件
        # upload_result = await client.upload_audio_file("../../data/刘星家_20231212_122300_家有儿女吃饭.mp3")
        # if upload_result.success:
        #     file_id = upload_result.data['file_id']
        #     print(f"文件上传成功，文件ID: {file_id}")
        #
        #     # 3. 提交转写任务
        #     transcribe_task = await client.transcribe_audio(file_id)
        #     if transcribe_task.success:
        #         task_id = transcribe_task.data['task_id']
        #         print(f"转写任务已提交: {task_id}")
        #
        #         # 4. 等待任务完成
        #         result = await client.wait_for_task_completion(task_id)
        #         if result.success:
        #             print(f"转写结果: {result.data['transcript']}")
        #         else:
        #             print(f"转写失败: {result.message}")

        r1 = await client.register_voiceprint_direct("刘星", "../../data/sample/刘星.mp3")
        print(f"声纹注册结果: {r1.message}, 数据: {r1.data}")
        r2 = await client.transcribe_file_direct("../../data/刘星家_20231212_122300_家有儿女吃饭.mp3")
        print(f"转写结果: {r2.message}, 数据: {r2.data}")
        # 5. 获取声纹列表
        voiceprints = await client.list_voiceprints()
        if voiceprints.success:
            print(f"已注册声纹: {voiceprints.data}")
        # await client.delete_speaker("刘星")
        #
        # await client.delete_speaker_audio_sample("夏东海2", "d178f549-d74e-4437-9538-d5b0e1f53853")
        #
        # voiceprints = await client.list_voiceprints()
        # if voiceprints.success:
        #     print(f"已注册声纹: {voiceprints.data}")

        # await client.delete_speaker()


if __name__ == "__main__":


    # 运行示例
    asyncio.run(example_usage())