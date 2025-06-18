# ============================================================================
# Voice2Text SDK - 标准响应类和基础框架
# ============================================================================

from typing import Generic, TypeVar, Optional, Any, Dict, List, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from voice2text.tran.tran2 import VectorConfigFactory, ConfigFactory, VectorAsyncVoice2TextService
from voice2text.tran.vector_base import VectorDBType

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
    audio_file: str  # 文件路径或上传的文件ID
    threshold: Optional[float] = None
    auto_register_unknown: bool = True
    priority: int = 5
    batch_size_s: int = 300
    hotword: str = ""


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
    speaker_name: str
    sample_count: int
    created_at: datetime
    last_updated: datetime
    total_duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


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

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import asyncio
import aiofiles
import uvicorn

from pathlib import Path


class VoiceSDKServer:
    """Voice2Text SDK 服务端"""

    def __init__(self, voice_service, upload_dir: str = "./uploads"):
        self.voice_service = voice_service
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)

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
        async def upload_audio_file(file: UploadFile = File(...)):
            """上传音频文件"""
            try:
                # 验证文件类型
                if not self._is_valid_audio_file(file.filename):
                    return ApiResponse.error_response(
                        "不支持的音频格式",
                        code=ResponseCode.UNPROCESSABLE_ENTITY.value
                    ).to_dict()

                # 生成文件ID和保存路径
                file_id = str(uuid.uuid4())
                file_extension = Path(file.filename).suffix
                save_path = self.upload_dir / f"{file_id}{file_extension}"

                # 异步保存文件
                async with aiofiles.open(save_path, 'wb') as f:
                    content = await file.read()
                    await f.write(content)

                # 创建上传结果
                result = FileUploadResult(
                    file_id=file_id,
                    filename=file.filename,
                    file_size=len(content),
                    file_type=file.content_type,
                    upload_time=datetime.now()
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
                # 获取文件路径
                audio_path = self._get_file_path(request.audio_file)
                if not audio_path:
                    return ApiResponse.error_response(
                        "音频文件不存在",
                        code=ResponseCode.NOT_FOUND.value
                    ).to_dict()

                # 提交异步任务
                task_id = await self.voice_service.transcribe_file_async(
                    str(audio_path),
                    threshold=request.threshold,
                    auto_register_unknown=request.auto_register_unknown,
                    priority=request.priority,
                    batch_size_s=request.batch_size_s,
                    hotword=request.hotword
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
        async def register_voiceprint(person_name: str, audio_file: str):
            """注册声纹"""
            try:
                audio_path = self._get_file_path(audio_file)
                if not audio_path:
                    return ApiResponse.error_response(
                        "音频文件不存在",
                        code=ResponseCode.NOT_FOUND.value
                    ).to_dict()

                task_id = await self.voice_service.register_voice_async(
                    person_name, str(audio_path)
                )

                return ApiResponse.success_response(
                    {"task_id": task_id},
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

                # 转换为标准DTO
                voiceprint_list = []
                for speaker_id, info in voices.items():
                    voiceprint = VoicePrintInfo(
                        speaker_id=speaker_id,
                        speaker_name=info.get("speaker_name", "未命名"),
                        sample_count=info.get("sample_count", 0),
                        created_at=datetime.fromisoformat(info.get("created_at", datetime.now().isoformat())),
                        last_updated=datetime.fromisoformat(info.get("last_updated", datetime.now().isoformat())),
                        total_duration=info.get("total_duration", 0.0),
                        metadata=info.get("metadata", {})
                    )
                    voiceprint_list.append(voiceprint.__dict__)

                return ApiResponse.success_response(voiceprint_list).to_dict()

            except Exception as e:
                return ApiResponse.error_response(
                    f"获取声纹列表失败: {str(e)}",
                    code=ResponseCode.INTERNAL_ERROR.value
                ).to_dict()

        @self.app.post("/api/v1/voiceprints/search")
        async def search_similar_voices(audio_file: str, top_k: int = 10, threshold: float = 0.5):
            """搜索相似声纹"""
            try:
                audio_path = self._get_file_path(audio_file)
                if not audio_path:
                    return ApiResponse.error_response(
                        "音频文件不存在",
                        code=ResponseCode.NOT_FOUND.value
                    ).to_dict()

                similar_voices = await self.voice_service.search_similar_voices(
                    str(audio_path), top_k, threshold
                )

                # 转换为标准DTO
                results = []
                for speaker_id, similarity in similar_voices:
                    similar_voice = SimilarVoice(
                        speaker_id=speaker_id,
                        speaker_name=speaker_id,  # 这里可以查询实际姓名
                        similarity=similarity
                    )
                    results.append(similar_voice.__dict__)

                return ApiResponse.success_response(results).to_dict()

            except Exception as e:
                return ApiResponse.error_response(
                    f"搜索相似声纹失败: {str(e)}",
                    code=ResponseCode.INTERNAL_ERROR.value
                ).to_dict()

        @self.app.put("/api/v1/speakers/{speaker_id}/rename")
        async def rename_speaker(speaker_id: str, new_name: str):
            """重命名说话人"""
            try:
                task_id = await self.voice_service.rename_voice_print_async(speaker_id, new_name)

                return ApiResponse.success_response(
                    {"task_id": task_id},
                    "重命名任务已提交",
                    code=ResponseCode.ACCEPTED.value
                ).to_dict()

            except Exception as e:
                return ApiResponse.error_response(
                    f"重命名说话人失败: {str(e)}",
                    code=ResponseCode.INTERNAL_ERROR.value
                ).to_dict()

        @self.app.delete("/api/v1/speakers/{speaker_id}")
        async def delete_speaker(speaker_id: str):
            """删除说话人"""
            try:
                task_id = await self.voice_service.delete_speaker_async(speaker_id)

                return ApiResponse.success_response(
                    {"task_id": task_id},
                    "删除任务已提交",
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

    def _is_valid_audio_file(self, filename: str) -> bool:
        """验证音频文件格式"""
        if not filename:
            return False

        valid_extensions = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg'}
        return Path(filename).suffix.lower() in valid_extensions

    def _get_file_path(self, file_identifier: str) -> Optional[Path]:
        """获取文件路径"""
        # 如果是文件ID，从上传目录查找
        for ext in ['.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg']:
            file_path = self.upload_dir / f"{file_identifier}{ext}"
            if file_path.exists():
                return file_path

        # 如果是完整路径
        file_path = Path(file_identifier)
        if file_path.exists():
            return file_path

        return None


# ============================================================================
# SDK客户端实现
# ============================================================================

import httpx
from typing import BinaryIO


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

    async def upload_audio_file(self, file_path: str) -> ApiResponse[FileUploadResult]:
        """上传音频文件"""
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
    async def transcribe_audio(self, audio_file: str, **kwargs) -> ApiResponse[TaskInfo]:
        """提交音频转写任务"""
        request_data = TranscribeRequest(audio_file=audio_file, **kwargs).__dict__
        return await self._make_request('POST', '/api/v1/audio/transcribe', json=request_data)

    async def get_task_status(self, task_id: str) -> ApiResponse[TaskInfo]:
        """获取任务状态"""
        return await self._make_request('GET', f'/api/v1/tasks/{task_id}')

    async def get_task_result(self, task_id: str) -> ApiResponse[TranscribeResult]:
        """获取任务结果"""
        return await self._make_request('GET', f'/api/v1/tasks/{task_id}/result')

    async def register_voiceprint(self, person_name: str, audio_file: str) -> ApiResponse[Dict]:
        """注册声纹"""
        params = {'person_name': person_name, 'audio_file': audio_file}
        return await self._make_request('POST', '/api/v1/voiceprints/register', params=params)

    async def list_voiceprints(self, include_unnamed: bool = True) -> ApiResponse[List[VoicePrintInfo]]:
        """获取声纹列表"""
        params = {'include_unnamed': include_unnamed}
        return await self._make_request('GET', '/api/v1/voiceprints/list', params=params)

    async def search_similar_voices(self, audio_file: str, top_k: int = 10, threshold: float = 0.5) -> ApiResponse[
        List[SimilarVoice]]:
        """搜索相似声纹"""
        params = {'audio_file': audio_file, 'top_k': top_k, 'threshold': threshold}
        return await self._make_request('POST', '/api/v1/voiceprints/search', params=params)

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

        # 2. 上传音频文件
        upload_result = await client.upload_audio_file("../../data/刘星家_20231212_122300_家有儿女吃饭.mp3")
        if upload_result.success:
            file_id = upload_result.data['file_id']
            print(f"文件上传成功，文件ID: {file_id}")

            # 3. 提交转写任务
            transcribe_task = await client.transcribe_audio(file_id)
            if transcribe_task.success:
                task_id = transcribe_task.data['task_id']
                print(f"转写任务已提交: {task_id}")

                # 4. 等待任务完成
                result = await client.wait_for_task_completion(task_id)
                if result.success:
                    print(f"转写结果: {result.data['transcript']}")
                else:
                    print(f"转写失败: {result.message}")

        # 5. 获取声纹列表
        voiceprints = await client.list_voiceprints()
        if voiceprints.success:
            print(f"已注册声纹数量: {len(voiceprints.data)}")


if __name__ == "__main__":


    # 运行示例
    asyncio.run(example_usage())