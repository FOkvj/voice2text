# ============================================================================
# Voice2Text SDK - 更新的API实现，集成文件管理器
# ============================================================================

from typing import Generic, TypeVar, Optional, Any, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from voice2text.tran.filesystem import StorageType, S3StorageConfig
from voice2text.tran.schema.dto import ServiceStatus, ApiResponse, ResponseCode, FileUploadResult, TranscribeRequest, \
    TaskInfo, TranscribeResult, VoiceprintRegisterRequest, VoicePrintInfo, SpeakerStatistics
from voice2text.tran.schema.prints import SampleInfo
from voice2text.tran.speech2text import STTAsyncVoice2TextService, STTConfigFactory

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

from voice2text.tran.vector_base import VectorDBType, ChromaDBConfig


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

    def __init__(self, voice_service: STTAsyncVoice2TextService):
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
                sample_info: SampleInfo = await self.voice_service.register_voice_async(
                    request.person_name,
                    input_data
                )

                return ApiResponse.success_response(
                    sample_info,
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
                voice_prints = await self.voice_service.list_registered_voices_async(include_unnamed)
                return ApiResponse.success_response(voice_prints).to_dict()

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



async def start_server():
    import uvicorn
    # 1. 创建语音服务实例
    asr_config = STTConfigFactory.create_funasr_config(
        model_name="paraformer-zh",
        device="cpu"
    )

    speaker_config = STTConfigFactory.create_speaker_config(
        threshold=0.5,
        device="cpu"
    )

    vector_db_config = ChromaDBConfig(db_type=VectorDBType.CHROMADB, persist_directory='./voice_vectors', collection_name='voice_prints')

    s3_config = S3StorageConfig(
        storage_type=StorageType.S3,
        bucket_name="voice",
        endpoint_url="http://localhost:9000",
        access_key_id="admin",
        secret_access_key="minioadmin123",
        prefix="stt"
    )

    # 创建向量服务配置
    service_config = STTConfigFactory.create_stt_service_config(
        asr_config=asr_config,
        speaker_config=speaker_config,
        vector_db_config=vector_db_config,
        storage_config=s3_config,
        max_transcribe_concurrent=2,
        max_speaker_concurrent=3,
        task_timeout=300.0
    )

    voice_service = STTAsyncVoice2TextService(service_config)
    await voice_service.start()
    # 2. 创建并启动服务器
    server = VoiceSDKServer(voice_service)

    # 3. 使用uvicorn运行FastAPI应用
    config = uvicorn.Config(
        app=server.app,
        host="0.0.0.0",
        port=8765,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(start_server())