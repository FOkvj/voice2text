import asyncio
import mimetypes
import os
import tempfile
import urllib
from http.client import HTTPException

from typing import Generic, TypeVar, Optional, Any, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from voice2text.tran.filesystem import StorageType, S3StorageConfig
from voice2text.tran.schema.dto import ServiceStatus, ApiResponse, ResponseCode, FileUploadResult, TranscribeRequest, \
    TaskInfo, TranscribeResult, VoiceprintRegisterRequest, VoicePrintInfo, SpeakerStatistics
from voice2text.tran.schema.prints import SampleInfo
from voice2text.tran.server import VoiceSDKServer
from voice2text.tran.speech2text import STTAsyncVoice2TextService, STTConfigFactory, TranscriptionStrategy
from voice2text.tran.vector_base import ChromaDBConfig


async def start_server():
    import uvicorn
    import logging
    from voice2text.tran.config import (
        asr_config, whisper_config, speaker_config,
        vector_db_config, s3_config, service_config, server_config
    )

    # 配置日志
    logger = logging.getLogger(__name__)
    logger.info("正在从配置加载服务设置...")

    # 1. 创建语音服务实例
    asr_config_obj = STTConfigFactory.create_funasr_config(
        model_name=asr_config.model_name,
        device=asr_config.device
    )
    logger.info(f"ASR配置: 模型={asr_config.model_name}, 设备={asr_config.device}")

    whisper_config_obj = STTConfigFactory.create_whisper_config(
        model_name=whisper_config.model_name,
        device=whisper_config.device,
        use_auth_token=whisper_config.use_auth_token
    )
    logger.info(f"Whisper配置: 模型={whisper_config.model_name}, 设备={whisper_config.device}")

    speaker_config_obj = STTConfigFactory.create_speaker_config(
        threshold=speaker_config.threshold,
        device=speaker_config.device
    )
    logger.info(f"说话人配置: 阈值={speaker_config.threshold}, 设备={speaker_config.device}")

    vector_db_config_obj = ChromaDBConfig(
        db_type=vector_db_config.db_type,
        persist_directory=vector_db_config.persist_directory,
        collection_name=vector_db_config.collection_name
    )
    logger.info(f"向量数据库配置: 类型={vector_db_config.db_type}, 目录={vector_db_config.persist_directory}")

    s3_config_obj = S3StorageConfig(
        storage_type=s3_config.storage_type,
        bucket_name=s3_config.bucket_name,
        endpoint_url=s3_config.endpoint_url,
        access_key_id=s3_config.access_key_id,
        secret_access_key=s3_config.secret_access_key,
        prefix=s3_config.prefix
    )
    logger.info(f"S3存储配置: 端点={s3_config.endpoint_url}, 桶={s3_config.bucket_name}")

    # 创建向量服务配置
    service_config_obj = STTConfigFactory.create_stt_service_config(
        asr_config=asr_config_obj,
        whisper_config=whisper_config_obj,
        speaker_config=speaker_config_obj,
        vector_db_config=vector_db_config_obj,
        storage_config=s3_config_obj,
        transcription_strategy=service_config.transcription_strategy,
        language_model_mapping=service_config.language_model_mapping,
        max_transcribe_concurrent=service_config.max_transcribe_concurrent,
        max_speaker_concurrent=service_config.max_speaker_concurrent,
        task_timeout=service_config.task_timeout
    )
    logger.info(
        f"服务配置: 转写策略={service_config.transcription_strategy}, 最大转写并发={service_config.max_transcribe_concurrent}")

    voice_service = STTAsyncVoice2TextService(service_config_obj)
    await voice_service.start()
    # 2. 创建并启动服务器
    server = VoiceSDKServer(voice_service)

    # 3. 使用uvicorn运行FastAPI应用
    logger.info(f"启动服务器: 主机={server_config.host}, 端口={server_config.port}")
    config = uvicorn.Config(
        app=server.app,
        host=server_config.host,
        port=server_config.port,
        log_level=server_config.log_level
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(start_server())