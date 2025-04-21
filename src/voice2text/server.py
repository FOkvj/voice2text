import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from voice2text.api.schema import TranscriptionResult, ErrorResponse
from voice2text.tran.funasr_transcriber import FunASRTranscriber

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Audio Transcription Service API",
    description="提供音频转写和说话人识别服务",
    version="1.0.0",
    openapi_tags=[{
        "name": "transcription",
        "description": "音频转写相关操作"
    }, {
        "name": "voiceprints",
        "description": "声纹管理相关操作"
    }]
)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 数据模型定义






# 初始化转写器 (实际项目中应该使用依赖注入)
# 初始化转写器


# 临时文件存储目录
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def parse_filename(filename):
    """解析文件名，提取地点和时间信息"""
    # 移除扩展名
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]

    # 使用正则表达式匹配"地点_时间_名称"格式
    match = re.match(r'^(.+?)_(\d{8}_\d{6})_(.+)$', name_without_ext)
    if not match:
        return None, None, None

    location = match.group(1)
    time_str = match.group(2)
    name = match.group(3)

    # 解析时间
    try:
        time_obj = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
        date_str = time_obj.strftime("%Y/%m/%d")
        time_only_str = time_obj.strftime("%H:%M:%S")
        return location, date_str, time_only_str
    except ValueError:
        return location, None, None


@app.post("/api/v1/transcribe",
          response_model=TranscriptionResult,
          tags=["transcription"],
          responses={
              400: {"model": ErrorResponse},
              500: {"model": ErrorResponse}
          })
async def transcribe_audio(
        file: UploadFile = File(..., description="音频文件(WAV/MP3等格式)"),
        threshold: float = 0.4,
        auto_register: bool = True,
        hotword: Optional[str] = None
):
    """音频转写接口，支持说话人识别"""
    request_id = str(uuid.uuid4())
    try:
        # 验证文件类型
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.wav', '.mp3', '.m4a', '.flac']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_code": "INVALID_FILE_TYPE",
                    "message": f"不支持的文件类型: {file_ext}"
                }
            )

        # 保存上传的文件
        temp_filename = f"{request_id}{file_ext}"
        temp_path = os.path.join(UPLOAD_DIR, temp_filename)

        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())


        file_location, file_date, file_time = parse_filename(file.filename)
        # 执行转写
        transcript, auto_registered, audio_duration = transcriber.transcribe_file(
            temp_path,
            threshold=threshold,
            auto_register_unknown=auto_register,
            hotword=hotword,
            file_location=file_location,
            file_date=file_date,
            file_time=file_time
        )

        # 构建响应
        return {
            "request_id": request_id,
            "status": "completed",
            "result": transcript,
            "audio_duration": audio_duration,
            "created_at": datetime.now(),
            "metadata": {
                "auto_registered_speakers": auto_registered,
                "original_filename": file.filename
            }
        }

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "TRANSCRIPTION_ERROR",
                "message": "音频转写过程中发生错误",
                "details": {"error": str(e)}
            }
        )
    finally:
        # 清理临时文件
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/api/v1/voiceprints",
         tags=["voiceprints"])
def list_voice_prints(include_unnamed: bool = True):
    """获取已注册的声纹列表"""
    try:
        # 这里应该从您的VoicePrintManager获取实际数据
        return transcriber.list_registered_voices(include_unnamed=include_unnamed)
    except Exception as e:
        logger.error(f"Failed to list voice prints: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "VOICEPRINT_LIST_ERROR",
                "message": "获取声纹列表失败"
            }
        )


# 其他API端点类似实现...

if __name__ == "__main__":
    import uvicorn

    transcriber = FunASRTranscriber(
        device="cuda" if torch.cuda.is_available() else "cpu",
        voice_prints_path=os.path.join(os.path.expanduser("~"), ".cache", "voice_prints_ecapa.pkl"),
        funasr_model="paraformer-zh",
        funasr_model_revision="v2.0.4",
        vad_model="fsmn-vad",
        vad_model_revision="v2.0.4",
        punc_model="ct-punc",
        punc_model_revision="v2.0.4",
        spk_model="cam++",
        spk_model_revision="v2.0.2"
    )

    uvicorn.run(app, host="0.0.0.0", port=8765)