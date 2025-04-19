from datetime import datetime
from typing import Optional, Dict

from pydantic import BaseModel, Field

class VoicePrintInfo(BaseModel):
    id: str
    name: Optional[str] = None
    created_at: datetime
    audio_samples: int

class TranscriptionResult(BaseModel):
    request_id: str = Field(..., description="请求唯一ID")
    status: str = Field(..., description="处理状态")
    result: str = Field(..., description="转写分段结果")
    audio_duration: float = Field(..., description="音频时长(秒)")
    language: Optional[str] = Field(None, description="检测到的语言")
    created_at: datetime = Field(..., description="处理完成时间")
    metadata: Optional[Dict] = Field(None, description="附加元数据")


class VoicePrintInfo(BaseModel):
    id: str = Field(..., description="声纹ID")
    name: Optional[str] = Field(None, description="声纹名称(如果有)")
    created_at: datetime = Field(..., description="注册时间")
    audio_samples: int = Field(..., description="用于注册的音频样本数")


class ErrorResponse(BaseModel):
    error_code: str = Field(..., description="错误代码")
    message: str = Field(..., description="错误信息")
    details: Optional[Dict] = Field(None, description="错误详情")