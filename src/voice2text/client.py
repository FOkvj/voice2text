import logging
import os
from typing import Optional, List, Dict

import requests

from core_pak.api.schema import VoicePrintInfo, TranscriptionResult

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 数据模型(与服务端保持一致)





class AudioTranscriptionClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """
        初始化客户端

        :param base_url: 服务端基础URL
        :param api_key: 可选API密钥
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

        # 配置默认超时
        self.timeout = 30  # 秒

    def transcribe_audio(
            self,
            audio_path: str,
            threshold: float = 0.4,
            auto_register: bool = True,
            hotword: Optional[str] = None
    ) -> TranscriptionResult:
        """
        上传音频文件进行转写

        :param audio_path: 音频文件路径
        :param threshold: 声纹识别阈值(0-1)
        :param auto_register: 是否自动注册未知说话人
        :param hotword: 热词(提升识别特定词汇的准确性)
        :return: 转写结果对象
        :raises: TranscriptionError 如果转写失败
        """
        url = f"{self.base_url}/api/v1/transcribe"

        try:
            with open(audio_path, "rb") as f:
                files = {"file": (os.path.basename(audio_path), f)}
                params = {
                    "threshold": threshold,
                    "auto_register": str(auto_register).lower()
                }
                if hotword:
                    params["hotword"] = hotword

                response = self.session.post(
                    url,
                    files=files,
                    params=params,
                    timeout=self.timeout
                )

            # 检查响应状态
            if response.status_code != 200:
                error_data = response.json()
                raise TranscriptionError(
                    code=error_data.get("error_code", "UNKNOWN_ERROR"),
                    message=error_data.get("message", "Unknown error"),
                    details=error_data.get("details")
                )

            # 解析并验证响应数据
            return TranscriptionResult(**response.json())

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise TranscriptionError(
                code="NETWORK_ERROR",
                message="网络请求失败",
                details={"exception": str(e)}
            )

    def list_voice_prints(self, include_unnamed: bool = True) -> List[VoicePrintInfo]:
        """获取已注册的声纹列表"""
        url = f"{self.base_url}/api/v1/voiceprints"
        try:
            response = self.session.get(
                url,
                params={"include_unnamed": include_unnamed},
                timeout=self.timeout
            )

            if response.status_code != 200:
                error_data = response.json()
                raise VoicePrintError(
                    code=error_data.get("error_code", "UNKNOWN_ERROR"),
                    message=error_data.get("message", "Unknown error")
                )

            return [VoicePrintInfo(**item) for item in response.json()]

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list voice prints: {str(e)}")
            raise VoicePrintError(
                code="NETWORK_ERROR",
                message="获取声纹列表失败"
            )


class TranscriptionError(Exception):
    """转写错误异常类"""

    def __init__(self, code: str, message: str, details: Optional[Dict] = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"{code}: {message}")


class VoicePrintError(Exception):
    """声纹操作错误异常类"""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


# 使用示例
if __name__ == "__main__":
    # 初始化客户端
    client = AudioTranscriptionClient("http://localhost:8765")

    try:
        # 转写音频文件
        # result = client.transcribe_audio("../data/刘星家_20231212_122300_家有儿女吃饭.mp3")
        print("转写成功！结果摘要:")
        # print(f"音频时长: {result.audio_duration:.2f}秒")

        # 获取声纹列表
        voice_prints = client.list_voice_prints()
        print("\n已注册声纹:")
        for vp in voice_prints:
            print(f"- ID: {vp.id}, 名称: {vp.name or '未命名'}, 样本数: {vp.audio_samples}")

    except TranscriptionError as e:
        print(f"转写失败: {e.message} (代码: {e.code})")
    except VoicePrintError as e:
        print(f"获取声纹失败: {e.message}")
    except Exception as e:
        print(f"发生未知错误: {str(e)}")