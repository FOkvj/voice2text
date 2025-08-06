import asyncio
import io
import json
import re
import urllib
from pathlib import Path

import aiohttp
from typing import Optional, Dict, List, Callable
from datetime import datetime
import aiofiles
from pydantic import TypeAdapter
# ============================================================================
# SDK客户端实现
# ============================================================================

import httpx

from voice2text.tran.schema.dto import ApiResponse, ServiceStatus, FileUploadResult, ResponseCode, TranscribeRequest, \
    TaskInfo, VoiceprintRegisterRequest, SpeakerStatistics, TranscribeResult, VoicePrintInfo
from voice2text.tran.schema.prints import SampleInfo


class VoiceSDKClient:
    """Voice2Text SDK 客户端"""

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 300.0):
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

    async def _make_request(self, method: str, endpoint: str, response_model=Dict, **kwargs) -> ApiResponse:
        """发送HTTP请求并返回标准响应"""
        try:
            response = await self.client.request(method, endpoint, **kwargs)
            response.raise_for_status()

            response_data = response.json()

            adapter = TypeAdapter(response_model)

            # 转换为标准ApiResponse
            return ApiResponse(
                success=response_data.get('success', False),
                code=response_data.get('code', response.status_code),
                message=response_data.get('message', ''),
                data=adapter.validate_python(response_data.get('data', {})),
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
        return await self._make_request('GET', '/health', ServiceStatus)

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

    async def submit_transcribe_and_get_result(
            self,
            file_id: str,
            wait_for_completion: bool = True,
            delete_after_processing: bool = True,
            poll_interval: float = 1.0,
            timeout: float = 300.0,
            **transcribe_kwargs
    ) -> ApiResponse[Dict]:
        """
        提交转写任务并获取结果

        Args:
            file_id: 已上传文件的ID
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
            # 1. 提交转写任务
            transcribe_task = await self.transcribe_audio(
                file_id,
                delete_after_processing=delete_after_processing,
                **transcribe_kwargs
            )
            if not transcribe_task.success:
                return transcribe_task

            task_id = transcribe_task.data['task_id']

            # 2. 如果不等待完成，直接返回任务信息
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

            # 3. 等待任务完成并获取结果
            result = await self.wait_for_task_completion(task_id, poll_interval, timeout)

            if result.success:
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
                return result

        except Exception as e:
            return ApiResponse.error_response(f"转写任务处理失败: {str(e)}")

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
                # 2. 使用新函数提交转写任务并获取结果
                result = await self.submit_transcribe_and_get_result(
                    file_id=file_id,
                    wait_for_completion=wait_for_completion,
                    delete_after_processing=delete_after_processing,
                    poll_interval=poll_interval,
                    timeout=timeout,
                    **transcribe_kwargs
                )

                # 如果任务失败且不会自动删除文件，手动清理
                if not result.success and not delete_after_processing:
                    try:
                        await self.delete_file(file_id)
                    except:
                        pass

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
    ) -> ApiResponse[SampleInfo]:
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
                    return register_result
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

    async def delete_speaker_audio_sample(self, speaker_id: str, file_id: str) -> bool:
        """
        删除特定说话人的音频样本

        Args:
            speaker_id: 说话人ID
            file_id: 音频文件ID
        """
        response = await self._make_request('DELETE', f'/api/v1/speakers/{speaker_id}/samples/{file_id}')
        if response.success:
            return True
        else:
            return False

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
        return await self._make_request('POST', '/api/v1/audio/transcribe', TaskInfo, json=request_data)

    async def register_voiceprint(self, person_name: str, audio_file_id: str) -> \
    ApiResponse[SampleInfo]:
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
        return await self._make_request('POST', '/api/v1/voiceprints/register', SampleInfo, json=request_data)

    async def get_task_status(self, task_id: str) -> ApiResponse[TaskInfo]:
        """获取任务状态"""
        return await self._make_request('GET', f'/api/v1/tasks/{task_id}', TaskInfo)

    async def get_task_result(self, task_id: str) -> ApiResponse[TranscribeResult]:
        """获取任务结果"""
        return await self._make_request('GET', f'/api/v1/tasks/{task_id}/result', TranscribeResult)

    async def list_voiceprints(self, include_unnamed: bool = True) -> ApiResponse[List[VoicePrintInfo]]:
        """获取声纹列表"""
        params = {'include_unnamed': include_unnamed}
        return await self._make_request('GET', '/api/v1/voiceprints/list', List[VoicePrintInfo], params=params)


    async def rename_speaker(self, speaker_id: str, new_name: str) -> ApiResponse[Dict]:
        """重命名说话人"""
        params = {'new_name': new_name}
        return await self._make_request('PUT', f'/api/v1/speakers/{speaker_id}/rename', Dict, params=params)

    async def delete_speaker(self, speaker_id: str) -> bool:
        """删除说话人"""
        response = await self._make_request('DELETE', f'/api/v1/speakers/{speaker_id}')
        if response.success:
            return True
        else:
            return False

    async def get_statistics(self) -> ApiResponse[SpeakerStatistics]:
        """获取统计信息"""
        return await self._make_request('GET', '/api/v1/statistics', SpeakerStatistics)

    async def list_files(self, category: Optional[str] = None) -> ApiResponse[List[Dict]]:
        """列出文件"""
        params = {'category': category} if category else {}
        return await self._make_request('GET', '/api/v1/files/list', List[Dict], params=params)

    import io
    import json
    from typing import Dict, Optional

    async def get_file_stream(self, file_id: str) -> io.BytesIO:
        """
        下载文件流（返回BytesIO对象）

        Args:
            file_id: 文件ID

        Returns:
            ApiResponse containing BytesIO object with file content
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/files/{file_id}/download",
                    headers={'Authorization': f'Bearer {self.api_key}'} if self.api_key else {},
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    # 创建 BytesIO 对象并写入内容
                    file_stream = io.BytesIO()

                    # 处理流式响应
                    async for chunk in response.aiter_bytes():
                        file_stream.write(chunk)

                    # 重置指针到开头
                    file_stream.seek(0)
                    return file_stream
                else:
                    return None

        except httpx.TimeoutException:
            return ApiResponse.error_response("下载文件超时")
        except httpx.NetworkError as e:
            return ApiResponse.error_response(f"网络错误: {str(e)}")
        except Exception as e:
            return ApiResponse.error_response(f"下载文件失败: {str(e)}")


    def _extract_filename_from_disposition(self, content_disposition: str) -> str:
        """从Content-Disposition头中提取文件名（RFC 5987编码）"""
        if not content_disposition:
            return "unknown_file"

        # 处理 RFC 5987 编码格式: filename*=UTF-8''encoded_name
        if "filename*=" in content_disposition:
            filename_part = content_disposition.split("filename*=")[1]
            if filename_part.startswith("UTF-8''"):
                encoded_name = filename_part[7:].split(';')[0].strip()
                try:
                    return urllib.parse.unquote(encoded_name)
                except Exception:
                    return "unknown_file"

        return "unknown_file"

    async def download_file(
            self,
            file_id: str,
            local_path: str = None,
            dir: str = None,
            progress_callback: Optional[Callable[[int, int], None]] = None,
            timeout: int = 300
    ) -> bool:
        """
        优化的客户端文件下载方法
        支持进度回调和大文件下载

        Args:
            base_url: 服务器基础URL
            file_id: 文件ID
            local_path: 本地保存路径
            progress_callback: 进度回调函数 (downloaded_bytes, total_bytes)
            timeout: 超时时间（秒）

        Returns:
            下载是否成功
        """
        if local_path:
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
        if dir:
            dir = Path(dir)
            dir.mkdir(parents=True, exist_ok=True)

        try:
            timeout_config = aiohttp.ClientTimeout(total=timeout)

            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                url = f"{self.base_url}/api/v1/files/{file_id}/download"

                async with session.get(url) as response:
                    if response.status != 200:
                        print(f"Download failed with status: {response.status}")
                        return False
                    headers = response.headers

                    # 获取文件总大小
                    total_size = int(headers.get('Content-Length', 0))
                    downloaded = 0

                    save_path = local_path if local_path else dir.joinpath(self.extract_filename_from_headers(headers))
                    # 异步写入文件
                    async with aiofiles.open(save_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(64 * 1024):  # 64KB chunks
                            await f.write(chunk)
                            downloaded += len(chunk)

                            # 调用进度回调
                            if progress_callback:
                                progress_callback(downloaded, total_size)

                    return True

        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def extract_filename_from_headers(self, headers) -> str:
        """
        从HTTP响应头中提取文件名
        专门处理服务端发送的 filename*=UTF-8''encoded_name 格式

        Args:
            headers: HTTP响应头（字典或对象）

        Returns:
            str: 提取的文件名，失败则返回默认名

        Examples:
            headers = {'Content-Disposition': 'attachment; filename*=UTF-8\'\'%E5%88%98%E6%98%9F_sample5.wav'}
        """
        # 获取 Content-Disposition 头
        content_disposition = headers.get('Content-Disposition', '') if hasattr(headers, 'get') else headers.get(
            'Content-Disposition', '')

        if not content_disposition:
            return "download_file"

        # 匹配 filename*=UTF-8''encoded_name 格式
        match = re.search(r"filename\*=UTF-8''([^;]+)", content_disposition)
        if match:
            encoded_filename = match.group(1)
            try:
                # URL解码得到原始文件名
                return urllib.parse.unquote(encoded_filename)
            except:
                pass

        # 如果匹配失败，返回默认名
        return "download_file"
    async def delete_file(self, file_id: str) -> ApiResponse[Dict]:
        """删除文件"""
        return await self._make_request('DELETE', f'/api/v1/files/{file_id}')

    async def get_file_info(self, file_id: str) -> ApiResponse[Dict]:
        """获取文件信息"""
        return await self._make_request('GET', f'/api/v1/files/{file_id}/info')

    async def wait_for_task_completion(self, task_id: str, poll_interval: float = 1.0, timeout: float = 300000000.0) -> \
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



