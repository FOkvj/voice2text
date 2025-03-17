import requests
import argparse
import os
import time


class TranscriptionClient:
    def __init__(self, server_url="http://localhost:5000"):
        """初始化转写客户端"""
        self.server_url = server_url

    def check_server_health(self):
        """检查服务器健康状态"""
        try:
            response = requests.get(f"{self.server_url}/health")
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "unhealthy", "error": f"Status code: {response.status_code}"}
        except requests.exceptions.RequestException as e:
            return {"status": "unhealthy", "error": str(e)}

    def transcribe_audio(self, audio_file_path, language="zh"):
        """发送音频文件进行转写"""
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        print(f"Sending file {audio_file_path} to server for transcription...")
        start_time = time.time()

        with open(audio_file_path, 'rb') as f:
            files = {'audio': (os.path.basename(audio_file_path), f)}
            data = {'language': language}

            try:
                response = requests.post(
                    f"{self.server_url}/transcribe",
                    files=files,
                    data=data
                )

                elapsed_time = time.time() - start_time
                print(f"Request completed in {elapsed_time:.2f} seconds")

                if response.status_code == 200:
                    return response.json()
                else:
                    error_msg = f"Error: {response.status_code}"
                    try:
                        error_msg += f" - {response.json().get('error', '')}"
                    except:
                        pass
                    raise Exception(error_msg)

            except requests.exceptions.RequestException as e:
                raise ConnectionError(f"Failed to connect to server: {str(e)}")

    def save_transcription(self, transcription, output_file):
        """保存转写结果到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(transcription['text'])
        print(f"Transcription saved to {output_file}")

        # 如果需要，还可以保存JSON格式的详细信息
        json_output = output_file.replace('.txt', '.json')
        if json_output != output_file:
            import json
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(transcription['segments'], f, ensure_ascii=False, indent=2)
            print(f"Detailed segments saved to {json_output}")


def main():
    parser = argparse.ArgumentParser(description="Client for audio transcription service")
    parser.add_argument("--server", default="http://localhost:5000", help="Server URL")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--language", default="zh", help="Language code (default: zh)")
    parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    # 设置默认输出文件名
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.audio))[0]
        args.output = f"{base_name}_transcription.txt"

    client = TranscriptionClient(args.server)

    # 检查服务器健康状态
    health = client.check_server_health()
    if health["status"] != "healthy":
        print(f"Warning: Server may not be healthy: {health}")
        proceed = input("Do you want to proceed anyway? (y/n): ")
        if proceed.lower() != 'y':
            return

    try:
        # 发送转写请求
        result = client.transcribe_audio(args.audio, args.language)

        # 保存结果
        client.save_transcription(result, args.output)

        # 打印部分结果
        print("\nTranscription Preview:")
        preview_lines = result["text"].split('\n')[:5]
        for line in preview_lines:
            print(line)
        if len(preview_lines) < result["text"].count('\n') + 1:
            print("...")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()