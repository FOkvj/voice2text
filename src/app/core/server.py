import os
import whisper
from whisperx import load_align_model, align
from flask import Flask, request, jsonify
import tempfile
import logging
from werkzeug.utils import secure_filename

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 配置上传文件的最大大小 (500MB)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# 全局变量存储已加载的模型
whisper_model = None
align_model = None
align_metadata = None
model_name = "base"
device = "cpu"


class TranscriptionService:
    @staticmethod
    def load_model_with_timestamps(model_name="base", device="cpu"):
        """加载Whisper模型并准备对齐模型"""
        logger.info(f"Loading Whisper model: {model_name}...")
        # 加载Whisper模型
        whisper_model = whisper.load_model(model_name).to(device)
        # 加载WhisperX对齐模型（默认支持多语言，包括中文）
        align_model, metadata = load_align_model("zh", device=device)  # 明确指定语言为中文
        return whisper_model, align_model, metadata

    @staticmethod
    def transcribe_audio_with_timestamps(whisper_model, align_model, metadata, audio_file_path, language="zh"):
        """语音转文字并添加时间戳"""
        logger.info(f"Transcribing audio file: {audio_file_path}...")

        # 使用Whisper模型进行初步转录
        result = whisper_model.transcribe(audio_file_path, language=language)  # 强制指定语言
        segments = result["segments"]  # 获取初步转录结果

        # 使用WhisperX对齐时间戳
        aligned_segments = align(segments, align_model, metadata, audio_file_path, "cpu")

        # 准备输出结果
        formatted_output = []
        json_output = []

        for segment in aligned_segments['segments']:
            start_time = TranscriptionService.format_time(segment["start"])
            end_time = TranscriptionService.format_time(segment["end"])
            text = segment["text"]
            formatted_output.append(f"[{start_time}-{end_time}] {text}")

            json_output.append({
                "start": segment["start"],
                "end": segment["end"],
                "start_formatted": start_time,
                "end_formatted": end_time,
                "text": text
            })

        return {
            "text": "\n".join(formatted_output),
            "segments": json_output
        }

    @staticmethod
    def format_time(seconds):
        """将秒数格式化为 HH:MM:SS 格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"


@app.before_first_request
def initialize_models():
    """在第一个请求之前初始化模型"""
    global whisper_model, align_model, align_metadata, model_name, device
    whisper_model, align_model, align_metadata = TranscriptionService.load_model_with_timestamps(model_name, device)
    logger.info("Models initialized successfully")


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """处理音频转写请求"""
    global whisper_model, align_model, align_metadata

    # 检查是否有文件
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']

    # 检查文件名是否为空
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 获取语言参数，默认为中文
    language = request.form.get('language', 'zh')

    # 保存上传的文件
    filename = secure_filename(audio_file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio_file.save(file_path)

    try:
        # 转写音频
        result = TranscriptionService.transcribe_audio_with_timestamps(
            whisper_model, align_model, align_metadata, file_path, language
        )

        # 删除临时文件
        os.remove(file_path)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        # 确保临时文件被删除
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({"status": "healthy", "model": model_name})


if __name__ == "__main__":
    # 设置服务器端口和主机
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")

    # 启动服务器
    app.run(host=host, port=port, debug=False)