import whisper
from whisperx import load_align_model, align


# 加载Whisper模型并准备对齐模型
def load_model_with_timestamps(model_name="base", device="cpu"):
    print(f"Loading Whisper model: {model_name}...")
    # 加载Whisper模型
    whisper_model = whisper.load_model(model_name).to(device)
    # 加载WhisperX对齐模型（默认支持多语言，包括中文）
    align_model, metadata = load_align_model("zh", device=device)  # 明确指定语言为中文
    return whisper_model, align_model, metadata


# 语音转文字并添加时间戳
def transcribe_audio_with_timestamps(whisper_model, align_model, metadata, audio_file_path):
    print(f"Transcribing audio file: {audio_file_path}...")

    # 使用Whisper模型进行初步转录
    result = whisper_model.transcribe(audio_file_path, language="zh")  # 强制指定语言为中文
    segments = result["segments"]  # 获取初步转录结果

    # 使用WhisperX对齐时间戳
    aligned_segments = align(segments, align_model, metadata, audio_file_path, "cpu")

    # 格式化输出带时间戳的文本
    formatted_output = []
    for segment in aligned_segments['segments']:
        start_time = format_time(segment["start"])
        end_time = format_time(segment["end"])
        text = segment["text"]
        formatted_output.append(f"[{start_time}-{end_time}] {text}")

    return "\n".join(formatted_output)


# 将秒数格式化为 HH:MM:SS 格式
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


# 主函数
if __name__ == "__main__":
    # 配置参数
    model_name = "base"  # 模型大小，根据需求选择
    audio_file = "core/26.mp3"  # 输入音频文件路径
    device = "cpu"  # 如果有GPU支持，可以改为 "cuda"

    # 加载模型
    whisper_model, align_model, metadata = load_model_with_timestamps(model_name, device)

    # 转录音频文件并生成带时间戳的文本
    transcription = transcribe_audio_with_timestamps(whisper_model, align_model, metadata, audio_file)

    # 输出结果
    print("Transcription Result with Timestamps:")
    print(transcription)