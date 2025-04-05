import whisper
import os
from whisperx import load_align_model, align
from whisperx.diarize import DiarizationPipeline  # 引入说话人分离组件

# 设置模型缓存目录
MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "whisper_models")
ALIGN_MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "whisperx_models")

# 确保缓存目录存在
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(ALIGN_MODEL_CACHE_DIR, exist_ok=True)


# 加载Whisper模型、对齐模型和说话人分离模型
def load_models(model_name="base", device="cpu"):
    print(f"Loading Whisper model: {model_name} from cache if available...")

    # 加载Whisper模型
    whisper_model = whisper.load_model(
        model_name,
        download_root=MODEL_CACHE_DIR
    ).to(device)

    print("Loading alignment model from cache if available...")
    # 加载WhisperX对齐模型
    os.environ["HF_HOME"] = ALIGN_MODEL_CACHE_DIR
    align_model, metadata = load_align_model("zh", device=device)

    print("Loading diarization model...")
    # 加载说话人分离模型
    diarize_model = DiarizationPipeline(use_auth_token=None, device=device)

    return whisper_model, align_model, metadata, diarize_model


# 语音转文字并添加时间戳和说话人标识
def transcribe_with_speaker_diarization(whisper_model, align_model, metadata, diarize_model, audio_file_path):
    print(f"Transcribing audio file: {audio_file_path}...")

    # 使用Whisper模型进行初步转录
    result = whisper_model.transcribe(audio_file_path, language="zh")
    segments = result["segments"]

    # 使用WhisperX对齐时间戳
    aligned_segments = align(segments, align_model, metadata, audio_file_path, device="cpu")

    # 进行说话人分离
    diarize_segments = diarize_model(audio_file_path)

    # 将说话人信息与转录片段合并
    # 这里我们需要基于时间戳将说话人ID匹配到转录片段
    speaker_segments = []

    for segment in aligned_segments['segments']:
        # 查找当前片段的说话人
        segment_start = segment["start"]
        segment_end = segment["end"]
        speaker_id = find_speaker(segment_start, segment_end, diarize_segments)

        # 添加说话人信息到片段
        start_time = format_time(segment["start"])
        end_time = format_time(segment["end"])
        text = segment["text"]
        speaker_segments.append(f"[{start_time}-{end_time}] [说话人{speaker_id}] {text}")

    return "\n".join(speaker_segments)


# 根据时间戳查找说话人
def find_speaker(start_time, end_time, diarize_segments):
    # 简单的方法是找重叠最多的说话人
    # 实际应用中可能需要更复杂的逻辑
    best_speaker = None
    max_overlap = 0

    for speaker, turns in diarize_segments.items():
        for turn in turns:
            turn_start, turn_end = turn

            # 计算重叠
            overlap_start = max(start_time, turn_start)
            overlap_end = min(end_time, turn_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = speaker

    return best_speaker if best_speaker else "未知"


# 将秒数格式化为 HH:MM:SS 格式
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


# 主函数
if __name__ == "__main__":
    # 配置参数
    model_name = "base"
    audio_file = "core/26.mp3"
    device = "cpu"  # 如果有GPU支持，可以改为 "cuda"

    # 加载模型
    whisper_model, align_model, metadata, diarize_model = load_models(model_name, device)

    # 转录音频文件并生成带时间戳和说话人标识的文本
    transcription = transcribe_with_speaker_diarization(whisper_model, align_model, metadata, diarize_model, audio_file)

    # 输出结果
    print("Transcription Result with Timestamps and Speaker Diarization:")
    print(transcription)

    # 保存结果到文件
    output_file = audio_file.rsplit(".", 1)[0] + "_transcript_with_speakers.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcription)
    print(f"Transcription saved to {output_file}")