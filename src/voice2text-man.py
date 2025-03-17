import whisper
from whisperx import load_align_model, align
import torch
import numpy as np
import os
import librosa
import scipy
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
import soundfile as sf
import torchaudio
from speechbrain.pretrained import EncoderClassifier


# 加载Whisper模型并准备对齐模型
def load_model_with_timestamps(model_name="base", device="cpu"):
    print(f"Loading Whisper model: {model_name}...")
    # 加载Whisper模型
    whisper_model = whisper.load_model(model_name).to(device)
    # 加载WhisperX对齐模型（默认支持多语言，包括中文）
    align_model, metadata = load_align_model("zh", device=device)  # 明确指定语言为中文
    return whisper_model, align_model, metadata


# 加载SpeechBrain说话人嵌入模型
def load_speaker_encoder(device="cpu"):
    print("Loading SpeechBrain speaker embeddings model...")
    # 这个模型会在第一次运行时自动下载
    speaker_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    return speaker_encoder


# 对音频进行分段并提取说话人嵌入
def extract_speaker_embeddings(audio_file, speaker_encoder, min_segment_length=3.0):
    print(f"Extracting speaker embeddings from {audio_file}...")

    # 加载音频
    waveform, sample_rate = torchaudio.load(audio_file)

    # 如果是立体声，转换为单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 确保采样率为16kHz（SpeechBrain模型需要）
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # 加载和预处理音频
    audio, _ = librosa.load(audio_file, sr=sample_rate)
    audio_length = len(audio) / sample_rate

    # 使用语音活动检测(VAD)分割音频
    # 这里使用简单的能量阈值方法，也可以使用更复杂的VAD
    energy = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512)[0]
    threshold = np.mean(energy) * 0.5
    speech_frames = energy > threshold

    # 将连续的语音帧组合成片段
    speech_segments = []
    start = None
    for i, is_speech in enumerate(speech_frames):
        frame_time = i * 512 / sample_rate

        if is_speech and start is None:
            start = frame_time
        elif not is_speech and start is not None:
            end = frame_time
            if end - start >= min_segment_length:  # 只考虑足够长的片段
                speech_segments.append((start, end))
            start = None

    # 处理最后一个片段
    if start is not None:
        end = audio_length
        if end - start >= min_segment_length:
            speech_segments.append((start, end))

    # 提取每个片段的说话人嵌入
    embeddings = []
    segments_info = []

    for i, (start, end) in enumerate(speech_segments):
        # 计算起始和结束采样点
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)

        # 提取片段音频
        segment_audio = audio[start_sample:end_sample]

        # 将音频转换为tensor并调整形状
        segment_tensor = torch.FloatTensor(segment_audio).unsqueeze(0)

        # 提取说话人嵌入
        with torch.no_grad():
            embedding = speaker_encoder.encode_batch(segment_tensor)
            embedding = embedding.squeeze().cpu().numpy()

        embeddings.append(embedding)
        segments_info.append({
            "id": i,
            "start": start,
            "end": end
        })

    return np.array(embeddings), segments_info


# 聚类说话人嵌入以确定说话人数量
def cluster_speakers(embeddings, segments_info, max_speakers=5):
    print("Clustering speakers...")

    # 如果片段数量太少，无法聚类
    if len(embeddings) <= 1:
        for segment in segments_info:
            segment["speaker"] = "说话人_1"
        return segments_info

    # 计算嵌入向量之间的距离矩阵
    distance_matrix = cdist(embeddings, embeddings, metric='cosine')

    # 使用层次聚类确定说话人
    # 动态确定说话人数量，最多max_speakers个
    n_clusters_range = range(1, min(max_speakers + 1, len(embeddings) + 1))

    best_n_clusters = 2  # 默认值
    max_silhouette = -1

    # 简化：直接使用固定的聚类数，或者根据embeddings数量动态调整
    if len(embeddings) <= 3:
        best_n_clusters = min(len(embeddings), 2)
    else:
        best_n_clusters = min(max_speakers, len(embeddings) // 2)

    # 执行聚类
    clustering = AgglomerativeClustering(
        n_clusters=best_n_clusters,
        affinity='precomputed',
        linkage='average'
    ).fit(distance_matrix)

    labels = clustering.labels_

    # 将聚类结果添加到segments_info
    for i, segment in enumerate(segments_info):
        segment["speaker"] = f"说话人_{labels[i] + 1}"

    return segments_info


# 将转录与说话人进行匹配
def match_transcription_with_speakers(aligned_segments, speaker_segments):
    print("Matching transcription with speakers...")
    result = []

    for segment in aligned_segments['segments']:
        segment_start = segment["start"]
        segment_end = segment["end"]
        segment_text = segment["text"]

        # 找出与当前文本段重叠最多的说话人
        max_overlap = 0
        best_speaker = None

        for spk_segment in speaker_segments:
            spk_start = spk_segment["start"]
            spk_end = spk_segment["end"]

            # 计算重叠
            overlap_start = max(segment_start, spk_start)
            overlap_end = min(segment_end, spk_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = spk_segment["speaker"]

        # 保存结果
        result.append({
            "start": segment_start,
            "end": segment_end,
            "text": segment_text,
            "speaker": best_speaker if max_overlap > 0 else "未知"
        })

    return result


# 语音转文字并添加时间戳和说话人
def transcribe_audio_with_speakers(whisper_model, align_model, metadata, speaker_encoder, audio_file_path):
    print(f"Transcribing audio file: {audio_file_path}...")

    # 使用Whisper模型进行初步转录
    result = whisper_model.transcribe(audio_file_path, language="zh")  # 强制指定语言为中文
    segments = result["segments"]  # 获取初步转录结果

    # 使用WhisperX对齐时间戳
    aligned_segments = align(segments, align_model, metadata, audio_file_path, "cpu")

    # 提取说话人嵌入
    embeddings, segments_info = extract_speaker_embeddings(audio_file_path, speaker_encoder)

    # 聚类确定说话人
    speaker_segments = cluster_speakers(embeddings, segments_info)

    # 匹配转录与说话人
    segments_with_speakers = match_transcription_with_speakers(aligned_segments, speaker_segments)

    # 格式化输出带时间戳和说话人的文本
    formatted_output = []
    for segment in segments_with_speakers:
        start_time = format_time(segment["start"])
        end_time = format_time(segment["end"])
        text = segment["text"]
        speaker = segment["speaker"]
        formatted_output.append(f"[{start_time}-{end_time}] 【{speaker}】: {text}")

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
    speaker_encoder = load_speaker_encoder(device)

    # 转录音频文件并生成带时间戳和说话人的文本
    transcription = transcribe_audio_with_speakers(
        whisper_model, align_model, metadata,
        speaker_encoder, audio_file
    )

    # 输出结果
    print("Transcription Result with Timestamps and Speakers:")
    print(transcription)

    # 保存结果到文件
    with open("transcription_with_speakers.txt", "w", encoding="utf-8") as f:
        f.write(transcription)
    print(f"Transcription saved to transcription_with_speakers.txt")