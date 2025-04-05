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
from speechbrain.inference import EncoderClassifier
import glob


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


# 从注册音频中提取说话人声纹特征
def register_speaker(speaker_encoder, audio_file, speaker_name):
    print(f"Registering speaker: {speaker_name} from {audio_file}...")

    # 加载音频
    waveform, sample_rate = torchaudio.load(audio_file)

    # 如果是立体声，转换为单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 确保采样率为16kHz（SpeechBrain模型需要）
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # 提取说话人嵌入
    with torch.no_grad():
        embedding = speaker_encoder.encode_batch(waveform)
        embedding = embedding.squeeze().cpu().numpy()

    return {
        "name": speaker_name,
        "embedding": embedding
    }


# 注册一个目录中的所有说话人
def register_speakers_from_directory(speaker_encoder, directory_path):
    print(f"Registering speakers from directory: {directory_path}...")
    registered_speakers = []

    # 遍历目录中的所有音频文件
    audio_files = glob.glob(os.path.join(directory_path, "*.mp3")) + \
                  glob.glob(os.path.join(directory_path, "*.wav"))

    for audio_file in audio_files:
        # 从文件名获取说话人姓名（去掉扩展名）
        speaker_name = os.path.splitext(os.path.basename(audio_file))[0]
        # 注册说话人
        speaker_info = register_speaker(speaker_encoder, audio_file, speaker_name)
        registered_speakers.append(speaker_info)
        print(f"Registered: {speaker_name}")

    return registered_speakers


# 对音频进行分段并提取说话人嵌入
import torch
# from speechbrain.pretrained import VAD


def extract_speaker_embeddings(audio_file, speaker_encoder, min_segment_samples=1600):
    """
    使用SpeechBrain VAD提取说话人嵌入，并确保音频段长度足够

    参数:
        audio_file: 音频文件路径
        speaker_encoder: SpeechBrain说话人编码器
        min_segment_samples: 最小音频段长度（采样点数）
    """
    from speechbrain.pretrained import VAD
    import torch
    import torchaudio
    import numpy as np
    import librosa

    print(f"提取说话人嵌入，使用SpeechBrain VAD...")

    try:
        # 加载预训练VAD模型
        vad_model = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty")

        # 加载音频
        waveform, sample_rate = torchaudio.load(audio_file)

        # 如果是立体声，转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 重采样到16kHz (如果需要)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # 使用VAD检测语音段
        speech_prob = vad_model.get_speech_prob_chunk(waveform)
        boundaries = vad_model.get_boundaries(speech_prob)

        print(f"VAD检测到 {len(boundaries)} 个语音段")

        # 提取每个语音段的说话人嵌入
        embeddings = []
        segments_info = []

        for i, (start, end) in enumerate(boundaries):
            # 将PyTorch张量转换为Python数值
            start = int(round(float(start)))
            end = int(round(float(end)))

            # 确保段落长度足够
            segment_length = end - start
            if segment_length < min_segment_samples:
                # 如果太短，扩展段落（向两侧扩展）
                padding_needed = min_segment_samples - segment_length
                padding_start = padding_needed // 2
                padding_end = padding_needed - padding_start

                new_start = max(0, start - padding_start)
                new_end = min(waveform.shape[1], end + padding_end)

                print(f"段落 {i + 1} 太短 ({segment_length} 采样点)，扩展到 {new_end - new_start} 采样点")
                start = new_start
                end = new_end

            # 再次检查段落长度是否足够
            if end - start < min_segment_samples:
                # 如果仍然太短（可能在音频开始或结束处），跳过
                print(f"跳过段落 {i + 1}，长度仍然不足 ({end - start} < {min_segment_samples})")
                continue

            # 提取片段音频
            try:
                segment_audio = waveform[:, start:end]

                # 确保音频形状正确
                print(f"段落 {i + 1} 形状: {segment_audio.shape}")

                # 确保段落能被处理（某些模型可能需要特定的最小长度）
                if segment_audio.shape[1] < 400:  # 25ms @ 16kHz
                    print(f"跳过段落 {i + 1}，长度太短 ({segment_audio.shape[1]} < 400)")
                    continue

                # 提取说话人嵌入
                with torch.no_grad():
                    # 计算长度特征（SpeechBrain需要）
                    wav_lens = torch.tensor([1.0])  # 标准化长度

                    # 调用SpeechBrain的encode_batch
                    embedding = speaker_encoder.encode_batch(segment_audio, wav_lens=wav_lens)
                    embedding = embedding.squeeze().cpu().numpy()

                # 转换为时间（秒）
                start_time = start / sample_rate
                end_time = end / sample_rate

                embeddings.append(embedding)
                segments_info.append({
                    "id": len(embeddings) - 1,  # 使用当前索引作为ID
                    "start": start_time,
                    "end": end_time
                })

                print(f"成功处理段落 {i + 1}，时间 [{start_time:.2f}s - {end_time:.2f}s]")

            except Exception as e:
                print(f"处理段落 {i + 1} 时出错: {e}")

        print(f"总共提取了 {len(embeddings)} 个有效的说话人嵌入")

        # 如果没有成功提取任何嵌入，尝试处理整个音频
        if len(embeddings) == 0:
            print("未能提取任何说话人嵌入，尝试处理整个音频...")

            try:
                with torch.no_grad():
                    wav_lens = torch.tensor([1.0])
                    embedding = speaker_encoder.encode_batch(waveform, wav_lens=wav_lens)
                    embedding = embedding.squeeze().cpu().numpy()

                audio_length = waveform.shape[1] / sample_rate

                return np.array([embedding]), [{
                    "id": 0,
                    "start": 0,
                    "end": audio_length
                }]
            except Exception as e:
                print(f"处理整个音频时出错: {e}")
                return np.array([]), []

        return np.array(embeddings), segments_info

    except Exception as e:
        print(f"提取说话人嵌入时发生错误: {e}")
        return np.array([]), []


# 识别说话人（与已注册说话人匹配）
def identify_speakers(embeddings, segments_info, registered_speakers, similarity_threshold=0.3):
    print("Identifying speakers based on registered profiles...")

    for i, embedding in enumerate(embeddings):
        # 找到最匹配的已注册说话人
        best_score = -1
        best_speaker = None

        for speaker in registered_speakers:
            # 计算余弦相似度 (1 - 余弦距离)
            similarity = 1 - cdist(
                embedding.reshape(1, -1),
                speaker["embedding"].reshape(1, -1),
                metric='cosine'
            )[0][0]

            if similarity > best_score:
                best_score = similarity
                best_speaker = speaker["name"]

        # 只有当相似度超过阈值时才识别为已知说话人
        if best_score >= similarity_threshold:
            segments_info[i]["speaker"] = best_speaker
            segments_info[i]["confidence"] = best_score
        else:
            # 低于阈值时识别为未知说话人
            segments_info[i]["speaker"] = "未知说话人"
            segments_info[i]["confidence"] = best_score

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
        confidence = 0

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
                confidence = spk_segment.get("confidence", 0)

        # 保存结果
        result.append({
            "start": segment_start,
            "end": segment_end,
            "text": segment_text,
            "speaker": best_speaker if max_overlap > 0 else "未知",
            "confidence": confidence
        })

    return result


# 语音转文字并添加时间戳和说话人
def transcribe_audio_with_registered_speakers(
        whisper_model, align_model, metadata,
        speaker_encoder, audio_file_path,
        registered_speakers
):
    print(f"Transcribing audio file: {audio_file_path}...")

    # 使用Whisper模型进行初步转录
    result = whisper_model.transcribe(audio_file_path, language="zh")  # 强制指定语言为中文
    segments = result["segments"]  # 获取初步转录结果

    # 使用WhisperX对齐时间戳
    aligned_segments = align(segments, align_model, metadata, audio_file_path, "cpu")

    # 提取说话人嵌入
    embeddings, segments_info = extract_speaker_embeddings(audio_file_path, speaker_encoder)

    # 识别说话人（与已注册说话人匹配）
    speaker_segments = identify_speakers(embeddings, segments_info, registered_speakers)

    # 匹配转录与说话人
    segments_with_speakers = match_transcription_with_speakers(aligned_segments, speaker_segments)

    # 格式化输出带时间戳和说话人的文本
    formatted_output = []
    for segment in segments_with_speakers:
        start_time = format_time(segment["start"])
        end_time = format_time(segment["end"])
        text = segment["text"]
        speaker = segment["speaker"]
        confidence = segment.get("confidence", 0)

        # 添加可选的置信度显示
        confidence_str = f"(置信度: {confidence:.2f})" if confidence > 0 else ""
        formatted_output.append(f"[{start_time}-{end_time}] 【{speaker}】{confidence_str}: {text}")

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
    audio_file = "data/家有儿女吃饭.mp3"  # 输入音频文件路径
    speakers_dir = "data/sample"  # 包含已注册说话人的目录
    device = "cpu"  # 如果有GPU支持，可以改为 "cuda"

    # 加载模型
    whisper_model, align_model, metadata = load_model_with_timestamps(model_name, device)
    speaker_encoder = load_speaker_encoder(device)

    # 注册已知说话人
    registered_speakers = register_speakers_from_directory(speaker_encoder, speakers_dir)
    print(f"Registered {len(registered_speakers)} speakers: {[s['name'] for s in registered_speakers]}")

    # 转录音频文件并生成带时间戳和已识别说话人的文本
    transcription = transcribe_audio_with_registered_speakers(
        whisper_model, align_model, metadata,
        speaker_encoder, audio_file,
        registered_speakers
    )

    # 输出结果
    print("Transcription Result with Timestamps and Identified Speakers:")
    print(transcription)

    # 保存结果到文件
    with open("transcription_with_identified_speakers.txt", "w", encoding="utf-8") as f:
        f.write(transcription)
    print(f"Transcription saved to transcription_with_identified_speakers.txt")