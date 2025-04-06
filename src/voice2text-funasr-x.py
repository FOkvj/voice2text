import os
import pickle
import numpy as np
import pandas as pd
import librosa
import torch
from funasr import AutoModel
from speechbrain.inference import EncoderClassifier

# 默认路径常量
DEFAULT_VOICE_PRINTS_PATH = os.path.join(os.path.expanduser("~"), ".cache", "voice_prints_ecapa.pkl")


class AudioProcessor:
    """处理音频文件操作"""

    @staticmethod
    def process_audio(audio_path, target_sr=16000):
        """处理音频文件并返回适合提取声纹的音频数据"""
        wav, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return wav, sr


class ModelManager:
    """管理所有必需模型的加载和初始化"""

    def __init__(self, device="cpu",
                 funasr_model="paraformer-zh",
                 funasr_model_revision="v2.0.4",
                 vad_model="fsmn-vad",
                 vad_model_revision="v2.0.4",
                 punc_model="ct-punc",
                 punc_model_revision="v2.0.4",
                 spk_model="cam++",
                 spk_model_revision="v2.0.2"):
        self.device = device

        # FunASR模型参数
        self.funasr_model = funasr_model
        self.funasr_model_revision = funasr_model_revision
        self.vad_model = vad_model
        self.vad_model_revision = vad_model_revision
        self.punc_model = punc_model
        self.punc_model_revision = punc_model_revision
        self.spk_model = spk_model
        self.spk_model_revision = spk_model_revision

        self.asr_model = None
        self.speaker_encoder = None

    def load_all_models(self):
        """加载所有必要的模型"""
        print(f"Loading FunASR model: {self.funasr_model} with speaker model: {self.spk_model}...")
        self._load_funasr_model()
        self._load_speaker_encoder()
        return (self.asr_model, self.speaker_encoder)

    def _load_funasr_model(self):
        """加载FunASR语音识别模型"""
        try:
            self.asr_model = AutoModel(
                model=self.funasr_model,
                model_revision=self.funasr_model_revision,
                vad_model=self.vad_model,
                vad_model_revision=self.vad_model_revision,
                punc_model=self.punc_model,
                punc_model_revision=self.punc_model_revision,
                spk_model=self.spk_model,
                spk_model_revision=self.spk_model_revision
            )
            print("Successfully loaded FunASR model with speaker diarization.")
        except Exception as e:
            print(f"Error loading FunASR model: {e}")
            raise

    def _load_speaker_encoder(self):
        """加载ECAPA-TDNN声纹验证模型"""
        print("Loading ECAPA-TDNN speaker verification model...")
        try:
            self.speaker_encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
            print("Successfully loaded ECAPA-TDNN speaker verification model.")
        except Exception as e:
            print(f"Error loading ECAPA-TDNN model: {e}")
            raise


class VoicePrintManager:
    """管理声纹注册和识别"""

    def __init__(self, speaker_encoder, voice_prints_path=DEFAULT_VOICE_PRINTS_PATH):
        self.speaker_encoder = speaker_encoder
        self.voice_prints_path = voice_prints_path
        self.voice_prints = self._load_voice_prints()

    def _load_voice_prints(self):
        """加载已有声纹或创建空字典"""
        if os.path.exists(self.voice_prints_path):
            with open(self.voice_prints_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def clear_voice_prints(self):
        """清空所有已注册的声纹"""
        old_count = len(self.voice_prints)
        self.voice_prints = {}
        self.save_voice_prints()
        print(f"已清空 {old_count} 个声纹缓存。")

    def save_voice_prints(self):
        """保存当前声纹到磁盘"""
        # 确保目录存在
        os.makedirs(os.path.dirname(self.voice_prints_path), exist_ok=True)
        with open(self.voice_prints_path, 'wb') as f:
            pickle.dump(self.voice_prints, f)

    def register_voice(self, person_name, audio_file_path):
        """从音频文件注册新声纹"""
        print(f"Registering voice for: {person_name} from audio file")

        # 处理音频文件
        wav, sr = AudioProcessor.process_audio(audio_file_path)

        # 转换为tensor
        wav_tensor = torch.FloatTensor(wav).unsqueeze(0)

        # 提取声纹嵌入向量
        with torch.no_grad():
            embedding = self.speaker_encoder.encode_batch(wav_tensor)
            embedding = embedding.squeeze().cpu().numpy()

        # 添加到声纹集合
        self.voice_prints[person_name] = embedding
        self.save_voice_prints()

        print(f"Voice print for {person_name} registered successfully.")
        return self.voice_prints

    def register_voices_from_directory(self, directory_path):
        """从目录中所有音频文件注册声纹
        文件名(不包括扩展名)用作说话人名称。
        支持mp3、wav、m4a和其他常见音频格式。
        """
        print(f"Registering voice prints from directory: {directory_path}")

        if not os.path.isdir(directory_path):
            print(f"Error: {directory_path} is not a valid directory")
            return self.voice_prints

        # 支持的音频扩展名列表
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
        registered_count = 0

        # 处理目录中的每个音频文件
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            # 跳过目录和非音频文件
            if os.path.isdir(file_path):
                continue

            _, ext = os.path.splitext(filename)
            if ext.lower() not in audio_extensions:
                continue

            # 使用不带扩展名的文件名作为说话人名称
            speaker_name = os.path.splitext(filename)[0]

            try:
                # 注册声纹
                self.register_voice(speaker_name, file_path)
                registered_count += 1
            except Exception as e:
                print(f"Error registering {speaker_name}: {e}")

        print(f"Successfully registered {registered_count} voice prints from directory")
        return self.voice_prints

    def identify_speakers(self, diarize_segments, audio_file_path, threshold=0.3):
        """基于注册声纹识别说话人"""
        print("基于已注册声纹识别说话人...")

        # 如果没有注册声纹，返回未知说话人
        if not self.voice_prints:
            print("未找到已注册的声纹。")
            unique_speakers = diarize_segments['speaker'].unique()
            return {speaker: f"未知说话人_{i}" for i, speaker in enumerate(unique_speakers)}

        # 处理音频文件
        wav, sr = AudioProcessor.process_audio(audio_file_path)

        # 存储说话人身份的字典
        speaker_identities = {}

        # 处理每个唯一的说话人
        for speaker in diarize_segments['speaker'].unique():
            # 获取该说话人的所有片段
            speaker_segments = diarize_segments[diarize_segments['speaker'] == speaker]

            print(f"处理说话人 {speaker}, 共 {len(speaker_segments)} 个片段")

            # 收集该说话人的所有音频片段
            speaker_wavs = []

            for _, segment in speaker_segments.iterrows():
                start = segment['start']
                end = segment['end']

                # 提取时间片段
                start_sample = int(start * sr)
                end_sample = min(int(end * sr), len(wav))

                if end_sample > start_sample:
                    segment_audio = wav[start_sample:end_sample]
                    speaker_wavs.append(segment_audio)

            if speaker_wavs:
                # 合并所有片段以获得更好的声纹
                combined_wav = np.concatenate(speaker_wavs)

                print(f"说话人 {speaker} 的合并音频长度: {len(combined_wav) / sr:.2f}秒")

                # 确保有足够的音频数据用于可靠的声纹提取
                # min_samples = sr * 1.5  # 至少1.5秒
                # if len(combined_wav) >= min_samples:
                    # 转换为tensor
                wav_tensor = torch.FloatTensor(combined_wav).unsqueeze(0)

                # 提取嵌入向量
                with torch.no_grad():
                    embedding = self.speaker_encoder.encode_batch(wav_tensor)
                    embedding = embedding.squeeze().cpu().numpy()

                # 与注册声纹比较
                best_match = None
                best_score = 0

                for person_name, registered_embedding in self.voice_prints.items():
                    # 计算余弦相似度
                    similarity = self._cosine_similarity(embedding, registered_embedding)
                    print(f"与 {person_name} 的相似度: {similarity:.4f}")

                    if similarity > best_score:
                        best_score = similarity
                        best_match = person_name

                # 分配身份（如果相似度高于阈值）
                if best_score >= threshold:
                    speaker_identities[speaker] = best_match
                    print(f"说话人 {speaker} 被识别为: {best_match} (相似度: {best_score:.4f})")
                else:
                    speaker_identities[speaker] = f"未知:{speaker}"
                    print(f"说话人 {speaker} 未能识别 (最高相似度: {best_score:.4f}, 低于阈值 {threshold})")
                # else:
                #     speaker_identities[speaker] = f"片段过短_{speaker}"
                #     print(f"说话人 {speaker} 的音频时长不足 (仅 {len(combined_wav) / sr:.2f}秒)")
            else:
                speaker_identities[speaker] = f"无有效音频_{speaker}"
                print(f"说话人 {speaker} 没有有效的音频片段")

        return speaker_identities

    def _cosine_similarity(self, a, b):
        """计算两个向量之间的余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def list_registered_voices(self):
        """列出所有注册声纹"""
        if not self.voice_prints:
            print("No voice prints registered.")
            return

        print(f"Currently registered voice prints ({len(self.voice_prints)}):")
        for i, name in enumerate(self.voice_prints.keys(), 1):
            print(f"  {i}. {name}")


class FunASRTranscriber:
    """使用FunASR进行转写并识别说话人的主类"""

    def __init__(self, device="cpu",
                 model_manager=None, voice_print_manager=None,
                 voice_prints_path=DEFAULT_VOICE_PRINTS_PATH,
                 funasr_model="paraformer-zh",
                 funasr_model_revision="v2.0.4",
                 vad_model="fsmn-vad",
                 vad_model_revision="v2.0.4",
                 punc_model="ct-punc",
                 punc_model_revision="v2.0.4",
                 spk_model="cam++",
                 spk_model_revision="v2.0.2"):
        """初始化，可选依赖注入管理器"""

        # 创建模型管理器（如果未提供）
        if model_manager is None:
            model_manager = ModelManager(
                device=device,
                funasr_model=funasr_model,
                funasr_model_revision=funasr_model_revision,
                vad_model=vad_model,
                vad_model_revision=vad_model_revision,
                punc_model=punc_model,
                punc_model_revision=punc_model_revision,
                spk_model=spk_model,
                spk_model_revision=spk_model_revision
            )
            # 加载所有模型
            self.models = model_manager.load_all_models()
            self.asr_model, self.speaker_encoder = self.models
        else:
            # 使用提供的模型管理器
            self.model_manager = model_manager
            self.asr_model = model_manager.asr_model
            self.speaker_encoder = model_manager.speaker_encoder

        # 创建声纹管理器（如果未提供）
        if voice_print_manager is None and self.speaker_encoder is not None:
            self.voice_print_manager = VoicePrintManager(
                self.speaker_encoder,
                voice_prints_path=voice_prints_path
            )
        else:
            self.voice_print_manager = voice_print_manager

    def transcribe_file(self, audio_file_path, batch_size_s=300, hotword=''):
        """转写音频文件并进行说话人识别"""
        print(f"转写音频文件: {audio_file_path}...")

        # 1. 使用FunASR进行转写和说话人分离
        print("步骤1: 使用FunASR进行转写(包含说话人分离)...")
        result = self.asr_model.generate(
            input=audio_file_path,
            batch_size_s=batch_size_s,
            hotword=hotword
        )

        # 检查是否有sentence_info字段（包含说话人分离信息）
        if 'sentence_info' not in result[0]:
            print("警告: FunASR结果不包含说话人分离信息(sentence_info字段)。")
            print("使用完整转写文本，不包含说话人信息。")
            # 提取文本和时间戳
            full_text = result[0]['text']
            timestamps = result[0].get('timestamp', [])

            # 创建单一说话人的转写结果
            transcript = f"[00:00:00-{self._format_time(len(full_text) / 10)}] [未知说话人] {full_text}"

        else:
            # 2. 从sentence_info中提取说话人分段信息
            print("步骤2: 处理说话人分段信息...")
            sentence_segments = result[0]['sentence_info']
            print(f"发现 {len(sentence_segments)} 个说话人片段。")

            # 将sentence_info转换为DataFrame格式，方便处理
            segments_data = []
            for segment in sentence_segments:
                segments_data.append({
                    'speaker': segment['spk'],
                    'start': segment['start'] / 1000.0,  # 转换为秒
                    'end': segment['end'] / 1000.0,  # 转换为秒
                    'text': segment['text']
                })
            diarize_segments = pd.DataFrame(segments_data)

            # 3. 使用新方法识别说话人身份
            speaker_mapping = self.voice_print_manager.identify_speakers(diarize_segments, audio_file_path)

            # 4. 格式化输出
            print("步骤3: 格式化最终输出...")
            speaker_segments_formatted = []

            for segment in sentence_segments:
                # 获取时间和说话人信息
                start_ms = segment['start']
                end_ms = segment['end']
                spk_id = segment['spk']
                text = segment['text']

                # 转换时间为秒并格式化
                start_time = self._format_time(start_ms / 1000.0)
                end_time = self._format_time(end_ms / 1000.0)

                # 获取说话人身份（如果有映射）或使用通用标签
                speaker_name = speaker_mapping.get(spk_id, f"Speaker_{spk_id}")

                # 添加格式化的片段
                speaker_segments_formatted.append(f"[{start_time}-{end_time}] [{speaker_name}] {text}")

            # 将片段连接成单个字符串
            transcript = "\n".join(speaker_segments_formatted)

        # 保存输出到文件
        output_file = audio_file_path.rsplit(".", 1)[0] + "_transcript.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"转写结果已保存到 {output_file}")

        return transcript

    def _merge_same_speaker_segments(self, segments, max_gap_ms=3000):
        """合并相同说话人的短片段，用于提高声纹识别准确性

        Args:
            segments: FunASR返回的sentence_info片段列表
            max_gap_ms: 允许合并的最大时间间隔（毫秒）

        Returns:
            合并后的片段列表
        """
        if not segments:
            return []

        merged = []
        i = 0
        while i < len(segments):
            current = segments[i].copy()  # 复制当前片段
            i += 1

            # 查找相同说话人的后续片段进行合并
            while i < len(segments) and segments[i]['spk'] == current['spk']:
                next_seg = segments[i]
                # 如果时间间隔小于阈值，则合并
                if next_seg['start'] - current['end'] <= max_gap_ms:
                    # 更新结束时间
                    current['end'] = next_seg['end']
                    # 合并文本
                    current['text'] += " " + next_seg['text']
                    # 合并时间戳（如果存在）
                    if 'timestamp' in current and 'timestamp' in next_seg:
                        current['timestamp'].extend(next_seg['timestamp'])
                    i += 1
                else:
                    # 时间间隔太大，不合并
                    break

            merged.append(current)

        return merged

    def _merge_adjacent_segments(self, formatted_segments, max_gap_sec=1.0):
        """合并相同说话人的相邻段落"""
        if not formatted_segments:
            return []

        merged = []
        current_segment = formatted_segments[0]
        current_speaker = current_segment.split("] [")[1].split("]")[0]
        current_text = current_segment.split("] ", 2)[2]
        current_time_range = current_segment.split("] ")[0][1:]
        current_start = self._parse_time(current_time_range.split("-")[0])
        current_end = self._parse_time(current_time_range.split("-")[1])

        for i in range(1, len(formatted_segments)):
            segment = formatted_segments[i]
            parts = segment.split("] [")
            time_range = parts[0][1:]
            speaker = parts[1].split("]")[0]
            text = segment.split("] ", 2)[2]

            start = self._parse_time(time_range.split("-")[0])
            end = self._parse_time(time_range.split("-")[1])

            # 如果相同说话人且间隔小于阈值，则合并
            if speaker == current_speaker and start - current_end <= max_gap_sec:
                current_text += " " + text
                current_end = end
            else:
                # 添加当前已完成的段落
                merged_start = self._format_time(current_start)
                merged_end = self._format_time(current_end)
                merged.append(f"[{merged_start}-{merged_end}] [{current_speaker}] {current_text}")

                # 开始新段落
                current_speaker = speaker
                current_text = text
                current_start = start
                current_end = end

        # 添加最后一个段落
        merged_start = self._format_time(current_start)
        merged_end = self._format_time(current_end)
        merged.append(f"[{merged_start}-{merged_end}] [{current_speaker}] {current_text}")

        return merged

    def register_voice(self, person_name, audio_file_path):
        """注册新声纹"""
        return self.voice_print_manager.register_voice(person_name, audio_file_path)

    def register_voices_from_directory(self, directory_path):
        """从目录中所有音频文件注册声纹"""
        return self.voice_print_manager.register_voices_from_directory(directory_path)

    def list_registered_voices(self):
        """列出所有注册声纹"""
        return self.voice_print_manager.list_registered_voices()

    def clear_voice_prints(self):
        """清空所有已注册的声纹"""
        return self.voice_print_manager.clear_voice_prints()

    @staticmethod
    def _format_time(seconds):
        """将秒格式化为HH:MM:SS格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    @staticmethod
    def _parse_time(time_str):
        """将HH:MM:SS格式的时间字符串解析为秒数"""
        parts = time_str.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        return hours * 3600 + minutes * 60 + seconds


def main():
    """主函数，演示使用方法"""
    # 配置
    audio_file = "data/家有儿女吃饭.mp3"
    device = "cpu"  # 如果有GPU可用，改为"cuda"

    # 创建转写器
    transcriber = FunASRTranscriber(
        device=device,
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

    # 列出注册声纹、
    # transcrib
    transcriber.list_registered_voices()

    # 注册单独声纹（根据需要取消注释）
    # transcriber.register_voice("刘星", "data/sample/刘星.mp3")
    transcriber.clear_voice_prints()
    transcriber.register_voices_from_directory("/Users/dongzhancai1/Desktop/voice2text/src/data/sample")
    # transcriber.register_voice("刘梅", "data/sample/刘梅2.mp3")

    # 或者一次性注册目录中的所有声纹
    # transcriber.register_voices_from_directory("data/sample")

    # 转写音频文件
    transcript = transcriber.transcribe_file(audio_file)

    # 打印结果
    print("Transcription with Timestamps and Speaker Identification:")
    print(transcript)


if __name__ == "__main__":
    main()