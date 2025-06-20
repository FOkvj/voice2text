import os
from datetime import datetime, timedelta

import librosa
import pandas as pd
from funasr import AutoModel
from speechbrain.inference import EncoderClassifier

from voice2text.tran.voiceprint_manager import VoicePrintManager

# 默认路径常量
DEFAULT_VOICE_PRINTS_PATH = os.path.join(os.path.expanduser("~"), ".cache", "voice_print")
DEFAULT_VOICEPRINT_MAX_LENGTH = 30  # 默认声纹最大长度（秒）


class AudioProcessor:
    """处理音频文件操作"""

    @staticmethod
    def process_audio(audio_path, target_sr=16000):
        """处理音频文件并返回适合提取声纹的音频数据"""
        wav, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return wav, sr

    @staticmethod
    def save_audio_segment(wav_data, output_path, sr=16000):
        """将音频片段保存为文件"""
        sf.write(output_path, wav_data, sr)
        return output_path

    @staticmethod
    def trim_audio(wav_data, sr, max_length_sec):
        """将音频修剪到指定的最大长度"""
        max_samples = int(max_length_sec * sr)
        if len(wav_data) > max_samples:
            # 从中间截取，以获得更好的声纹质量
            mid_point = len(wav_data) // 2
            half_length = max_samples // 2
            start_idx = max(0, mid_point - half_length)
            end_idx = min(len(wav_data), mid_point + half_length)
            return wav_data[start_idx:end_idx]
        return wav_data


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
                savedir="models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
            print("Successfully loaded ECAPA-TDNN speaker verification model.")
        except Exception as e:
            print(f"Error loading ECAPA-TDNN model: {e}")
            raise


import os
import soundfile as sf


class FunASRTranscriber:
    """使用FunASR进行转写并识别说话人的主类"""

    def __init__(self, device="cpu",
                 model_manager=None, voice_print_manager=None,
                 voice_prints_path=DEFAULT_VOICE_PRINTS_PATH,
                 max_voiceprint_length=DEFAULT_VOICEPRINT_MAX_LENGTH,
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
                voice_prints_dir=voice_prints_path,
                max_voiceprint_length=max_voiceprint_length
            )
        else:
            self.voice_print_manager = voice_print_manager

    def _format_output_segment(self, start_time, end_time, speaker_name, text, file_location, file_date, file_time):
        """格式化输出段落，包含地点和时间信息"""
        # 解析原始时间戳
        start_sec = self._parse_time(start_time)
        end_sec = self._parse_time(end_time)

        # 计算相对于文件时间的偏移
        file_datetime = datetime.strptime(f"{file_date} {file_time}", "%Y/%m/%d %H:%M:%S")
        start_datetime = file_datetime + timedelta(seconds=start_sec)
        end_datetime = file_datetime + timedelta(seconds=end_sec)

        # 格式化新时间戳
        new_start_time = start_datetime.strftime("%Y/%m/%d-%H:%M:%S")
        new_end_time = end_datetime.strftime("%Y/%m/%d-%H:%M:%S")

        return f"[{file_location}][{new_start_time}-{new_end_time}] [{speaker_name}] {text}"

    def transcribe_file(self, audio_file_path,
                        batch_size_s=300,
                        hotword='',
                        threshold=0.4,
                        auto_register_unknown=True,
                        file_location=None,
                        file_date=None,
                        file_time=None):
        """转写音频文件并进行说话人识别，可选自动注册未知说话人
        返回: (转写文本, 自动注册的说话人字典和音频样本, 音频时长秒数)
        """
        print(f"转写音频文件: {audio_file_path}...")

        # 从文件名中提取地点和时间信息
        if not file_location or not file_date or not file_time:
            print("警告: 文件名不符合'地点_时间_名称.mp3'格式，将使用默认时间戳格式")
            use_filename_info = False
        else:
            use_filename_info = True

        # 1. 使用FunASR进行转写和说话人分离
        print("步骤1: 使用FunASR进行转写(包含说话人分离)...")
        result = self.asr_model.generate(
            input=audio_file_path,
            batch_size_s=batch_size_s,
            hotword=hotword
        )

        # 初始化变量
        transcript = ""
        auto_registered_speakers = {}
        voiceprint_audio_samples = {}
        audio_duration = 0.0  # 初始化音频时长

        # 检查是否有sentence_info字段（包含说话人分离信息）
        if 'sentence_info' not in result[0]:
            print("警告: FunASR结果不包含说话人分离信息(sentence_info字段)。")
            print("使用完整转写文本，不包含说话人信息。")
            # 提取文本和时间戳
            full_text = result[0]['text']
            timestamps = result[0].get('timestamp', [])

            # 计算音频时长（如果没有时间戳，使用文本长度估算）
            if timestamps:
                audio_duration = timestamps[-1][1] / 1000.0  # 转换为秒
            else:
                audio_duration = len(full_text) / 10  # 简单估算

            # 创建单一说话人的转写结果
            if use_filename_info:
                transcript = self._format_output_segment(
                    "00:00:00",
                    self._format_time(audio_duration),
                    "未知说话人",
                    full_text,
                    file_location,
                    file_date,
                    file_time
                )
            else:
                transcript = f"[00:00:00-{self._format_time(audio_duration)}] [未知说话人] {full_text}"

        else:
            # 2. 从sentence_info中提取说话人分段信息
            print("步骤2: 处理说话人分段信息...")
            sentence_segments = result[0]['sentence_info']
            print(f"发现 {len(sentence_segments)} 个说话人片段。")

            # 计算音频总时长（取最后一个片段的结束时间）
            audio_duration = max(seg['end'] for seg in sentence_segments) / 1000.0  # 转换为秒

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

            # 3. 使用新方法识别说话人身份，启用自动注册
            speaker_mapping, auto_registered, voiceprint_samples = self.voice_print_manager.identify_speakers(
                diarize_segments,
                audio_file_path,
                threshold=threshold,
                auto_register=auto_register_unknown
            )
            auto_registered_speakers = auto_registered
            voiceprint_audio_samples = voiceprint_samples

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
                if use_filename_info:
                    formatted_segment = self._format_output_segment(
                        start_time,
                        end_time,
                        speaker_name,
                        text,
                        file_location,
                        file_date,
                        file_time
                    )
                else:
                    formatted_segment = f"[{start_time}-{end_time}] [{speaker_name}] {text}"

                speaker_segments_formatted.append(formatted_segment)

            # 将片段连接成单个字符串
            transcript = " \n ".join(speaker_segments_formatted)

        # 保存输出到文件
        output_file = audio_file_path.rsplit(".", 1)[0] + "_transcript.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"转写结果已保存到 {output_file}")

        # 如果自动注册了新声纹，打印提示信息
        if auto_registered_speakers:
            print("\n自动注册了以下未知说话人:")
            for speaker_id, info in auto_registered_speakers.items():
                print(
                    f"  - 说话人ID: {speaker_id} (原始ID: {info['original_id']}, 音频长度: {info['audio_length']:.2f}秒)")
                if 'audio_path' in info and info['audio_path']:
                    print(f"    声纹音频样本: {info['audio_path']}")

            print("\n您可以使用 rename_voice_print 方法为这些自动注册的声纹分配人名。")
            print("例如: transcriber.rename_voice_print('" + list(auto_registered_speakers.keys())[0] + "', '新人名')")

        # 返回结果字典
        return {
            "transcript": transcript,
            "auto_registered_speakers": auto_registered_speakers,
            "voiceprint_audio_samples": voiceprint_audio_samples,
            "audio_duration": audio_duration,
            "output_file": output_file
        }

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

    def list_registered_voices(self, include_unnamed=True):
        """列出所有注册声纹，包括未命名声纹"""
        return self.voice_print_manager.list_registered_voices(include_unnamed)

    def clear_voice_prints(self):
        """清空所有已注册的声纹"""
        return self.voice_print_manager.clear_voice_prints()

    def rename_voice_print(self, speaker_id, new_name):
        """将自动注册的声纹ID重命名为人名"""
        return self.voice_print_manager.rename_voice_print(speaker_id, new_name)

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
    audio_file = "../../data/刘星家_20231212_122300_家有儿女吃饭.mp3"
    device = "cpu"  # 如果有GPU可用，改为"cuda"

    # 创建转写器
    transcriber = FunASRTranscriber(
        device=device,
        voice_prints_path=os.path.join(os.path.expanduser("~"), ".cache"),
        max_voiceprint_length=30,  # 限制声纹最大长度为30秒
        funasr_model="paraformer-zh",
        funasr_model_revision="v2.0.4",
        # funasr_model="dengcunqin/speech_paraformer_large_asr_mtl-16k-common-vocab11666-pytorch",
        # funasr_model_revision="master",
        vad_model="fsmn-vad",
        vad_model_revision="v2.0.4",
        # vad_model=None,
        # vad_model_revision=None,
        punc_model="ct-punc",
        punc_model_revision="v2.0.4",
        spk_model="cam++",
        spk_model_revision="v2.0.2"
    )

    # 列出注册声纹
    transcriber.clear_voice_prints()
    transcriber.list_registered_voices()


    # 转写音频文件，启用自动注册未知声纹
    result = transcriber.transcribe_file(
        audio_file,
        threshold=0.5,
        auto_register_unknown=True
    )

    # 打印结果
    print("\nTranscription with Timestamps and Speaker Identification:")
    print(result["transcript"])

    # 打印自动注册的声纹信息
    if result["auto_registered_speakers"]:
        print("\n自动注册的声纹信息:")
        for speaker_id, info in result["auto_registered_speakers"].items():
            print(f"  - {speaker_id}: 音频长度 {info['audio_length']:.2f}秒")
            if 'audio_path' in info:
                print(f"    声纹音频样本: {info['audio_path']}")

    # 示范如何重命名声纹
    if result["auto_registered_speakers"]:
        first_speaker = list(result["auto_registered_speakers"].keys())[0]
        print(f"\n示例 - 重命名声纹: transcriber.rename_voice_print('{first_speaker}', '刘星')")


if __name__ == "__main__":
    main()