import glob
import os
import pickle
import shutil
import uuid

import librosa
import numpy as np
import soundfile as sf
import torch


class VoicePrintManager:
    """管理声纹注册和识别 - 使用目录结构存储声纹和样本，确保声纹嵌入向量随样本变化而更新"""

    def __init__(self, speaker_encoder,
                 voice_prints_dir=os.path.join(os.path.expanduser("~"), ".cache", "voice_prints"),
                 max_voiceprint_length=30,
                 max_samples_per_voiceprint=5):
        """
        初始化声纹管理器

        Args:
            speaker_encoder: 声纹提取模型
            voice_prints_dir: 声纹存储的根目录
            max_voiceprint_length: 声纹最大长度（秒）
            max_samples_per_voiceprint: 每个声纹最多存储的样本数量
        """
        self.speaker_encoder = speaker_encoder
        self.voice_prints_dir = voice_prints_dir
        self.max_voiceprint_length = max_voiceprint_length
        self.max_samples_per_voiceprint = max_samples_per_voiceprint
        self.target_sr = 16000  # 标准采样率

        # 创建声纹目录（如果不存在）
        os.makedirs(self.voice_prints_dir, exist_ok=True)

        # 声纹嵌入向量缓存（用于提高性能）
        self.voice_prints_cache = {}
        self.unnamed_voice_prints_cache = {}

        # 初始化时从存储的样本重新计算所有声纹嵌入向量
        self._recalculate_all_embeddings()

    def _recalculate_all_embeddings(self):
        """从存储的样本重新计算所有声纹嵌入向量"""
        print("正在从样本重新计算所有声纹嵌入向量...")

        # 清空缓存
        self.voice_prints_cache = {}
        self.unnamed_voice_prints_cache = {}

        # 遍历所有声纹目录
        for speaker_id in os.listdir(self.voice_prints_dir):
            speaker_dir = os.path.join(self.voice_prints_dir, speaker_id)

            # 跳过非目录和隐藏目录
            if not os.path.isdir(speaker_dir) or speaker_id.startswith('.'):
                continue

            # 获取所有样本文件
            sample_files = glob.glob(os.path.join(speaker_dir, "sample*.wav"))

            if sample_files:
                # 从所有样本计算嵌入向量
                embedding = self._calculate_embedding_from_samples(sample_files)

                # 存储到相应的缓存中
                if speaker_id.startswith("Speaker_"):
                    self.unnamed_voice_prints_cache[speaker_id] = embedding
                else:
                    self.voice_prints_cache[speaker_id] = embedding

        print(
            f"重新计算完成，命名声纹: {len(self.voice_prints_cache)}，未命名声纹: {len(self.unnamed_voice_prints_cache)}")

        # 保存更新后的缓存
        self._save_embedding_cache()

    def _calculate_embedding_from_samples(self, sample_files):
        """
        从多个样本文件计算平均嵌入向量

        Args:
            sample_files: 样本文件路径列表

        Returns:
            平均嵌入向量
        """
        if not sample_files:
            return None

        all_embeddings = []

        for sample_path in sample_files:
            try:
                # 加载音频样本
                wav, sr = librosa.load(sample_path, sr=self.target_sr, mono=True)

                # 提取单个样本的嵌入向量
                embedding = self._extract_embedding(wav)
                all_embeddings.append(embedding)
            except Exception as e:
                print(f"处理样本 {sample_path} 时出错: {e}")

        if not all_embeddings:
            return None

        # 计算平均嵌入向量
        avg_embedding = np.mean(all_embeddings, axis=0)

        # 确保向量已归一化
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        return avg_embedding

    def _save_embedding_cache(self):
        """保存声纹嵌入向量缓存"""
        cache_path = os.path.join(self.voice_prints_dir, "embeddings_cache.pkl")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump((self.voice_prints_cache, self.unnamed_voice_prints_cache), f)
            print(f"已保存声纹嵌入向量缓存")
        except Exception as e:
            print(f"保存声纹缓存时出错: {e}")

    def get_voiceprint_sample_paths(self, voiceprint_id=None):
        """
        获取声纹样本文件路径列表

        Args:
            voiceprint_id: 声纹ID，如果为None则返回所有声纹的样本

        Returns:
            包含声纹样本文件ID的字典，格式为: {voiceprint_id: [sample1_id, sample2_id, ...]}
            其中sample_id的格式为 "voiceprint_id+sample1.wav"
        """
        result = {}

        if voiceprint_id:
            # 只返回指定声纹ID的样本
            voiceprint_dir = os.path.join(self.voice_prints_dir, voiceprint_id)
            if os.path.exists(voiceprint_dir) and os.path.isdir(voiceprint_dir):
                sample_files = sorted(glob.glob(os.path.join(voiceprint_dir, "sample*.wav")))
                sample_ids = [f"{voiceprint_id}+{os.path.basename(f)}" for f in sample_files]
                result[voiceprint_id] = sample_ids
        else:
            # 返回所有声纹的样本
            for vp_dir in os.listdir(self.voice_prints_dir):
                full_dir_path = os.path.join(self.voice_prints_dir, vp_dir)
                if os.path.isdir(full_dir_path) and not vp_dir.startswith('.'):
                    sample_files = sorted(glob.glob(os.path.join(full_dir_path, "sample*.wav")))
                    if sample_files:  # 只添加有样本文件的声纹
                        sample_ids = [f"{vp_dir}+{os.path.basename(f)}" for f in sample_files]
                        result[vp_dir] = sample_ids

        return result

    def get_sample_path_by_id(self, sample_id):
        """
        根据样本ID获取实际文件路径

        Args:
            sample_id: 样本ID，格式为 "voiceprint_id+sample1.wav"

        Returns:
            样本文件的完整路径，如果不存在则返回None
        """
        if "+" not in sample_id:
            print(f"无效的样本ID格式: {sample_id}")
            return None

        voiceprint_id, sample_filename = sample_id.split("+", 1)
        sample_path = os.path.join(self.voice_prints_dir, voiceprint_id, sample_filename)

        if os.path.exists(sample_path):
            return sample_path
        else:
            print(f"样本文件不存在: {sample_path}")
            return None

    def clear_voice_prints(self):
        """清空所有已注册的声纹"""
        # 保留根目录，删除所有内容
        for item in os.listdir(self.voice_prints_dir):
            item_path = os.path.join(self.voice_prints_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            elif item != "embeddings_cache.pkl":  # 保留缓存文件名但清空内容
                os.remove(item_path)

        # 清空缓存
        self.voice_prints_cache = {}
        self.unnamed_voice_prints_cache = {}
        self._save_embedding_cache()

        print("已清空所有声纹数据")

    def register_voice(self, person_name, audio_file_path):
        """
        从音频文件注册新声纹

        Args:
            person_name: 人名，将作为声纹ID
            audio_file_path: 音频文件路径

        Returns:
            声纹ID和样本ID
        """
        print(f"正在注册声纹: {person_name}")

        # 处理音频文件
        wav, sr = self._process_audio(audio_file_path)

        # 限制声纹长度
        wav = self._trim_audio(wav, sr, self.max_voiceprint_length)

        # 创建或获取声纹目录
        voiceprint_dir = os.path.join(self.voice_prints_dir, person_name)
        os.makedirs(voiceprint_dir, exist_ok=True)

        # 查找下一个可用的样本编号
        existing_samples = glob.glob(os.path.join(voiceprint_dir, "sample*.wav"))
        next_sample_num = len(existing_samples) + 1

        # 检查是否超过最大样本数
        if next_sample_num > self.max_samples_per_voiceprint:
            print(f"警告: 声纹 {person_name} 已达到最大样本数 {self.max_samples_per_voiceprint}")
            # 删除最旧的样本
            oldest_sample = sorted(existing_samples)[0]
            os.remove(oldest_sample)
            next_sample_num = self.max_samples_per_voiceprint

        # 保存音频样本
        sample_filename = f"sample{next_sample_num}.wav"
        sample_path = os.path.join(voiceprint_dir, sample_filename)
        self._save_audio(wav, sample_path, sr)

        # 获取更新后的所有样本文件
        updated_samples = glob.glob(os.path.join(voiceprint_dir, "sample*.wav"))

        # 从所有样本计算嵌入向量
        embedding = self._calculate_embedding_from_samples(updated_samples)

        # 更新缓存
        self.voice_prints_cache[person_name] = embedding
        self._save_embedding_cache()

        sample_id = f"{person_name}+{sample_filename}"
        print(f"声纹注册成功: {person_name}，样本ID: {sample_id}")

        return person_name, sample_id

    def register_voice_from_embedding(self, embedding, audio_data=None, sr=16000):
        """
        从嵌入向量注册未命名声纹

        Args:
            embedding: 声纹嵌入向量
            audio_data: 音频数据
            sr: 采样率

        Returns:
            声纹ID和样本ID
        """
        # 生成唯一ID
        speaker_id = f"Speaker_{uuid.uuid4().hex[:8]}"

        # 创建声纹目录
        voiceprint_dir = os.path.join(self.voice_prints_dir, speaker_id)
        os.makedirs(voiceprint_dir, exist_ok=True)

        # 如果提供了音频数据，保存声纹音频样本
        sample_id = None
        if audio_data is not None:
            sample_filename = "sample1.wav"
            sample_path = os.path.join(voiceprint_dir, sample_filename)
            self._save_audio(audio_data, sample_path, sr)
            sample_id = f"{speaker_id}+{sample_filename}"
            print(f"声纹样本已保存: {sample_path}")

            # 从样本计算嵌入向量
            embedding = self._extract_embedding(audio_data)

        # 保存嵌入向量到缓存
        self.unnamed_voice_prints_cache[speaker_id] = embedding
        self._save_embedding_cache()

        print(f"未命名声纹注册成功: {speaker_id}")
        return speaker_id, sample_id

    def rename_voice_print(self, speaker_id, new_name):
        """
        重命名声纹ID为人名

        Args:
            speaker_id: 原声纹ID
            new_name: 新名称

        Returns:
            是否重命名成功
        """
        # 检查原声纹目录是否存在
        src_dir = os.path.join(self.voice_prints_dir, speaker_id)
        if not os.path.exists(src_dir) or not os.path.isdir(src_dir):
            print(f"声纹 {speaker_id} 不存在")
            return False

        # 检查目标目录是否已存在
        dst_dir = os.path.join(self.voice_prints_dir, new_name)
        if os.path.exists(dst_dir):
            print(f"目标名称 {new_name} 已存在，请选择其他名称")
            return False

        # 更新缓存
        embedding = None
        is_unnamed = False

        if speaker_id in self.unnamed_voice_prints_cache:
            embedding = self.unnamed_voice_prints_cache[speaker_id]
            del self.unnamed_voice_prints_cache[speaker_id]
            is_unnamed = True
        elif speaker_id in self.voice_prints_cache:
            embedding = self.voice_prints_cache[speaker_id]
            del self.voice_prints_cache[speaker_id]

        # 重命名目录
        try:
            shutil.move(src_dir, dst_dir)

            # 如果有嵌入向量，更新到新名称
            if embedding is not None:
                self.voice_prints_cache[new_name] = embedding
                self._save_embedding_cache()

            print(f"声纹 {speaker_id} 已重命名为 {new_name}")
            return True
        except Exception as e:
            print(f"重命名声纹时出错: {e}")
            return False

    def register_voices_from_directory(self, directory_path):
        """
        从目录中所有音频文件注册声纹

        Args:
            directory_path: 包含音频文件的目录路径

        Returns:
            注册的声纹ID和样本ID字典
        """
        directory_path = os.path.expanduser(directory_path)
        print(f"从目录注册声纹: {directory_path}")
        if not os.path.isdir(directory_path):
            print(f"错误: {directory_path} 不是有效目录")
            return {}

        # 支持的音频扩展名列表
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
        registered_voices = {}

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
                voiceprint_id, sample_id = self.register_voice(speaker_name, file_path)
                registered_voices[voiceprint_id] = sample_id
            except Exception as e:
                print(f"注册 {speaker_name} 时出错: {e}")

        print(f"成功从目录注册了 {len(registered_voices)} 个声纹")
        return registered_voices

    def identify_speakers(self, diarize_segments, audio_file_path, threshold=0.3, auto_register=True):
        """
        基于注册声纹识别说话人

        Args:
            diarize_segments: 语音分段信息
            audio_file_path: 音频文件路径
            threshold: 识别阈值
            auto_register: 是否自动注册未知声纹

        Returns:
            说话人身份映射、自动注册的声纹信息和音频样本ID
        """
        print("基于已注册声纹识别说话人...")

        # 处理音频文件
        wav, sr = self._process_audio(audio_file_path)

        # 存储说话人身份的字典
        speaker_identities = {}
        # 用于存储自动注册的未知声纹信息
        auto_registered_speakers = {}
        # 用于存储每个自动注册声纹的音频样本ID
        voiceprint_audio_samples = {}

        # 处理每个唯一的说话人
        for speaker in diarize_segments['speaker'].unique():
            # 获取该说话人的所有片段
            speaker_segments = diarize_segments[diarize_segments['speaker'] == speaker]

            print(f"处理说话人 {speaker}, 共 {len(speaker_segments)} 个片段")

            # 收集该说话人的所有音频片段
            speaker_wavs = []
            segment_times = []  # 存储每个片段的时间信息

            for _, segment in speaker_segments.iterrows():
                start = segment['start']
                end = segment['end']

                # 提取时间片段
                start_sample = int(start * sr)
                end_sample = min(int(end * sr), len(wav))

                if end_sample > start_sample:
                    segment_audio = wav[start_sample:end_sample]
                    speaker_wavs.append(segment_audio)
                    segment_times.append((start, end))

            if speaker_wavs:
                # 合并所有片段以获得更好的声纹
                combined_wav = np.concatenate(speaker_wavs)
                audio_duration = len(combined_wav) / sr

                print(f"说话人 {speaker} 的合并音频长度: {audio_duration:.2f}秒")

                # 限制音频长度
                if audio_duration > self.max_voiceprint_length:
                    print(f"音频超过最大长度限制，将裁剪到 {self.max_voiceprint_length} 秒")
                    combined_wav = self._trim_audio(combined_wav, sr, self.max_voiceprint_length)
                    audio_duration = self.max_voiceprint_length

                # 提取嵌入向量
                embedding = self._extract_embedding(combined_wav)

                # 与注册声纹和未命名声纹比较
                best_match = None
                best_score = 0
                is_unnamed = False

                # 先检查命名声纹
                for person_name, registered_embedding in self.voice_prints_cache.items():
                    # 计算余弦相似度
                    similarity = self._cosine_similarity(embedding, registered_embedding)
                    print(f"与 {person_name} 的相似度: {similarity:.4f}")

                    if similarity > best_score:
                        best_score = similarity
                        best_match = person_name
                        is_unnamed = False

                # 再检查未命名声纹
                for speaker_id, unnamed_embedding in self.unnamed_voice_prints_cache.items():
                    # 计算余弦相似度
                    similarity = self._cosine_similarity(embedding, unnamed_embedding)
                    print(f"与 {speaker_id} 的相似度: {similarity:.4f}")

                    if similarity > best_score:
                        best_score = similarity
                        best_match = speaker_id
                        is_unnamed = True

                # 分配身份（如果相似度高于阈值）
                if best_score >= threshold:
                    speaker_identities[speaker] = best_match
                    print(
                        f"说话人 {speaker} 被识别为: {best_match} (相似度: {best_score:.4f}, 是未命名声纹: {is_unnamed})")
                else:
                    # 未识别到已知声纹，如果启用了自动注册，则注册为新声纹
                    if auto_register:
                        # 注册声纹并保存音频样本
                        new_speaker_id, sample_id = self.register_voice_from_embedding(
                            embedding, combined_wav, sr
                        )
                        speaker_identities[speaker] = new_speaker_id
                        auto_registered_speakers[new_speaker_id] = {
                            'original_id': speaker,
                            'audio_length': audio_duration,
                            'sample_id': sample_id
                        }
                        if sample_id:
                            voiceprint_audio_samples[new_speaker_id] = sample_id
                        print(f"未识别的说话人 {speaker} 已自动注册为新声纹: {new_speaker_id}")
                    else:
                        speaker_identities[speaker] = f"未知:{speaker}"
                        print(f"说话人 {speaker} 未能识别 (最高相似度: {best_score:.4f}, 低于阈值 {threshold})")
            else:
                speaker_identities[speaker] = f"无有效音频_{speaker}"
                print(f"说话人 {speaker} 没有有效的音频片段")

        # 返回识别结果、自动注册的声纹信息和音频样本ID
        return speaker_identities, auto_registered_speakers, voiceprint_audio_samples

    def list_registered_voices(self, include_unnamed=True):
        """
        列出所有注册声纹

        Returns:
            包含注册声纹和样本ID的字典
        """
        # 获取所有声纹样本路径
        all_samples = self.get_voiceprint_sample_paths()

        # 分离命名声纹和未命名声纹
        named_voice_prints = {}
        unnamed_voice_prints = {}

        for vp_id, samples in all_samples.items():
            if vp_id.startswith("Speaker_"):
                if include_unnamed:
                    unnamed_voice_prints[vp_id] = samples
            else:
                named_voice_prints[vp_id] = samples

        # 打印结果
        named_count = len(named_voice_prints)
        unnamed_count = len(unnamed_voice_prints)

        if named_count == 0 and unnamed_count == 0:
            print("No voice prints registered.")
            return {}

        if named_count > 0:
            print(f"当前已注册的命名声纹 ({named_count}):")
            for i, (name, samples) in enumerate(named_voice_prints.items(), 1):
                sample_count = len(samples)
                print(f"  {i}. {name} ({sample_count} 个样本)")
                for j, sample_id in enumerate(samples, 1):
                    print(f"     - 样本 {j}: {sample_id}")

        if include_unnamed and unnamed_count > 0:
            print(f"\n当前已注册的未命名声纹 ({unnamed_count}):")
            for i, (speaker_id, samples) in enumerate(unnamed_voice_prints.items(), 1):
                sample_count = len(samples)
                print(f"  {i}. {speaker_id} ({sample_count} 个样本)")
                for j, sample_id in enumerate(samples, 1):
                    print(f"     - 样本 {j}: {sample_id}")

        # 返回结果
        return {
            "named_voice_prints": named_voice_prints,
            "unnamed_voice_prints": unnamed_voice_prints if include_unnamed else {}
        }

    def _process_audio(self, audio_path, target_sr=16000):
        """处理音频文件并返回适合提取声纹的音频数据"""
        import librosa
        wav, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return wav, sr

    def _save_audio(self, wav_data, output_path, sr=16000):
        """将音频片段保存为文件"""
        sf.write(output_path, wav_data, sr)
        return output_path

    def _trim_audio(self, wav_data, sr, max_length_sec):
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

    def _extract_embedding(self, wav_data):
        """从音频数据提取声纹嵌入向量"""
        # 转换为tensor
        wav_tensor = torch.FloatTensor(wav_data).unsqueeze(0)

        # 提取声纹嵌入向量
        with torch.no_grad():
            embedding = self.speaker_encoder.encode_batch(wav_tensor)
            embedding = embedding.squeeze().cpu().numpy()

        # 确保向量已归一化
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def _cosine_similarity(self, a, b):
        """计算两个向量之间的余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def update_speaker_embedding(self, speaker_id):
        """
        更新指定说话人的嵌入向量

        Args:
            speaker_id: 说话人ID

        Returns:
            是否成功更新
        """
        speaker_dir = os.path.join(self.voice_prints_dir, speaker_id)

        if not os.path.exists(speaker_dir) or not os.path.isdir(speaker_dir):
            print(f"声纹 {speaker_id} 不存在")
            return False

        # 获取所有样本文件
        sample_files = glob.glob(os.path.join(speaker_dir, "sample*.wav"))

        if not sample_files:
            print(f"声纹 {speaker_id} 没有样本文件")
            return False

        # 从所有样本计算嵌入向量
        embedding = self._calculate_embedding_from_samples(sample_files)

        if embedding is None:
            print(f"无法从样本计算嵌入向量")
            return False

        # 更新缓存
        if speaker_id.startswith("Speaker_"):
            self.unnamed_voice_prints_cache[speaker_id] = embedding
        else:
            self.voice_prints_cache[speaker_id] = embedding

        self._save_embedding_cache()

        print(f"已更新 {speaker_id} 的声纹嵌入向量")
        return True

    def delete_sample(self, sample_id):
        """
        删除指定的声纹样本并更新嵌入向量

        Args:
            sample_id: 样本ID，格式为 "voiceprint_id+sample1.wav"

        Returns:
            是否成功删除
        """
        if "+" not in sample_id:
            print(f"无效的样本ID格式: {sample_id}")
            return False

        voiceprint_id, sample_filename = sample_id.split("+", 1)
        sample_path = os.path.join(self.voice_prints_dir, voiceprint_id, sample_filename)

        if not os.path.exists(sample_path):
            print(f"样本文件不存在: {sample_path}")
            return False

        # 删除样本文件
        try:
            os.remove(sample_path)
            print(f"已删除样本: {sample_id}")
        except Exception as e:
            print(f"删除样本时出错: {e}")
            return False

        # 检查是否还有其他样本
        remaining_samples = glob.glob(os.path.join(self.voice_prints_dir, voiceprint_id, "sample*.wav"))

        if remaining_samples:
            # 更新嵌入向量
            self.update_speaker_embedding(voiceprint_id)
        else:
            # 没有样本了，删除声纹
            print(f"声纹 {voiceprint_id} 已没有样本，将删除声纹")

            # 从缓存中删除
            if voiceprint_id in self.voice_prints_cache:
                del self.voice_prints_cache[voiceprint_id]
            elif voiceprint_id in self.unnamed_voice_prints_cache:
                del self.unnamed_voice_prints_cache[voiceprint_id]

            # 删除目录
            try:
                shutil.rmtree(os.path.join(self.voice_prints_dir, voiceprint_id))
            except Exception as e:
                print(f"删除声纹目录时出错: {e}")

        # 保存更新后的缓存
        self._save_embedding_cache()
        return True

    def add_sample(self, speaker_id, audio_file_path):
        """
        为现有声纹添加新样本

        Args:
            speaker_id: 说话人ID
            audio_file_path: 音频文件路径

        Returns:
            新样本ID
        """
        speaker_dir = os.path.join(self.voice_prints_dir, speaker_id)

        if not os.path.exists(speaker_dir) or not os.path.isdir(speaker_dir):
            print(f"声纹 {speaker_id} 不存在")
            return None

        # 处理音频文件
        wav, sr = self._process_audio(audio_file_path)

        # 限制声纹长度
        wav = self._trim_audio(wav, sr, self.max_voiceprint_length)

        # 查找下一个可用的样本编号
        existing_samples = glob.glob(os.path.join(speaker_dir, "sample*.wav"))
        next_sample_num = len(existing_samples) + 1

        # 检查是否超过最大样本数
        if next_sample_num > self.max_samples_per_voiceprint:
            print(f"警告: 声纹 {speaker_id} 已达到最大样本数 {self.max_samples_per_voiceprint}")
            # 删除最旧的样本
            oldest_sample = sorted(existing_samples)[0]
            os.remove(oldest_sample)
            next_sample_num = self.max_samples_per_voiceprint

        # 保存音频样本
        sample_filename = f"sample{next_sample_num}.wav"
        sample_path = os.path.join(speaker_dir, sample_filename)
        self._save_audio(wav, sample_path, sr)

        # 更新嵌入向量
        self.update_speaker_embedding(speaker_id)

        sample_id = f"{speaker_id}+{sample_filename}"
        print(f"已为声纹 {speaker_id} 添加新样本: {sample_id}")

        return sample_id