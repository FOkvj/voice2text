# ============================================================================
# 基于向量数据库的增强声纹管理器
# ============================================================================

import tempfile
from typing import Union, BinaryIO, Optional, Tuple

import numpy as np

from voice2text.tran.vector_base import (
    VectorDatabaseFactory, VectorDBType,
    VectorDBConfigFactory
)


class AudioInputHandler:
    """音频输入处理器 - 统一处理文件路径和二进制数据"""

    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self._temp_files = []  # 跟踪临时文件以便清理

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_temp_files()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_temp_files()

    def prepare_audio_input(self,
                            audio_input: Union[str, bytes, BinaryIO],
                            target_sr: int = 16000) -> Tuple[str, bool]:
        """
        准备音频输入，统一转换为文件路径

        Args:
            audio_input: 音频输入（文件路径、字节数据或文件对象）
            target_sr: 目标采样率

        Returns:
            Tuple[str, bool]: (音频文件路径, 是否为临时文件)
        """
        if isinstance(audio_input, str):
            # 文件路径
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            return audio_input, False

        elif isinstance(audio_input, bytes):
            # 字节数据
            return self._save_bytes_to_temp_file(audio_input), True

        elif hasattr(audio_input, 'read'):
            # 文件对象
            return self._save_file_object_to_temp_file(audio_input), True

        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

    def load_audio_data(self,
                        audio_input: Union[str, bytes, BinaryIO],
                        target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        直接加载音频数据，不创建临时文件

        Args:
            audio_input: 音频输入
            target_sr: 目标采样率

        Returns:
            Tuple[np.ndarray, int]: (音频数据, 采样率)
        """
        if isinstance(audio_input, str):
            # 文件路径
            wav, sr = librosa.load(audio_input, sr=target_sr, mono=True)
            return wav, sr

        elif isinstance(audio_input, bytes):
            # 字节数据
            with io.BytesIO(audio_input) as buffer:
                wav, sr = sf.read(buffer)
                if sr != target_sr:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
                return wav, target_sr

        elif hasattr(audio_input, 'read'):
            # 文件对象
            audio_input.seek(0)  # 确保从开头读取
            wav, sr = sf.read(audio_input)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav, target_sr

        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

    def _save_bytes_to_temp_file(self, audio_bytes: bytes) -> str:
        """保存字节数据到临时文件"""
        temp_file = os.path.join(
            self.temp_dir,
            f"temp_audio_{uuid.uuid4().hex[:8]}.wav"
        )

        # 检测音频格式并保存
        try:
            # 尝试直接作为wav格式读取
            with io.BytesIO(audio_bytes) as buffer:
                wav, sr = sf.read(buffer)
                sf.write(temp_file, wav, sr)
        except Exception:
            # 如果失败，直接写入字节（假设是有效的音频格式）
            with open(temp_file, 'wb') as f:
                f.write(audio_bytes)

        self._temp_files.append(temp_file)
        return temp_file

    def _save_file_object_to_temp_file(self, file_obj: BinaryIO) -> str:
        """保存文件对象到临时文件"""
        temp_file = os.path.join(
            self.temp_dir,
            f"temp_audio_{uuid.uuid4().hex[:8]}.wav"
        )

        file_obj.seek(0)  # 确保从开头读取

        try:
            # 尝试作为音频文件读取并保存为wav
            wav, sr = sf.read(file_obj)
            sf.write(temp_file, wav, sr)
        except Exception:
            # 如果失败，直接复制字节
            file_obj.seek(0)
            with open(temp_file, 'wb') as f:
                f.write(file_obj.read())

        self._temp_files.append(temp_file)
        return temp_file

    def cleanup_temp_files(self):
        """清理临时文件"""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Warning: Failed to remove temp file {temp_file}: {e}")
        self._temp_files.clear()


import asyncio
import io
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, Union, BinaryIO, List

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from voice2text.tran.vector_base import VoicePrintRecord
from voice2text.tran.filesystem import ImprovedFileManager


class VectorEnhancedVoicePrintManager:
    """基于向量数据库的增强声纹管理器 - 支持多种音频输入格式"""

    def __init__(self,
                 speaker_model,  # SpeakerModel instance
                 file_manager: ImprovedFileManager,  # FileManager instance
                 vector_db,  # VectorDatabaseInterface instance
                 config,  # SpeakerConfig instance
                 target_sr: int = 16000):
        """
        初始化声纹管理器

        Args:
            speaker_model: 说话人模型实例
            file_manager: 文件管理器实例
            vector_db: 向量数据库接口实例
            config: 说话人配置
            target_sr: 目标采样率
        """
        self.speaker_model = speaker_model
        self.file_manager = file_manager
        self.vector_db = vector_db
        self.config = config
        self.target_sr = target_sr

        # 日志器
        self.logger = logging.getLogger(f"{__name__}.VectorEnhancedVoicePrintManager")

        # 内存缓存 - 用于快速访问
        self._speaker_cache = {}  # speaker_id -> averaged_embedding
        self._cache_dirty = True

    async def initialize(self):
        """初始化管理器 - 确保向量数据库连接和缓存加载"""
        # 确保向量数据库已连接
        if not await self.vector_db.collection_exists():
            await self.vector_db.create_collection()

        # 加载缓存
        await self._load_speaker_cache()
        self.logger.info("VectorEnhancedVoicePrintManager initialized successfully")

    async def _load_speaker_cache(self):
        """加载说话人缓存"""
        try:
            speakers = await self.vector_db.list_speakers()
            self._speaker_cache.clear()

            for speaker_id in speakers.keys():
                await self._update_speaker_cache(speaker_id)

            self._cache_dirty = False
            self.logger.info(f"Loaded cache for {len(speakers)} speakers")

        except Exception as e:
            self.logger.error(f"Error loading speaker cache: {e}")

    async def _update_speaker_cache(self, speaker_id: str):
        """更新特定说话人的缓存"""
        try:
            records = await self.vector_db.get_vectors_by_speaker(speaker_id)
            if records:
                # 计算平均嵌入向量
                embeddings = [record.embedding for record in records]
                avg_embedding = np.mean(embeddings, axis=0)
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                self._speaker_cache[speaker_id] = avg_embedding

        except Exception as e:
            self.logger.error(f"Error updating cache for speaker {speaker_id}: {e}")

    async def delete_speaker_audio_sample(self, speaker_id: str, file_id: str) -> bool:
        """
        删除特定说话人的某个音频样本

        Args:
            speaker_id: 说话人ID
            file_id: 音频文件ID

        Returns:
            bool: 删除是否成功
        """
        try:
            # 验证说话人是否存在
            speaker_records = await self.vector_db.get_vectors_by_speaker(speaker_id)
            if not speaker_records:
                self.logger.warning(f"Speaker {speaker_id} does not exist")
                return False

            # 查找指定的音频记录
            target_record = None
            for record in speaker_records:
                if record.id == file_id:
                    target_record = record
                    break

            if not target_record:
                self.logger.warning(f"Audio sample {file_id} not found for speaker {speaker_id}")
                return False

            # 删除向量数据库中的记录
            success = await self.vector_db.delete_vector(file_id)
            if not success:
                self.logger.error(f"Failed to delete vector record {file_id}")
                return False

            # 删除音频文件
            await self._delete_audio_file(file_id)

            # 更新缓存
            await self._update_speaker_cache(speaker_id)

            self.logger.info(f"Successfully deleted audio sample {file_id} for speaker {speaker_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting audio sample {file_id} for speaker {speaker_id}: {e}")
            return False

    async def register_voice(self,
                             person_name: str,
                             audio_input: Union[str, bytes, BinaryIO],
                             cleanup_temp: bool = True) -> Tuple[str, str]:
        """
        注册新声纹到向量数据库

        Args:
            person_name: 人名
            audio_input: 音频输入（文件路径、字节数据或文件对象）
            cleanup_temp: 是否自动清理临时文件

        Returns:
            Tuple[str, str]: (speaker_id, sample_id)
        """
        self.logger.info(f"Registering voice for: {person_name}")

        try:
            # 加载音频数据
            wav, sr = self._load_audio_data(audio_input, self.target_sr)
            wav = self._trim_audio(wav, sr, self.config.max_voiceprint_length)

            # 提取嵌入向量
            embedding = self.speaker_model.encode_audio(wav)

            # 检查现有样本数量
            existing_records = await self.vector_db.get_vectors_by_speaker(person_name)
            sample_num = len(existing_records) + 1

            # 如果超过最大样本数，删除最旧的样本
            if sample_num > self.config.max_samples_per_voiceprint:
                oldest_record = min(existing_records, key=lambda x: x.created_at)
                await self.vector_db.delete_vector(oldest_record.id)

                # 同时删除对应的音频文件
                await self._delete_audio_file(oldest_record.id)

                sample_num = self.config.max_samples_per_voiceprint

            # 保存音频文件
            audio_data = self._wav_to_bytes(wav, sr)
            filename = f"{person_name}_sample{sample_num}.wav"

            file_id = await self.file_manager.save_file(
                data=audio_data,
                filename=filename,
                category="voiceprint",
                metadata={
                    'speaker_id': person_name,
                    'sample_number': int(sample_num),
                    'audio_duration': float(len(wav) / sr)
                }
            )

            # 创建向量记录
            vector_record = VoicePrintRecord(
                id=file_id,
                speaker_id=person_name,
                embedding=embedding,
                sample_number=sample_num,
                audio_duration=len(wav) / sr,
                created_at=datetime.now(),
                metadata={
                    'filename': filename,
                    'audio_file_id': file_id
                }
            )

            # 插入到向量数据库
            success = await self.vector_db.insert_vector(vector_record)
            if not success:
                # 如果向量插入失败，删除音频文件
                await self.file_manager.delete_file(file_id)
                raise RuntimeError("Failed to insert vector into database")

            # 更新缓存
            await self._update_speaker_cache(person_name)

            sample_id = f"{person_name}+sample{sample_num}.wav"
            self.logger.info(f"Voice registration successful for {person_name}, sample ID: {sample_id}")

            return person_name, sample_id

        except Exception as e:
            self.logger.error(f"Voice registration failed for {person_name}: {e}")
            raise

    async def identify_speakers(self,
                                diarize_segments: pd.DataFrame,
                                audio_input: Union[str, bytes, BinaryIO],
                                threshold: float = 0.4,
                                auto_register: bool = True) -> Tuple[Dict, Dict, Dict]:
        """
        基于向量数据库识别说话人

        Args:
            diarize_segments: 分离的语音段落
            audio_input: 音频输入（文件路径、字节数据或文件对象）
            threshold: 识别阈值
            auto_register: 是否自动注册未知说话人

        Returns:
            Tuple[speaker_identities, auto_registered_speakers, voiceprint_audio_samples]
        """
        self.logger.info("Identifying speakers using vector database...")

        # 加载音频数据
        wav, sr = self._load_audio_data(audio_input, self.target_sr)

        speaker_identities = {}
        auto_registered_speakers = {}
        voiceprint_audio_samples = {}

        for speaker in diarize_segments['speaker'].unique():
            speaker_segments = diarize_segments[diarize_segments['speaker'] == speaker]

            # 收集说话人音频片段
            speaker_wavs = []
            for _, segment in speaker_segments.iterrows():
                start_sample = int(segment['start'] * sr)
                end_sample = min(int(segment['end'] * sr), len(wav))

                if end_sample > start_sample:
                    segment_audio = wav[start_sample:end_sample]
                    speaker_wavs.append(segment_audio)

            if speaker_wavs:
                combined_wav = np.concatenate(speaker_wavs)
                audio_duration = len(combined_wav) / sr

                if audio_duration > self.config.max_voiceprint_length:
                    combined_wav = self._trim_audio(combined_wav, sr, self.config.max_voiceprint_length)
                    audio_duration = self.config.max_voiceprint_length

                # 提取嵌入向量
                embedding = self.speaker_model.encode_audio(combined_wav)

                # 在向量数据库中搜索相似声纹
                best_match, best_score = await self._find_best_match_in_db(embedding, threshold)

                if best_match and best_score >= threshold:
                    speaker_identities[speaker] = best_match
                    self.logger.info(f"Speaker {speaker} identified as: {best_match} (similarity: {best_score:.4f})")
                else:
                    if auto_register:
                        new_speaker_id, sample_id = await self._register_unknown_speaker(
                            embedding, combined_wav, sr, speaker
                        )
                        speaker_identities[speaker] = new_speaker_id
                        auto_registered_speakers[new_speaker_id] = {
                            'original_id': speaker,
                            'audio_length': audio_duration,
                            'sample_id': sample_id
                        }
                        voiceprint_audio_samples[new_speaker_id] = sample_id
                        self.logger.info(f"Unknown speaker {speaker} auto-registered as: {new_speaker_id}")
                    else:
                        speaker_identities[speaker] = f"未知:{speaker}"

        return speaker_identities, auto_registered_speakers, voiceprint_audio_samples

    async def _find_best_match_in_db(self, embedding: np.ndarray, threshold: float) -> Tuple[Optional[str], float]:
        """在向量数据库中找到最佳匹配"""
        try:
            # 搜索最相似的向量
            similar_vectors = await self.vector_db.search_similar_vectors(
                query_vector=embedding,
                top_k=10,  # 获取前10个最相似的
                threshold=threshold
            )

            if not similar_vectors:
                return None, 0.0

            # 按说话人分组并计算平均相似度
            speaker_scores = {}
            for record, similarity in similar_vectors:
                speaker_id = record.speaker_id
                if speaker_id not in speaker_scores:
                    speaker_scores[speaker_id] = []
                speaker_scores[speaker_id].append(similarity)

            # 计算每个说话人的平均相似度
            best_speaker = None
            best_score = 0.0

            for speaker_id, scores in speaker_scores.items():
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_speaker = speaker_id

            return best_speaker, best_score

        except Exception as e:
            self.logger.error(f"Error finding best match in database: {e}")
            return None, 0.0

    async def _register_unknown_speaker(self,
                                        embedding: np.ndarray,
                                        audio_data: np.ndarray,
                                        sr: int,
                                        original_speaker: str) -> Tuple[str, str]:
        """注册未知说话人到向量数据库"""
        speaker_id = f"Speaker_{uuid.uuid4().hex[:8]}"

        # 保存音频样本
        audio_bytes = self._wav_to_bytes(audio_data, sr)
        filename = f"{speaker_id}_sample1.wav"

        file_id = await self.file_manager.save_file(
            data=audio_bytes,
            filename=filename,
            category="voiceprint",
            metadata={
                'speaker_id': speaker_id,
                'sample_number': 1,
                'audio_duration': float(len(audio_data) / sr),
                'original_speaker': str(original_speaker)
            }
        )

        # 创建向量记录
        vector_record = VoicePrintRecord(
            id=file_id,
            speaker_id=speaker_id,
            embedding=embedding,
            sample_number=1,
            audio_duration=len(audio_data) / sr,
            created_at=datetime.now(),
            metadata={
                'filename': filename,
                'audio_file_id': file_id,
                'original_speaker': str(original_speaker)
            }
        )

        # 插入到向量数据库
        success = await self.vector_db.insert_vector(vector_record)
        if not success:
            # 如果向量插入失败，删除音频文件
            await self.file_manager.delete_file(file_id)
            raise RuntimeError("Failed to insert vector into database")

        # 更新缓存
        await self._update_speaker_cache(speaker_id)

        sample_id = f"{speaker_id}+sample1.wav"
        return speaker_id, sample_id

    async def list_registered_voices(self, include_unnamed: bool = True) -> Dict:
        """列出所有注册的声纹"""
        try:
            speakers_metadata = await self.vector_db.list_speakers()
            named_voices = {}
            unnamed_voices = {}

            for speaker_id in speakers_metadata.keys():
                if speaker_id.startswith("Speaker_"):
                    unnamed_voices[speaker_id] = speakers_metadata[speaker_id]
                else:
                    named_voices[speaker_id] = speakers_metadata[speaker_id]

            # 记录结果
            if named_voices:
                self.logger.info(f"Current registered named voiceprints ({len(named_voices)}):")
                for i, (name, samples) in enumerate(named_voices.items(), 1):
                    self.logger.info(f"  {i}. {name} ({len(samples)} samples)")

            if include_unnamed and unnamed_voices:
                self.logger.info(f"Current registered unnamed voiceprints ({len(unnamed_voices)}):")
                for i, (speaker_id, samples) in enumerate(unnamed_voices.items(), 1):
                    self.logger.info(f"  {i}. {speaker_id} ({len(samples)} samples)")

            return {
                "named_voice_prints": named_voices,
                "unnamed_voice_prints": unnamed_voices if include_unnamed else {}
            }

        except Exception as e:
            self.logger.error(f"Error listing registered voices: {e}")
            return {"named_voice_prints": {}, "unnamed_voice_prints": {}}

    async def rename_voice_print(self, old_speaker_id: str, new_name: str) -> bool:
        """重命名声纹"""
        try:
            # 检查新名称是否已存在
            existing_speakers = await self.vector_db.list_speakers()
            if new_name in existing_speakers.keys():
                self.logger.warning(f"Target name {new_name} already exists")
                return False

            # 获取旧说话人的所有记录
            records = await self.vector_db.get_vectors_by_speaker(old_speaker_id)
            if not records:
                self.logger.warning(f"Speaker {old_speaker_id} does not exist")
                return False

            # 更新所有记录的speaker_id
            for record in records:
                # 更新向量记录
                await self.vector_db.update_vector(
                    record.id,
                    {'speaker_id': new_name}
                )

                # 更新文件元数据
                await self.file_manager.metadata_storage.update(
                    record.id,
                    {'metadata.speaker_id': new_name}
                )

            # 更新缓存
            if old_speaker_id in self._speaker_cache:
                self._speaker_cache[new_name] = self._speaker_cache[old_speaker_id]
                del self._speaker_cache[old_speaker_id]

            self.logger.info(f"Voiceprint {old_speaker_id} renamed to {new_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error renaming voice print: {e}")
            return False

    async def delete_speaker(self, speaker_id: str) -> bool:
        """删除说话人的所有声纹数据"""
        try:
            # 删除向量数据库中的记录
            deleted_count = await self.vector_db.delete_vectors_by_speaker(speaker_id)

            # 删除音频文件
            voiceprint_files = await self._get_voiceprint_files_async()
            for file_info in voiceprint_files:
                metadata = file_info.get('metadata', {})
                if metadata.get('speaker_id') == speaker_id:
                    await self.file_manager.delete_file(file_info['file_id'])

            # 更新缓存
            if speaker_id in self._speaker_cache:
                del self._speaker_cache[speaker_id]

            self.logger.info(f"Deleted {deleted_count} voiceprint records for speaker {speaker_id}")
            return deleted_count > 0

        except Exception as e:
            self.logger.error(f"Error deleting speaker: {e}")
            return False

    async def clear_voice_prints(self):
        """清空所有声纹数据"""
        try:
            # 清空向量数据库
            await self.vector_db.clear_collection()

            # 删除所有声纹文件
            voiceprint_files = await self._get_voiceprint_files_async()
            for file_info in voiceprint_files:
                await self.file_manager.delete_file(file_info['file_id'])

            # 清空缓存
            self._speaker_cache.clear()

            self.logger.info("Cleared all voiceprint data")

        except Exception as e:
            self.logger.error(f"Error clearing voice prints: {e}")

    async def get_speaker_statistics(self) -> Dict[str, Any]:
        """获取说话人统计信息"""
        try:
            speakers = await self.vector_db.list_speakers()
            stats = {
                'total_speakers': len(speakers),
                'named_speakers': 0,
                'unnamed_speakers': 0,
                'total_samples': 0,
                'speakers_detail': {}
            }

            for speaker_id, meta in speakers.items():

                sample_count = len(meta)
                stats['total_samples'] += sample_count

                if speaker_id.startswith("Speaker_"):
                    stats['unnamed_speakers'] += 1
                else:
                    stats['named_speakers'] += 1

                stats['speakers_detail'][speaker_id] = {
                    'sample_count': sample_count,
                    'latest_update': max(r.created_at for r in meta) if meta else None
                }

            return stats

        except Exception as e:
            self.logger.error(f"Error getting speaker statistics: {e}")
            return {}

    async def search_similar_voices(self,
                                    audio_input: Union[str, bytes, BinaryIO],
                                    top_k: int = 10,
                                    threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        搜索相似的声纹

        Args:
            audio_input: 音频输入
            top_k: 返回最相似的前k个
            threshold: 相似度阈值

        Returns:
            相似声纹列表 [(speaker_id, similarity), ...]
        """
        try:
            # 加载音频并提取特征
            wav, sr = self._load_audio_data(audio_input, self.target_sr)

            # 裁剪音频长度
            max_length = self.config.max_voiceprint_length
            if len(wav) > max_length * sr:
                wav = wav[:int(max_length * sr)]

            # 提取嵌入向量
            embedding = self.speaker_model.encode_audio(wav)

            # 在向量数据库中搜索
            similar_vectors = await self.vector_db.search_similar_vectors(
                query_vector=embedding,
                top_k=top_k,
                threshold=threshold
            )

            # 整理结果
            results = [(record.speaker_id, similarity) for record, similarity in similar_vectors]

            self.logger.info(f"Found {len(results)} similar voices")
            return results

        except Exception as e:
            self.logger.error(f"Error searching similar voices: {e}")
            return []


    # ==================== 私有辅助方法 ====================

    def _load_audio_data(self,
                         audio_input: Union[str, bytes, BinaryIO],
                         target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        加载音频数据

        Args:
            audio_input: 音频输入
            target_sr: 目标采样率

        Returns:
            Tuple[np.ndarray, int]: (音频数据, 采样率)
        """
        if isinstance(audio_input, str):
            # 文件路径
            wav, sr = librosa.load(audio_input, sr=target_sr, mono=True)
            return wav, sr

        elif isinstance(audio_input, bytes):
            # 字节数据
            with io.BytesIO(audio_input) as buffer:
                wav, sr = sf.read(buffer)
                if sr != target_sr:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
                return wav, target_sr

        elif hasattr(audio_input, 'read'):
            # 文件对象
            audio_input.seek(0)  # 确保从开头读取
            wav, sr = sf.read(audio_input)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav, target_sr

        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

    async def _delete_audio_file(self, file_id: str):
        """删除音频文件"""
        try:
            await self.file_manager.delete_file(file_id)
        except Exception as e:
            self.logger.error(f"Error deleting audio file {file_id}: {e}")

    async def _get_voiceprint_files_async(self) -> List[Dict]:
        """异步获取声纹文件列表"""
        # 这里可能需要根据实际的文件管理器实现调整
        try:
            # 如果file_manager有异步方法
            if hasattr(self.file_manager, 'list_files_async'):
                return await self.file_manager.list_files_async(category="voiceprint")
            else:
                # 使用同步方法（在线程池中执行）
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    self.file_manager.list_files,
                    "voiceprint"
                )
        except Exception as e:
            self.logger.error(f"Error getting voiceprint files: {e}")
            return []

    def _trim_audio(self, wav_data: np.ndarray, sr: int, max_length_sec: int) -> np.ndarray:
        """裁剪音频到指定长度"""
        max_samples = int(max_length_sec * sr)
        if len(wav_data) > max_samples:
            mid_point = len(wav_data) // 2
            half_length = max_samples // 2
            start_idx = max(0, mid_point - half_length)
            end_idx = min(len(wav_data), mid_point + half_length)
            return wav_data[start_idx:end_idx]
        return wav_data

    def _wav_to_bytes(self, wav_data: np.ndarray, sr: int) -> bytes:
        """将音频数据转换为字节"""
        buffer = io.BytesIO()
        sf.write(buffer, wav_data, sr, format='WAV')
        return buffer.getvalue()

    # ==================== 上下文管理器支持 ====================

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # 如果需要，可以在这里进行清理工作
        pass
# ============================================================================
# 工厂函数 - 创建向量增强的声纹管理器
# ============================================================================

async def create_vector_voiceprint_manager(
        speaker_model,
        file_manager,
        config,
        vector_db_config: Optional[Dict[str, Any]] = None,
        db_type: VectorDBType = VectorDBType.CHROMADB
) -> VectorEnhancedVoicePrintManager:
    """
    创建基于向量数据库的声纹管理器

    Args:
        speaker_model: 说话人模型实例
        file_manager: 文件管理器实例
        config: 说话人配置
        vector_db_config: 向量数据库配置
        db_type: 数据库类型

    Returns:
        VectorEnhancedVoicePrintManager实例
    """

    # 默认配置
    if vector_db_config is None:
        vector_db_config = {}

    # 创建向量数据库配置
    if db_type == VectorDBType.CHROMADB:
        db_config = VectorDBConfigFactory.create_chromadb_config(
            **vector_db_config
        )
    elif db_type == VectorDBType.MILVUS:
        db_config = VectorDBConfigFactory.create_milvus_config(**vector_db_config)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

    # 创建向量数据库实例
    vector_db = VectorDatabaseFactory.create_database(db_type, db_config)

    # 连接数据库
    await vector_db.connect()

    # 创建声纹管理器
    manager = VectorEnhancedVoicePrintManager(
        speaker_model=speaker_model,
        file_manager=file_manager,
        vector_db=vector_db,
        config=config
    )

    await manager.initialize()
    return manager


# ============================================================================
# 使用示例
# ============================================================================

async def example_usage():
    """使用示例"""
    from voice2text.tran.vector_base import VectorDBType

    # 假设已有的组件（实际使用时需要创建真实的实例）
    speaker_model = None  # SpeakerModel instance
    file_manager = None  # FileManager instance
    config = None # speakerConfig instance

    # 创建向量数据库配置
    vector_db_config = {
        'persist_directory': './test_voice_vectors',
        'collection_name': 'test_voice_prints'
    }

    try:
        # 创建向量增强的声纹管理器
        manager = await create_vector_voiceprint_manager(
            speaker_model=speaker_model,
            file_manager=file_manager,
            config=config,
            vector_db_config=vector_db_config,
            db_type=VectorDBType.CHROMADB
        )

        async with manager:
            # 注册声纹
            speaker_id, sample_id = await manager.register_voice(
                "刘星",
                "../../data/sample/刘星.mp3"
            )
            print(f"注册成功: {speaker_id}, {sample_id}")
            speaker_id, sample_id = await manager.register_voice(
                "刘梅",
                "../../data/sample/刘梅.mp3"
            )
            print(f"注册成功: {speaker_id}, {sample_id}")
            # 列出声纹
            voices = await manager.list_registered_voices()
            print("注册的声纹:", voices)

            # 获取统计信息
            stats = await manager.get_speaker_statistics()
            print("统计信息:", stats)

    except Exception as e:
        print(f"示例运行出错: {e}")


if __name__ == "__main__":
    asyncio.run(example_usage())