# ============================================================================
# 异步任务管理模块 - 单机协程并发优化版本
# ============================================================================

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
from typing import Dict, Any, Optional, Callable, Awaitable, Union, BinaryIO

from voice2text.tran.filesystem import ImprovedFileManager, StorageConfig
from voice2text.tran.schema.dto import VoicePrintInfo
from voice2text.tran.schema.prints import SampleInfo
from voice2text.tran.vector_base import VectorDBConfig


class ASRModelType(Enum):
    """ASR模型类型枚举"""
    FUNASR = "funasr"
    WHISPER = "whisper"
    WAVE2VEC = "wave2vec"
    CUSTOM = "custom"


class SpeakerModelType(Enum):
    """说话人识别模型类型枚举"""
    ECAPA_TDNN = "ecapa_tdnn"
    XVECTOR = "xvector"
    CUSTOM = "custom"

@dataclass
class ModelConfig:
    """模型配置基类"""
    model_type: str
    model_name: str
    model_revision: Optional[str] = None
    device: str = "cpu"
    cache_dir: Optional[str] = None
    model_params: Dict[str, Any] = field(default_factory=dict)


class ASRModel(ABC):
    """ASR模型抽象基类"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._model = None

    @abstractmethod
    def load_model(self) -> None:
        """加载模型"""
        pass

    @abstractmethod
    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """转写音频"""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        pass

    def unload_model(self) -> None:
        """卸载模型"""
        self._model = None

@dataclass
class SpeakerConfig(ModelConfig):
    """说话人识别模型配置"""
    threshold: float = 0.4
    max_voiceprint_length: int = 30
    max_samples_per_voiceprint: int = 5

@dataclass
class FunASRConfig(ModelConfig):
    """FunASR模型配置"""
    vad_model: str = "fsmn-vad"
    vad_model_revision: str = "v2.0.4"
    punc_model: str = "ct-punc"
    punc_model_revision: str = "v2.0.4"
    spk_model: str = "cam++"
    spk_model_revision: str = "v2.0.2"

class FunASRModel(ASRModel):
    """FunASR模型实现"""

    def __init__(self, config: FunASRConfig):
        super().__init__(config)
        self.funasr_config = config

    def load_model(self) -> None:
        """加载FunASR模型"""
        try:
            from funasr import AutoModel

            self._model = AutoModel(
                model=self.funasr_config.model_name,
                model_revision=self.funasr_config.model_revision,
                vad_model=self.funasr_config.vad_model,
                vad_model_revision=self.funasr_config.vad_model_revision,
                punc_model=self.funasr_config.punc_model,
                punc_model_revision=self.funasr_config.punc_model_revision,
                spk_model=self.funasr_config.spk_model,
                spk_model_revision=self.funasr_config.spk_model_revision,
                **self.funasr_config.model_params
            )
            print(f"Successfully loaded FunASR model: {self.funasr_config.model_name}")
        except Exception as e:
            print(f"Error loading FunASR model: {e}")
            raise

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """使用FunASR转写音频"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        batch_size_s = kwargs.get('batch_size_s', 300)
        hotword = kwargs.get('hotword', '')

        result = self._model.generate(
            input=audio_path,
            batch_size_s=batch_size_s,
            hotword=hotword
        )

        return {
            'text': result[0].get('text', ''),
            'sentence_info': result[0].get('sentence_info', []),
            'timestamp': result[0].get('timestamp', [])
        }

    def is_loaded(self) -> bool:
        return self._model is not None

class SpeakerModel(ABC):
    """说话人识别模型抽象基类"""

    def __init__(self, config: SpeakerConfig):
        self.config = config
        self._model = None

    @abstractmethod
    def load_model(self) -> None:
        """加载模型"""
        pass

    @abstractmethod
    def encode_audio(self, audio_data) -> Any:
        """编码音频为声纹向量"""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        pass


class EcapaTdnnModel(SpeakerModel):
    """ECAPA-TDNN说话人识别模型"""

    def load_model(self) -> None:
        """加载ECAPA-TDNN模型"""
        try:
            from speechbrain.inference import EncoderClassifier

            self._model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.config.device}
            )
            print("Successfully loaded ECAPA-TDNN speaker model")
        except Exception as e:
            print(f"Error loading ECAPA-TDNN model: {e}")
            raise

    def encode_audio(self, audio_data):
        """编码音频为声纹向量"""
        if not self.is_loaded():
            raise RuntimeError("Speaker model not loaded")

        import torch

        wav_tensor = torch.FloatTensor(audio_data).unsqueeze(0)
        with torch.no_grad():
            embedding = self._model.encode_batch(wav_tensor)
            embedding = embedding.squeeze().cpu().numpy()

        import numpy as np
        return embedding / np.linalg.norm(embedding)

    def is_loaded(self) -> bool:
        return self._model is not None


def preprocess_audio_in_memory(input_path, target_sample_rate=16000):
    """在内存中预处理音频数据"""
    import torchaudio
    import torch

    waveform, sr = torchaudio.load(input_path)

    # 重采样
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        waveform = resampler(waveform)

    # 转换为单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return {
        "waveform": waveform,
        "sample_rate": target_sample_rate
    }

@dataclass
class WhisperConfig(ModelConfig):
    """Whisper模型配置"""
    language: Optional[str] = None
    task: str = "transcribe"
    diarize_model_path: str = "pyannote/speaker-diarization-3.1"
    use_auth_token: Optional[str] = None


class WhisperModel(ASRModel):
    """Whisper模型实现"""

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.whisper_config = config
        self.diarize_model = None

    def load_model(self) -> None:
        """加载Whisper模型"""
        try:
            import whisper

            self._model = whisper.load_model(
                self.whisper_config.model_name
            ).to(self.config.device)

            # 加载说话人分离模型
            self._load_diarize_model()

            print(f"Successfully loaded Whisper model: {self.whisper_config.model_name}")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            raise

    def _load_diarize_model(self):
        """加载说话人分离模型"""
        try:
            from pyannote.audio import Pipeline
            import torch
            import os

            auth_token = self.whisper_config.use_auth_token or os.getenv("HF_TOKEN")
            self.diarize_model = Pipeline.from_pretrained(
                self.whisper_config.diarize_model_path,
                use_auth_token=auth_token
            )

            if self.config.device != "cpu":
                self.diarize_model.to(torch.device(self.config.device))

            print("Successfully loaded diarization model")
        except Exception as e:
            print(f"Error loading diarization model: {e}")
            self.diarize_model = None

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """使用Whisper转写音频"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        # 使用Whisper进行转写
        result = self._model.transcribe(audio_path)
        segments = result["segments"]

        # 执行说话人分离
        diarize_segments = None
        if self.diarize_model:
            try:
                diarize_segments = self.diarize_model(preprocess_audio_in_memory(audio_path))
            except Exception as e:
                print(f"Diarization failed: {e}")

        # 将结果转换为统一格式
        sentence_info = []
        if segments:
            for i, segment in enumerate(segments):
                # 分配说话人ID
                speaker_id = f"spk_{i}"
                if diarize_segments:
                    # 找到与当前segment时间重叠最多的diarization segment
                    seg_start = segment.get('start', 0)
                    seg_end = segment.get('end', 0)

                    best_match_speaker = None
                    max_overlap = 0

                    for diar_segment, _, speaker in diarize_segments.itertracks(yield_label=True):
                        diar_start = diar_segment.start
                        diar_end = diar_segment.end

                        overlap_start = max(seg_start, diar_start)
                        overlap_end = min(seg_end, diar_end)
                        overlap_duration = max(0, overlap_end - overlap_start)

                        if overlap_duration > max_overlap:
                            max_overlap = overlap_duration
                            best_match_speaker = speaker

                    if best_match_speaker:
                        speaker_id = best_match_speaker

                sentence_info.append({
                    'spk': speaker_id,
                    'start': int(segment.get('start', 0) * 1000),  # 转换为毫秒
                    'end': int(segment.get('end', 0) * 1000),  # 转换为毫秒
                    'text': segment.get('text', '').strip()
                })

        return {
            'text': result.get('text', ''),
            'sentence_info': sentence_info,
            'timestamp': [[0, int(segments[-1]['end'] * 1000)]] if segments else []
        }

    def is_loaded(self) -> bool:
        return self._model is not None


class ModelFactory:
    """模型工厂类"""

    _asr_models = {
        ASRModelType.FUNASR: FunASRModel,
        ASRModelType.WHISPER: WhisperModel,
    }

    _speaker_models = {
        SpeakerModelType.ECAPA_TDNN: EcapaTdnnModel,
    }

    @classmethod
    def create_asr_model(cls, model_type: ASRModelType, config: ModelConfig) -> ASRModel:
        """创建ASR模型"""
        if model_type not in cls._asr_models:
            raise ValueError(f"Unsupported ASR model type: {model_type}")

        model_class = cls._asr_models[model_type]
        return model_class(config)

    @classmethod
    def create_speaker_model(cls, model_type: SpeakerModelType, config: SpeakerConfig) -> SpeakerModel:
        """创建说话人识别模型"""
        if model_type not in cls._speaker_models:
            raise ValueError(f"Unsupported speaker model type: {model_type}")

        model_class = cls._speaker_models[model_type]
        return model_class(config)

    @classmethod
    def register_asr_model(cls, model_type: ASRModelType, model_class: type):
        """注册新的ASR模型"""
        cls._asr_models[model_type] = model_class

    @classmethod
    def register_speaker_model(cls, model_type: SpeakerModelType, model_class: type):
        """注册新的说话人识别模型"""
        cls._speaker_models[model_type] = model_class

class TranscriptionStrategy(Enum):
    """转写策略枚举"""
    AUTO_SELECT = "auto_select"  # 根据语言自动选择
    FUNASR_ONLY = "funasr_only"  # 仅使用FunASR
    WHISPER_ONLY = "whisper_only"  # 仅使用Whisper
    HYBRID = "hybrid"  # 混合策略


@dataclass
class STTServiceConfig:
    """集成向量数据库的服务配置"""
    # ASR配置
    asr_config: Any  # FunASRConfig



    # 说话人识别配置
    speaker_config: Any  # SpeakerConfig

    # 向量数据库配置
    vector_db_config: VectorDBConfig

    # 文件存储配置
    storage_config: StorageConfig

    # 转写策略配置 (新增)
    transcription_strategy: TranscriptionStrategy = TranscriptionStrategy.AUTO_SELECT

    # 语言策略映射 (新增)
    language_model_mapping: Dict[str, str] = field(default_factory=lambda: {
        "auto": "whisper",
        "zh-CN": "funasr",
        "en-US": "whisper",
        "ja": "whisper",
        "ko": "whisper"
    })
    # Whisper配置 (新增)
    whisper_config: Optional[WhisperConfig] = None

    # 音频处理配置
    target_sr: int = 16000
    batch_size_s: int = 300

    # 异步任务配置
    max_transcribe_concurrent: int = 2
    max_speaker_concurrent: int = 3
    task_timeout: float = 300.0
    enable_task_logging: bool = True

    # 分布式配置
    instance_id: Optional[str] = None
    cluster_config: Optional[Dict[str, Any]] = None


@dataclass
class WhisperConfig(ModelConfig):
    """Whisper模型配置"""
    language: Optional[str] = None
    task: str = "transcribe"
    diarize_model_path: str = "pyannote/speaker-diarization-3.1"
    use_auth_token: Optional[str] = None







class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AsyncTaskManager:
    """异步任务管理器 - 基于协程的单机并发优化"""

    def __init__(self,
                 max_concurrent_tasks: int = 3,
                 max_gpu_concurrent: int = 1,
                 max_io_concurrent: int = 5,
                 thread_pool_size: int = 4,
                 task_timeout: float = 300.0,
                 enable_logging: bool = True):
        """
        初始化异步任务管理器

        Args:
            max_concurrent_tasks: 最大并发任务数
            max_gpu_concurrent: GPU密集型任务最大并发数
            max_io_concurrent: IO密集型任务最大并发数
            thread_pool_size: 线程池大小（用于CPU密集型任务）
            task_timeout: 任务超时时间（秒）
            enable_logging: 是否启用日志
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_gpu_concurrent = max_gpu_concurrent
        self.max_io_concurrent = max_io_concurrent
        self.task_timeout = task_timeout

        # 任务存储
        self.tasks: Dict[str, TaskInfo] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}

        # 信号量控制并发
        self.general_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.gpu_semaphore = asyncio.Semaphore(max_gpu_concurrent)
        self.io_semaphore = asyncio.Semaphore(max_io_concurrent)

        # 线程池（用于CPU密集型或同步任务）
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)

        # 任务队列
        self.task_queue = asyncio.Queue()
        self.priority_queue = asyncio.PriorityQueue()

        # 管理状态
        self.is_running = False
        self._worker_tasks = []
        self._cleanup_task = None

        # 回调函数
        self.task_callbacks: Dict[str, Callable] = {}

        # 日志配置
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # 性能统计
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_duration': 0.0,
            'current_load': 0
        }

    async def start(self) -> None:
        """启动任务管理器"""
        if self.is_running:
            return

        self.is_running = True

        # 启动工作协程
        self._worker_tasks = [
            asyncio.create_task(self._worker_loop(i))
            for i in range(self.max_concurrent_tasks)
        ]

        # 启动清理任务
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        self.logger.info(f"AsyncTaskManager started with {self.max_concurrent_tasks} workers")

    async def stop(self) -> None:
        """停止任务管理器"""
        if not self.is_running:
            return

        self.is_running = False

        # 取消所有运行中的任务
        for task_id, task in self.running_tasks.items():
            if not task.done():
                task.cancel()
                self.tasks[task_id].status = TaskStatus.CANCELLED

        # 等待工作协程完成
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)

        # 停止清理任务
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # 关闭线程池
        self.thread_pool.shutdown(wait=True)

        self.logger.info("AsyncTaskManager stopped")

    async def submit_task(self,
                          task_func: Union[Callable, Callable[..., Awaitable]],
                          *args,
                          task_id: Optional[str] = None,
                          priority: int = 5,
                          task_type: str = "general",
                          timeout: Optional[float] = None,
                          callback: Optional[Callable] = None,
                          metadata: Optional[Dict] = None,
                          **kwargs) -> str:
        """
        提交任务

        Args:
            task_func: 任务函数（可以是同步或异步函数）
            *args: 位置参数
            task_id: 任务ID（可选，会自动生成）
            priority: 任务优先级（数字越小优先级越高）
            task_type: 任务类型（general, gpu, io）
            timeout: 任务超时时间
            callback: 任务完成回调函数
            metadata: 任务元数据
            **kwargs: 关键字参数

        Returns:
            任务ID
        """
        if task_id is None:
            task_id = f"task_{uuid.uuid4().hex[:8]}"

        if task_id in self.tasks:
            raise ValueError(f"Task {task_id} already exists")

        # 创建任务信息
        task_info = TaskInfo(
            task_id=task_id,
            metadata=metadata or {}
        )
        task_info.metadata.update({
            'task_type': task_type,
            'priority': priority,
            'timeout': timeout or self.task_timeout
        })

        self.tasks[task_id] = task_info

        if callback:
            self.task_callbacks[task_id] = callback

        # 包装任务
        task_wrapper = {
            'task_id': task_id,
            'func': task_func,
            'args': args,
            'kwargs': kwargs,
            'priority': priority,
            'task_type': task_type,
            'timeout': timeout or self.task_timeout
        }

        # 根据优先级选择队列
        if priority <= 2:  # 高优先级任务
            await self.priority_queue.put((priority, time.time(), task_wrapper))
        else:
            await self.task_queue.put(task_wrapper)

        self.stats['total_tasks'] += 1
        self.logger.info(f"Task {task_id} submitted (type: {task_type}, priority: {priority})")

        return task_id

    async def get_result(self,
                         task_id: str,
                         timeout: Optional[float] = None,
                         poll_interval: float = 0.1) -> Any:
        """
        获取任务结果

        Args:
            task_id: 任务ID
            timeout: 等待超时时间
            poll_interval: 轮询间隔

        Returns:
            任务结果

        Raises:
            TimeoutError: 等待超时
            RuntimeError: 任务失败
            KeyError: 任务不存在
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        start_time = time.time()

        while True:
            task_info = self.tasks[task_id]

            if task_info.status == TaskStatus.COMPLETED:
                return task_info.result
            elif task_info.status == TaskStatus.FAILED:
                raise task_info.error
            elif task_info.status == TaskStatus.CANCELLED:
                raise asyncio.CancelledError(f"Task {task_id} was cancelled")

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")

            await asyncio.sleep(poll_interval)

    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id not in self.tasks:
            return False

        task_info = self.tasks[task_id]

        if task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False

        # 如果任务正在运行，取消协程
        if task_id in self.running_tasks:
            self.running_tasks[task_id].cancel()

        task_info.status = TaskStatus.CANCELLED
        task_info.completed_at = time.time()

        self.logger.info(f"Task {task_id} cancelled")
        return True

    def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """获取任务状态"""
        return self.tasks.get(task_id)

    def list_tasks(self,
                   status_filter: Optional[TaskStatus] = None,
                   task_type_filter: Optional[str] = None) -> Dict[str, TaskInfo]:
        """列出任务"""
        filtered_tasks = {}

        for task_id, task_info in self.tasks.items():
            if status_filter and task_info.status != status_filter:
                continue
            if task_type_filter and task_info.metadata.get('task_type') != task_type_filter:
                continue
            filtered_tasks[task_id] = task_info

        return filtered_tasks

    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        running_count = len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING])
        self.stats['current_load'] = running_count / self.max_concurrent_tasks
        return self.stats.copy()

    async def _worker_loop(self, worker_id: int) -> None:
        """工作协程主循环"""
        self.logger.info(f"Worker {worker_id} started")

        while self.is_running:
            try:
                # 优先处理高优先级任务
                task_wrapper = None

                try:
                    # 尝试从优先级队列获取任务（非阻塞）
                    _, _, task_wrapper = self.priority_queue.get_nowait()
                except asyncio.QueueEmpty:
                    try:
                        # 从普通队列获取任务（带超时）
                        task_wrapper = await asyncio.wait_for(
                            self.task_queue.get(), timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        continue

                if task_wrapper:
                    await self._execute_task(worker_id, task_wrapper)

            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)

        self.logger.info(f"Worker {worker_id} stopped")

    async def _execute_task(self, worker_id: int, task_wrapper: Dict) -> None:
        """执行任务"""
        task_id = task_wrapper['task_id']
        task_info = self.tasks[task_id]
        task_type = task_wrapper['task_type']

        # 选择合适的信号量
        semaphore = self._get_semaphore(task_type)

        async with semaphore:
            try:
                # 更新任务状态
                task_info.status = TaskStatus.RUNNING
                task_info.started_at = time.time()

                self.logger.info(f"Worker {worker_id} executing task {task_id}")

                # 创建任务协程
                if asyncio.iscoroutinefunction(task_wrapper['func']):
                    # 异步函数
                    coro = task_wrapper['func'](*task_wrapper['args'], **task_wrapper['kwargs'])
                else:
                    # 同步函数，在线程池中执行
                    loop = asyncio.get_event_loop()
                    coro = loop.run_in_executor(
                        self.thread_pool,
                        lambda: task_wrapper['func'](*task_wrapper['args'], **task_wrapper['kwargs'])
                    )

                # 执行任务（带超时）
                timeout = task_wrapper['timeout']
                task = asyncio.create_task(coro)
                self.running_tasks[task_id] = task

                try:
                    result = await asyncio.wait_for(task, timeout=timeout)

                    # 任务完成
                    task_info.status = TaskStatus.COMPLETED
                    task_info.result = result
                    task_info.progress = 100.0

                    self.stats['completed_tasks'] += 1
                    self.logger.info(f"Task {task_id} completed successfully")

                except asyncio.TimeoutError:
                    task.cancel()
                    raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds")

                finally:
                    if task_id in self.running_tasks:
                        del self.running_tasks[task_id]

            except Exception as e:
                # 任务失败
                task_info.status = TaskStatus.FAILED
                task_info.error = e

                self.stats['failed_tasks'] += 1
                self.logger.error(f"Task {task_id} failed: {e}")

            finally:
                # 更新完成时间
                task_info.completed_at = time.time()

                # 更新平均持续时间
                if task_info.started_at:
                    duration = task_info.completed_at - task_info.started_at
                    total_completed = self.stats['completed_tasks'] + self.stats['failed_tasks']
                    if total_completed > 0:
                        self.stats['average_duration'] = (
                                (self.stats['average_duration'] * (total_completed - 1) + duration) /
                                total_completed
                        )

                # 执行回调
                if task_id in self.task_callbacks:
                    try:
                        callback = self.task_callbacks[task_id]
                        if asyncio.iscoroutinefunction(callback):
                            await callback(task_info)
                        else:
                            callback(task_info)
                    except Exception as e:
                        self.logger.error(f"Callback error for task {task_id}: {e}")

    def _get_semaphore(self, task_type: str) -> asyncio.Semaphore:
        """根据任务类型获取对应的信号量"""
        if task_type == "gpu":
            return self.gpu_semaphore
        elif task_type == "io":
            return self.io_semaphore
        else:
            return self.general_semaphore

    async def _cleanup_loop(self) -> None:
        """清理任务循环"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # 每分钟清理一次
                await self._cleanup_completed_tasks()
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")

    async def _cleanup_completed_tasks(self, max_keep: int = 1000) -> None:
        """清理已完成的任务"""
        completed_tasks = [
            (task_id, task_info) for task_id, task_info in self.tasks.items()
            if task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
        ]

        if len(completed_tasks) > max_keep:
            # 按完成时间排序，保留最新的任务
            completed_tasks.sort(key=lambda x: x[1].completed_at or 0)
            tasks_to_remove = completed_tasks[:-max_keep]

            for task_id, _ in tasks_to_remove:
                del self.tasks[task_id]
                if task_id in self.task_callbacks:
                    del self.task_callbacks[task_id]

            self.logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")

    # 上下文管理器支持
    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


# ============================================================================
# 针对语音转文字服务的专用任务管理器
# ============================================================================

class VoiceTaskManager(AsyncTaskManager):
    """专门针对语音转文字服务的任务管理器"""

    def __init__(self,
                 max_transcribe_concurrent: int = 2,
                 max_speaker_concurrent: int = 3,
                 **kwargs):
        """
        初始化语音任务管理器

        Args:
            max_transcribe_concurrent: 最大转写并发数
            max_speaker_concurrent: 最大说话人识别并发数
        """
        super().__init__(
            max_concurrent_tasks=max_transcribe_concurrent + max_speaker_concurrent + 2,
            max_gpu_concurrent=max_transcribe_concurrent,
            max_io_concurrent=max_speaker_concurrent + 2,
            **kwargs
        )

        # 专用信号量
        self.transcribe_semaphore = asyncio.Semaphore(max_transcribe_concurrent)
        self.speaker_semaphore = asyncio.Semaphore(max_speaker_concurrent)

    async def submit_transcribe_task(self,
                                     transcribe_func: Callable,
                                     *args,
                                     **kwargs) -> str:
        """提交转写任务"""
        return await self.submit_task(
            transcribe_func,
            *args,
            task_type="transcribe",
            priority=1,
            **kwargs
        )

    async def submit_speaker_task(self,
                                  speaker_func: Callable,
                                  *args,
                                  **kwargs) -> str:
        """提交说话人识别任务"""
        return await self.submit_task(
            speaker_func,
            *args,
            task_type="speaker",
            priority=2,
            **kwargs
        )

    def _get_semaphore(self, task_type: str) -> asyncio.Semaphore:
        """根据任务类型获取对应的信号量"""
        if task_type == "transcribe":
            return self.transcribe_semaphore
        elif task_type == "speaker":
            return self.speaker_semaphore
        else:
            return super()._get_semaphore(task_type)




import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from voice2text.tran.voiceprint_manager_v2 import (
    create_vector_voiceprint_manager, VectorEnhancedVoicePrintManager
)
from voice2text.tran.vector_base import VectorDBType





class STTAsyncVoice2TextService:
    """集成向量数据库的异步语音转文字服务 - 支持多种音频输入格式"""

    def __init__(self, config: STTServiceConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.STTAsyncVoice2TextServicee")

        # 初始化组件（在initialize中异步初始化）
        self.file_manager: ImprovedFileManager = None
        self.asr_model = None
        self.whisper_model = None  # 新增Whisper模型
        self.speaker_model = None
        self.voice_print_manager: VectorEnhancedVoicePrintManager = None
        self.task_manager = None

        # 存储配置
        self.storage_config = config.storage_config


    async def initialize(self):
        """异步初始化服务"""
        self.logger.info("Initializing STTAsyncVoice2TextService...")

        # 初始化文件管理器
        await self._initialize_file_manager()

        # 初始化模型
        self._initialize_models()

        # 初始化向量增强的声纹管理器
        await self._initialize_vector_voiceprint_manager()

        # 初始化任务管理器
        await self._initialize_task_manager()

        self.logger.info("STTAsyncVoice2TextService initialized successfully")

    async def _initialize_file_manager(self):
        """异步初始化文件管理器"""
        from voice2text.tran.filesystem import StorageFactory

        # 创建存储实例
        storage = StorageFactory.create_storage(self.storage_config)

        # 创建文件管理器
        self.file_manager = ImprovedFileManager(storage)
        await self.file_manager.initialize()

        self.logger.info("File manager initialized successfully")

    def _initialize_models(self):
        """初始化ASR和说话人模型"""
        self.logger.info("Initializing models...")

        # 创建FunASR模型
        asr_type = ASRModelType(self.config.asr_config.model_type)
        self.asr_model = ModelFactory.create_asr_model(asr_type, self.config.asr_config)

        self.asr_model.load_model()

        # 创建Whisper模型 (如果配置了)
        if self.config.whisper_config:
            whisper_type = ASRModelType.WHISPER
            self.whisper_model = ModelFactory.create_asr_model(whisper_type, self.config.whisper_config)
            self.whisper_model.load_model()
            self.logger.info("Whisper model initialized")

        # 创建说话人识别模型
        speaker_type = SpeakerModelType(self.config.speaker_config.model_type)
        self.speaker_model = ModelFactory.create_speaker_model(speaker_type, self.config.speaker_config)
        self.speaker_model.load_model()

        self.logger.info("Models initialized successfully")
    async def _initialize_vector_voiceprint_manager(self):
        """初始化向量增强的声纹管理器"""
        self.logger.info("Initializing vector-enhanced voiceprint manager...")

        # 准备向量数据库配置
        vector_db_config = self.config.vector_db_config

        # 创建向量增强的声纹管理器
        self.voice_print_manager = await create_vector_voiceprint_manager(
            speaker_model=self.speaker_model,
            file_manager=self.file_manager,
            config=self.config.speaker_config,
            vector_db_config=vector_db_config
        )

        self.logger.info("Vector-enhanced voiceprint manager initialized")

    async def _initialize_task_manager(self):
        """初始化任务管理器"""

        self.task_manager = VoiceTaskManager(
            max_transcribe_concurrent=self.config.max_transcribe_concurrent,
            max_speaker_concurrent=self.config.max_speaker_concurrent,
            task_timeout=self.config.task_timeout,
            enable_logging=self.config.enable_task_logging
        )

        self.logger.info("Task manager initialized")

    async def start(self):
        """启动服务"""
        await self.initialize()
        await self.task_manager.start()
        self.logger.info("STTAsyncVoice2TextService started")

    async def stop(self):
        """停止服务"""
        self.logger.info("Stopping STTAsyncVoice2TextService...")

        if self.task_manager:
            await self.task_manager.stop()

        # 如果声纹管理器有向量数据库连接，断开连接
        if self.voice_print_manager and hasattr(self.voice_print_manager, 'vector_db'):
            await self.voice_print_manager.vector_db.disconnect()

        # 断开文件管理器连接
        if self.file_manager:
            await self.file_manager.storage.disconnect()

        # 清理音频处理器
        if self.audio_handler:
            self.audio_handler.cleanup_temp_files()

        self.logger.info("STTAsyncVoice2TextService stopped")

    def _select_asr_model(self, language: Optional[str] = None) -> ASRModel:
        """根据策略选择ASR模型"""
        if self.config.transcription_strategy == TranscriptionStrategy.FUNASR_ONLY:
            return self.asr_model
        elif self.config.transcription_strategy == TranscriptionStrategy.WHISPER_ONLY:
            if not self.whisper_model:
                raise RuntimeError("Whisper model not configured but WHISPER_ONLY strategy selected")
            return self.whisper_model
        elif self.config.transcription_strategy == TranscriptionStrategy.AUTO_SELECT:
            # 根据语言选择模型
            if language:
                preferred_model = self.config.language_model_mapping.get(language, "funasr")
                if preferred_model == "whisper" and self.whisper_model:
                    self.logger.info(f"Using Whisper for language: {language}")
                    return self.whisper_model
                else:
                    self.logger.info(f"Using FunASR for language: {language}")
                    return self.asr_model
            else:
                # 默认使用FunASR
                return self.asr_model
        else:
            # 默认使用Whisper
            return self.whisper_model

    # ==================== 核心转写方法 ====================

    async def transcribe_file_async(self,
                                    audio_input: Union[str, bytes, BinaryIO],
                                    threshold: float = None,
                                    auto_register_unknown: bool = True,
                                    priority: int = 5,
                                    language: Optional[str] = None,  # 新增语言参数
                                    **kwargs) -> str:
        """
        异步转写音频（支持多种输入格式和转写引擎）

        Args:
            audio_input: 音频输入（文件路径、字节数据或文件对象）
            threshold: 识别阈值
            auto_register_unknown: 是否自动注册未知说话人
            priority: 任务优先级
            language: 语言代码 ("auto", "zh", "en", etc.)
            **kwargs: 其他参数

        Returns:
            任务ID
        """

        async def transcribe_task():
            return await self._async_transcribe_file(
                audio_input,
                threshold,
                auto_register_unknown,
                language,  # 传递语言参数
                kwargs
            )

        # 生成输入描述用于元数据
        input_description = self._get_input_description(audio_input)

        task_id = await self.task_manager.submit_task(
            transcribe_task,
            task_type="transcribe",
            priority=priority,
            metadata={
                'audio_input': input_description,
                'threshold': threshold,
                'auto_register': auto_register_unknown,
                'language': language
            }
        )

        self.logger.info(f"Submitted transcribe task {task_id} for input: {input_description}, language: {language}")
        return task_id
    async def _async_transcribe_file(self,
                                     audio_input: Union[str, bytes, BinaryIO],
                                     threshold: float = None,
                                     auto_register_unknown: bool = True,
                                     language: Optional[str] = None,
                                     kwargs: Dict = None) -> Dict[str, Any]:
        """异步转写方法（支持多种输入格式和转写引擎）"""
        kwargs = kwargs or {}
        input_desc = self._get_input_description(audio_input)

        self.logger.info(f"Starting transcription for: {input_desc}, language: {language}")

        if threshold is None:
            threshold = self.config.speaker_config.threshold

        # 解析文件信息
        file_location = kwargs.get('file_location', '未知地点')
        file_date = kwargs.get('file_date', '未知日期')
        file_time = kwargs.get('file_time', '未知时间')

        try:
            # 选择ASR模型
            selected_model = self._select_asr_model(language)
            model_name = "Whisper" if selected_model == self.whisper_model else "FunASR"
            self.logger.info(f"Using {model_name} for transcription")

            # ASR转写
            asr_result = selected_model.transcribe(audio_input, **kwargs)

            # 处理转写结果
            transcript = ""
            auto_registered_speakers = {}
            voiceprint_audio_samples = {}
            audio_duration = 0.0

            if 'sentence_info' not in asr_result or not asr_result['sentence_info']:
                # 单说话人处理
                full_text = asr_result['text']
                timestamps = asr_result.get('timestamp', [])

                if timestamps:
                    audio_duration = timestamps[-1][1] / 1000.0
                else:
                    audio_duration = len(full_text) / 10  # 简单估算

                transcript = self._format_single_speaker_output(
                    full_text, audio_duration, file_location, file_date, file_time
                )

                self.logger.info(
                    f"Single speaker transcription completed with {model_name}, duration: {audio_duration:.2f}s")
            else:
                # 多说话人处理
                sentence_segments = asr_result['sentence_info']
                audio_duration = max(seg['end'] for seg in sentence_segments) / 1000.0

                # 转换为DataFrame
                segments_data = []
                for segment in sentence_segments:
                    segments_data.append({
                        'speaker': segment['spk'],
                        'start': segment['start'] / 1000.0,
                        'end': segment['end'] / 1000.0,
                        'text': segment['text']
                    })
                diarize_segments = pd.DataFrame(segments_data)

                self.logger.info(
                    f"Multi-speaker transcription with {model_name}, {len(diarize_segments)} segments, duration: {audio_duration:.2f}s")

                # 使用原始音频输入进行说话人识别
                speaker_mapping, auto_registered, voiceprint_samples = await self.voice_print_manager.identify_speakers(
                    diarize_segments,
                    audio_input,
                    threshold=threshold,
                    auto_register=auto_register_unknown
                )

                auto_registered_speakers = auto_registered
                voiceprint_audio_samples = voiceprint_samples

                # 格式化输出
                transcript = self._format_multi_speaker_output(
                    sentence_segments, speaker_mapping, file_location, file_date, file_time
                )

            # 保存转写结果
            output_file = self._save_transcript_from_input(audio_input, transcript)

            if auto_registered_speakers:
                self._log_auto_registered_info(auto_registered_speakers)

            self.logger.info(f"Transcription completed successfully with {model_name} for: {input_desc}")

            return {
                "transcript": transcript,
                "auto_registered_speakers": auto_registered_speakers,
                "voiceprint_audio_samples": voiceprint_audio_samples,
                "audio_duration": audio_duration,
                "output_file": output_file,
                "model_used": model_name  # 新增返回使用的模型信息
            }

        except Exception as e:
            self.logger.error(f"Transcription failed for {input_desc}: {e}")
            raise
    # ==================== 声纹管理方法 ====================

    async def register_voice_async(self,
                                   person_name: str,
                                   audio_input: Union[str, bytes, BinaryIO]) -> SampleInfo:
        """异步注册声纹（支持多种输入格式）"""


        return await self.voice_print_manager.register_voice(person_name, audio_input)



    async def list_registered_voices_async(self, include_unnamed: bool = True) -> List[VoicePrintInfo]:
        """异步列出注册的声纹"""
        return await self.voice_print_manager.list_registered_voices(include_unnamed)

    async def get_speaker_statistics_async(self) -> Dict[str, Any]:
        """异步获取说话人统计信息"""
        return await self.voice_print_manager.get_speaker_statistics()

    # ==================== 批量处理方法 ====================

    async def batch_register_voices(self, voice_data: List[Tuple[str, Union[str, bytes, BinaryIO]]]) -> List[str]:
        """
        批量注册声纹（支持混合输入类型）

        Args:
            voice_data: [(person_name, audio_input), ...] 列表

        Returns:
            任务ID列表
        """
        task_ids = []

        self.logger.info(f"Starting batch voice registration for {len(voice_data)} voices")

        for person_name, audio_input in voice_data:
            task_id = await self.register_voice_async(person_name, audio_input)
            task_ids.append(task_id)

        self.logger.info(f"Submitted {len(task_ids)} voice registration tasks")
        return task_ids

    async def batch_transcribe_files(self,
                                     audio_inputs: List[Union[str, bytes, BinaryIO]],
                                     **kwargs) -> List[str]:
        """
        批量异步转写音频（支持混合输入类型）

        Args:
            audio_inputs: 音频输入列表（可以是混合类型）
            **kwargs: 转写参数

        Returns:
            任务ID列表
        """
        task_ids = []

        self.logger.info(f"Starting batch transcription for {len(audio_inputs)} files")

        for i, audio_input in enumerate(audio_inputs):
            task_id = await self.transcribe_file_async(
                audio_input,
                priority=i + 1,  # 按顺序设置优先级
                **kwargs
            )
            task_ids.append(task_id)

        self.logger.info(f"Submitted {len(task_ids)} transcription tasks")
        return task_ids

    # ==================== 便利方法 ====================

    async def transcribe_from_bytes(self,
                                    audio_bytes: bytes,
                                    threshold: float = None,
                                    auto_register_unknown: bool = True,
                                    **kwargs) -> str:
        """从字节数据转写音频"""
        return await self.transcribe_file_async(
            audio_bytes,
            threshold=threshold,
            auto_register_unknown=auto_register_unknown,
            **kwargs
        )

    async def transcribe_from_file_object(self,
                                          file_obj: BinaryIO,
                                          threshold: float = None,
                                          auto_register_unknown: bool = True,
                                          **kwargs) -> str:
        """从文件对象转写音频"""
        return await self.transcribe_file_async(
            file_obj,
            threshold=threshold,
            auto_register_unknown=auto_register_unknown,
            **kwargs
        )

    async def register_voice_from_bytes(self, person_name: str, audio_bytes: bytes) -> str:
        """从字节数据注册声纹"""
        return await self.register_voice_async(person_name, audio_bytes)

    async def register_voice_from_file_object(self, person_name: str, file_obj: BinaryIO) -> str:
        """从文件对象注册声纹"""
        return await self.register_voice_async(person_name, file_obj)

    # ==================== 任务管理方法 ====================

    async def get_transcribe_result(self,
                                    task_id: str,
                                    timeout: Optional[float] = None) -> Dict[str, Any]:
        """获取异步转写结果"""
        return await self.task_manager.get_result(task_id, timeout)

    async def wait_for_all_results(self,
                                   task_ids: List[str],
                                   timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """等待所有任务完成并返回结果"""
        results = []

        self.logger.info(f"Waiting for {len(task_ids)} tasks to complete")

        for task_id in task_ids:
            try:
                result = await self.get_transcribe_result(task_id, timeout)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Task {task_id} failed: {e}")
                results.append({"error": str(e), "task_id": task_id})

        self.logger.info(f"Completed waiting for {len(results)} results")
        return results

    async def get_task_progress(self, task_id: str) -> Dict[str, Any]:
        """获取任务进度"""
        task_info = self.task_manager.get_task_status(task_id)
        if not task_info:
            return {"error": "Task not found"}

        return {
            "task_id": task_id,
            "status": task_info.status.value,
            "progress": task_info.progress,
            "created_at": task_info.created_at,
            "started_at": task_info.started_at,
            "completed_at": task_info.completed_at,
            "metadata": task_info.metadata
        }

    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        success = await self.task_manager.cancel_task(task_id)
        if success:
            self.logger.info(f"Task {task_id} cancelled successfully")
        else:
            self.logger.warning(f"Failed to cancel task {task_id}")
        return success

    def get_service_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        return self.task_manager.get_stats()

    def list_active_tasks(self):
        """列出活跃任务"""
        return self.task_manager.list_tasks(status_filter=TaskStatus.RUNNING)

    # ==================== 向量数据库管理方法 ====================


    async def restore_vector_database(self, backup_path: str) -> bool:
        """从备份恢复向量数据库"""
        try:
            self.logger.info(f"Starting vector database restore from: {backup_path}")

            # 读取备份数据
            import json
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)

            # 清空现有数据
            await self.voice_print_manager.vector_db.clear_collection()

            # 恢复数据
            from vector_base import VoicePrintRecord
            records = [VoicePrintRecord.from_dict(data) for data in backup_data]

            success = await self.voice_print_manager.vector_db.batch_insert_vectors(records)

            if success:
                # 重新加载缓存
                await self.voice_print_manager._load_speaker_cache()
                self.logger.info(
                    f"Vector database restore completed: {len(records)} records restored from {backup_path}")
                return True
            else:
                self.logger.error("Vector database restore failed during batch insert")
                return False

        except Exception as e:
            self.logger.error(f"Vector database restore failed: {e}")
            return False

    # ==================== 同步版本的便利方法 ====================

    def transcribe_sync(self,
                        audio_input: Union[str, bytes, BinaryIO],
                        timeout: float = 300.0,
                        **kwargs) -> Dict[str, Any]:
        """
        同步转写方法（阻塞直到完成）

        Args:
            audio_input: 音频输入
            timeout: 超时时间
            **kwargs: 转写参数

        Returns:
            转写结果
        """

        async def _sync_transcribe():
            task_id = await self.transcribe_file_async(audio_input, **kwargs)
            return await self.get_transcribe_result(task_id, timeout)

        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果已经在异步环境中，抛出异常
            raise RuntimeError("Cannot use sync method in async context. Use transcribe_file_async instead.")
        else:
            return loop.run_until_complete(_sync_transcribe())

    def register_voice_sync(self,
                            person_name: str,
                            audio_input: Union[str, bytes, BinaryIO],
                            timeout: float = 60.0) -> Tuple[str, str]:
        """
        同步注册声纹方法（阻塞直到完成）

        Args:
            person_name: 人名
            audio_input: 音频输入
            timeout: 超时时间

        Returns:
            (speaker_id, sample_id)
        """

        async def _sync_register():
            task_id = await self.register_voice_async(person_name, audio_input)
            return await self.task_manager.get_result(task_id, timeout)

        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("Cannot use sync method in async context. Use register_voice_async instead.")
        else:
            return loop.run_until_complete(_sync_register())

    # ==================== 私有辅助方法 ====================

    def set_transcription_strategy(self, strategy: TranscriptionStrategy):
        """设置转写策略"""
        self.config.transcription_strategy = strategy
        self.logger.info(f"Transcription strategy changed to: {strategy.value}")

    def update_language_mapping(self, language: str, model: str):
        """更新语言-模型映射"""
        if model not in ["whisper", "funasr"]:
            raise ValueError("Model must be 'whisper' or 'funasr'")

        self.config.language_model_mapping[language] = model
        self.logger.info(f"Language mapping updated: {language} -> {model}")

    def _get_input_description(self, audio_input: Union[str, bytes, BinaryIO]) -> str:
        """获取音频输入的描述"""
        if isinstance(audio_input, str):
            return f"file_path: {audio_input}"
        elif isinstance(audio_input, bytes):
            return f"bytes_data: {len(audio_input)} bytes"
        elif hasattr(audio_input, 'read'):
            name = getattr(audio_input, 'name', 'unknown')
            return f"file_object: {name}"
        else:
            return f"unknown_type: {type(audio_input)}"

    def _save_transcript_from_input(self, audio_input: Union[str, bytes, BinaryIO], transcript: str) -> str:
        """根据音频输入类型保存转写结果"""
        if isinstance(audio_input, str):
            # 文件路径，使用原文件名
            output_file = audio_input.rsplit(".", 1)[0] + "_transcript.txt"
        else:
            # 二进制数据或文件对象，生成新文件名
            output_file = f"transcript_{uuid.uuid4().hex[:8]}.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcript)

        self.logger.info(f"Transcript saved to: {output_file}")
        return output_file

    def _format_single_speaker_output(self, text: str, duration: float,
                                      location: Optional[str], date: Optional[str],
                                      time: Optional[str]) -> str:
        """格式化单说话人输出"""
        start_time = "00:00:00"
        end_time = self._format_time(duration)

        if location and date and time:
            try:
                file_datetime = datetime.strptime(f"{date} {time}", "%Y/%m/%d %H:%M:%S")
                end_datetime = file_datetime + timedelta(seconds=duration)
                new_start_time = file_datetime.strftime("%Y/%m/%d-%H:%M:%S")
                new_end_time = end_datetime.strftime("%Y/%m/%d-%H:%M:%S")
                return f"[{location}][{new_start_time}-{new_end_time}] [未知说话人] {text}"
            except ValueError as e:
                self.logger.warning(f"Failed to parse datetime {date} {time}: {e}")
                return f"[{start_time}-{end_time}] [未知说话人] {text}"
        else:
            return f"[{start_time}-{end_time}] [未知说话人] {text}"

    def _format_multi_speaker_output(self, sentence_segments: List[Dict],
                                     speaker_mapping: Dict[str, str],
                                     location: Optional[str], date: Optional[str],
                                     time: Optional[str]) -> str:
        """格式化多说话人输出"""
        formatted_segments = []

        for segment in sentence_segments:
            start_time = self._format_time(segment['start'] / 1000.0)
            end_time = self._format_time(segment['end'] / 1000.0)
            spk_id = segment['spk']
            text = segment['text']

            speaker_name = speaker_mapping.get(spk_id, f"Speaker_{spk_id}")

            if location and date and time:
                formatted_segment = self._format_output_segment(
                    start_time, end_time, speaker_name, text, location, date, time
                )
            else:
                formatted_segment = f"[{start_time}-{end_time}] [{speaker_name}] {text}"

            formatted_segments.append(formatted_segment)

        return " \n ".join(formatted_segments)

    def _format_output_segment(self, start_time: str, end_time: str, speaker_name: str,
                               text: str, file_location: str, file_date: str, file_time: str) -> str:
        """格式化输出段落"""
        try:
            start_sec = self._parse_time(start_time)
            end_sec = self._parse_time(end_time)

            file_datetime = datetime.strptime(f"{file_date} {file_time}", "%Y/%m/%d %H:%M:%S")
            start_datetime = file_datetime + timedelta(seconds=start_sec)
            end_datetime = file_datetime + timedelta(seconds=end_sec)

            new_start_time = start_datetime.strftime("%Y/%m/%d-%H:%M:%S")
            new_end_time = end_datetime.strftime("%Y/%m/%d-%H:%M:%S")

            return f"[{file_location}][{new_start_time}-{new_end_time}] [{speaker_name}] {text}"
        except Exception as e:
            self.logger.warning(f"Failed to format segment with datetime: {e}")
            return f"[{start_time}-{end_time}] [{speaker_name}] {text}"

    def _log_auto_registered_info(self, auto_registered_speakers: Dict):
        """记录自动注册信息"""
        self.logger.info(f"Auto-registered {len(auto_registered_speakers)} unknown speakers:")
        for speaker_id, info in auto_registered_speakers.items():
            self.logger.info(f"  - Speaker ID: {speaker_id} (original: {info['original_id']}, "
                             f"duration: {info['audio_length']:.2f}s)")
            if 'sample_id' in info:
                self.logger.info(f"    Sample ID: {info['sample_id']}")

        if auto_registered_speakers:
            first_speaker = list(auto_registered_speakers.keys())[0]
            self.logger.info(
                f"You can rename speakers using: await service.rename_voice_print_async('{first_speaker}', 'new_name')")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化时间为HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    @staticmethod
    def _parse_time(time_str: str) -> float:
        """解析时间字符串为秒数"""
        parts = time_str.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    # ==================== 上下文管理器支持 ====================

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

# ============================================================================
# 配置工厂扩展
# ============================================================================

class STTConfigFactory:
    """服务配置工厂"""

    @staticmethod
    def create_funasr_config(
            model_name: str = "paraformer-zh",
            model_revision: str = "v2.0.4",
            device: str = "cpu"
    ) -> FunASRConfig:
        """创建FunASR配置"""
        return FunASRConfig(
            model_type=ASRModelType.FUNASR.value,
            model_name=model_name,
            model_revision=model_revision,
            device=device
        )

    @staticmethod
    def create_whisper_config(
            model_name: str = "large-v3-turbo",
            device: str = "cpu",
            language: Optional[str] = None,
            diarize_model_path: str = "pyannote/speaker-diarization-3.1",
            use_auth_token: Optional[str] = None
    ) -> WhisperConfig:
        """创建Whisper配置"""
        return WhisperConfig(
            model_type=ASRModelType.WHISPER.value,
            model_name=model_name,
            device=device,
            language=language,
            diarize_model_path=diarize_model_path,
            use_auth_token=use_auth_token
        )

    @staticmethod
    def create_speaker_config(
            model_type: SpeakerModelType = SpeakerModelType.ECAPA_TDNN,
            device: str = "cpu",
            threshold: float = 0.4
    ) -> SpeakerConfig:
        """创建说话人识别配置"""
        return SpeakerConfig(
            model_type=model_type.value,
            model_name="speechbrain/spkrec-ecapa-voxceleb",
            device=device,
            threshold=threshold
        )

    @staticmethod
    def create_stt_service_config(
            asr_config,
            speaker_config,
            vector_db_config: VectorDBConfig,
            storage_config: StorageConfig,
            whisper_config: Optional[WhisperConfig] = None,  # 新增
            transcription_strategy: TranscriptionStrategy = TranscriptionStrategy.AUTO_SELECT,  # 新增
            language_model_mapping: Optional[Dict[str, str]] = None,  # 新增
            **kwargs
    ) -> STTServiceConfig:
        """创建向量服务配置"""

        if vector_db_config is None:
            vector_db_config = {}

        # 设置默认语言映射
        if language_model_mapping is None:
            language_model_mapping = {
                "auto": "whisper",
                "zh": "funasr",
                "en": "whisper",
                "ja": "whisper",
                "ko": "whisper"
            }

        return STTServiceConfig(
            asr_config=asr_config,
            whisper_config=whisper_config,
            speaker_config=speaker_config,
            vector_db_config=vector_db_config,
            storage_config=storage_config,
            transcription_strategy=transcription_strategy,
            language_model_mapping=language_model_mapping,
            **kwargs
        )


# ============================================================================
# 使用示例
# ============================================================================

async def main_stt_example():
    """向量数据库版本的使用示例"""

    # 创建配置（需要根据实际情况调整导入）
    # from your_original_file import ConfigFactory  # 需要从原文件导入

    asr_config = STTConfigFactory.create_funasr_config(
        model_name="paraformer-zh",
        device="cpu"
    )

    speaker_config = STTConfigFactory.create_speaker_config(
        threshold=0.5,
        device="cpu"
    )

    # 向量数据库配置
    vector_db_config = {
        'persist_directory': './voice_print_vectors',
        'collection_name': 'production_voice_prints'
    }

    # 创建向量服务配置
    service_config = STTConfigFactory.create_stt_service_config(
        asr_config=asr_config,
        speaker_config=speaker_config,
        vector_db_type=VectorDBType.CHROMADB,
        vector_db_config=vector_db_config,
        max_transcribe_concurrent=2,
        max_speaker_concurrent=3,
        task_timeout=300.0
    )

    # 创建并使用向量异步服务
    async with STTAsyncVoice2TextService(service_config) as service:
        # 注册声纹
        print("=== 注册声纹 ===")
        register_task_id = await service.register_voice_async(
            "刘星",
            "../../data/sample/刘星.mp3"
        )

        register_result = await service.get_transcribe_result(register_task_id)
        print(f"注册结果: {register_result}")
        register_task_id = await service.register_voice_async(
            "刘梅",
            "../../data/sample/刘梅.mp3"
        )
        register_task_id = await service.register_voice_async(
            "夏东海",
            "../../data/sample/夏东海.mp3"
        )

        # 列出已注册声纹
        print("\n=== 已注册声纹 ===")
        voices = await service.list_registered_voices_async()
        print(f"注册的声纹: {voices}")

        # 转写音频文件
        print("\n=== 转写音频 ===")
        audio_file = "../../data/刘星家_20231212_122300_家有儿女吃饭.mp3"
        task_id = await service.transcribe_file_async(
            audio_file,
            auto_register_unknown=True,
            priority=1
        )

        print(f"提交转写任务: {task_id}")
        result = await service.get_transcribe_result(task_id)
        print(f"转写结果: {result['transcript']}")

        # # 搜索相似声纹
        # print("\n=== 搜索相似声纹 ===")
        # similar_voices = await service.search_similar_voices(
        #     "path/to/query_audio.wav",
        #     top_k=5,
        #     threshold=0.7
        # )
        # print(f"相似声纹: {similar_voices}")

        # 获取统计信息
        print("\n=== 统计信息 ===")
        stats = await service.get_speaker_statistics_async()
        print(f"说话人统计: {stats}")

        register_results = await service.list_registered_voices_async()
        print(f"所有已注册的声纹: {register_results}")

        # 备份向量数据库
        print("\n=== 备份数据库 ===")
        backup_success = await service.backup_vector_database("./voice_backup.json")
        print(f"备份结果: {backup_success}")


if __name__ == "__main__":
    # 运行向量数据库示例
    print("=== 向量数据库版本示例 ===")
    asyncio.run(main_stt_example())
