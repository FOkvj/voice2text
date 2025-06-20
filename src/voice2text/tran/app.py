import asyncio

from voice2text.tran.server import VoiceSDKServer
from voice2text.tran.speech2text import VectorConfigFactory, VectorAsyncVoice2TextService, ConfigFactory
from vector_base import VectorDBType
import uvicorn
#
# if __name__ == "__main__":
#     asr_config = ConfigFactory.create_funasr_config(
#         model_name="paraformer-zh",
#         device="cpu"
#     )
#
#     speaker_config = ConfigFactory.create_speaker_config(
#         threshold=0.5,
#         device="cpu"
#     )
#
#     # 向量数据库配置
#     vector_db_config = {
#         'persist_directory': './voice_print_vectors',
#         'collection_name': 'production_voice_prints'
#     }
#
#     # 创建向量服务配置
#     service_config = VectorConfigFactory.create_vector_service_config(
#         asr_config=asr_config,
#         speaker_config=speaker_config,
#         vector_db_type=VectorDBType.CHROMADB,
#         vector_db_config=vector_db_config,
#         max_transcribe_concurrent=2,
#         max_speaker_concurrent=3,
#         task_timeout=300.0
#     )
#
#     # 启动FastAPI服务
#     VoiceSDKServer(VectorAsyncVoice2TextService(service_config)).start()


async def start_server():
    # 1. 创建语音服务实例
    asr_config = ConfigFactory.create_funasr_config(
        model_name="paraformer-zh",
        device="cpu"
    )

    speaker_config = ConfigFactory.create_speaker_config(
        threshold=0.5,
        device="cpu"
    )

    # 向量数据库配置
    vector_db_config = {
        'persist_directory': './voice_print_vectors',
        'collection_name': 'production_voice_prints'
    }

    # 创建向量服务配置
    service_config = VectorConfigFactory.create_vector_service_config(
        asr_config=asr_config,
        speaker_config=speaker_config,
        vector_db_type=VectorDBType.CHROMADB,
        vector_db_config=vector_db_config,
        max_transcribe_concurrent=2,
        max_speaker_concurrent=3,
        task_timeout=300.0
    )

    voice_service = VectorAsyncVoice2TextService(service_config)
    await voice_service.start()
    # 2. 创建并启动服务器
    server = VoiceSDKServer(voice_service)

    # 3. 使用uvicorn运行FastAPI应用
    config = uvicorn.Config(
        app=server.app,
        host="0.0.0.0",
        port=8765,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(start_server())