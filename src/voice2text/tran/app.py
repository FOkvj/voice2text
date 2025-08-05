# ============================================================================
# 使用示例
# ============================================================================
import asyncio

from voice2text.tran.client import VoiceSDKClient


async def example_usage():
    """SDK使用示例"""

    # 客户端使用示例
    async with VoiceSDKClient(base_url="http://localhost:8765", timeout=600) as client:
        # 1. 健康检查
        # health = await client.health_check()
        # print(f"服务状态: {health.message}")
        # await client.rename_speaker("Speaker_68dc353b", "夏东海2")
        # 注册声纹
        #
        # upload_voice = await client.upload_audio_file("../../data/sample/刘星.mp3", category="voiceprint")
        # if upload_voice.success:
        #     register_result = await client.register_voiceprint("刘星", upload_voice.data['file_id'])
        #
        #
        # # 2. 上传音频文件
        # upload_result = await client.upload_audio_file("../../data/刘星家_20231212_122300_家有儿女吃饭.mp3")
        # if upload_result.success:
        #     file_id = upload_result.data['file_id']
        #     print(f"文件上传成功，文件ID: {file_id}")
        #
        #     # 3. 提交转写任务
        #     transcribe_task = await client.transcribe_audio(file_id)
        #     if transcribe_task.success:
        #         task_id = transcribe_task.data['task_id']
        #         print(f"转写任务已提交: {task_id}")
        #
        #         # 4. 等待任务完成
        #         result = await client.wait_for_task_completion(task_id)
        #         if result.success:
        #             print(f"转写结果: {result.data['transcript']}")
        #         else:
        #             print(f"转写失败: {result.message}")

        r1 = await client.register_voiceprint_direct("刘星", "../../data/sample/刘星.mp3")
        print(f"声纹注册结果: {r1.message}, 数据: {r1.data}")
        # r2 = await client.transcribe_file_direct("../../data/刘星家_20231212_122300_家有儿女吃饭.mp3")
        # print(f"转写结果: {r2.message}, 数据: {r2.data}")
        # 5. 获取声纹列表
        # voiceprints = await client.list_voiceprints()
        # if voiceprints.success:
        #     print(f"已注册声纹: {voiceprints.data}")

        # await client.delete_speaker("刘星")
        #
        # await client.delete_speaker_audio_sample("夏东海2", "d178f549-d74e-4437-9538-d5b0e1f53853")
        #
        voiceprints = await client.list_voiceprints()
        if voiceprints.success:
            print(f"已注册声纹: {voiceprints.data}")

        # await client.delete_speaker()
        # await client.download_file("160a3e39-107e-4727-a371-eeff3b095eaf", "./刘星_sample5.")


if __name__ == "__main__":


    # 运行示例
    asyncio.run(example_usage())