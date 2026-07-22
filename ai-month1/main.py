from llm_client.base_llm import LLMBaseClient

if __name__ == "__main__":
    client = LLMBaseClient()

    # 测试1: 普通问答
    res1 = client.chat_single(
        system_prompt="你是一名资深Python开发工程师, 回答简洁精炼",
        prompt="解释一下FastAPI相比Flask的优势"
    )
    print("===== 正常调用结果 =====")
    print(res1)