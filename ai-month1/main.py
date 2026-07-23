from llm_client.chat_session import ChatSession
from llm_client.base_llm import LLMBaseClient

if __name__ == "__main__":
    llm = LLMBaseClient()
    session = ChatSession(system_content="你叫小助手，记住用户信息", max_token_limit=4000)
    res = session.chat(user_input="我叫张三", llm=llm)
    print(res["content"])
    res2 = session.chat(user_input="我叫什么名字", llm=llm)
    print(res2["content"])
    print("完整消息列表", session.get_all_messages())
    # 手动清空历史
    session.clear_history()