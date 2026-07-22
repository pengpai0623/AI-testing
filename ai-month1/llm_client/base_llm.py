import os
import requests
from dotenv import load_dotenv
from typing import List, Dict, Optional

# 加载环境变量
load_dotenv()

class LLMBaseClient:
    def __init__(self):
        self.api_key = os.getenv("DOUBAO_API_KEY")
        self.endpoint = os.getenv("DOUBAO_ENDPOINT")
        self.model_name = os.getenv("DOUBAO_MODEL")
        # 环境变量通过 os.getenv() 读取出来，永远是字符串 str 类型
        self.timeout = int(os.getenv("REQUEST_TIMEOUT"))

        if not self.api_key:
            raise ValueError("DOUBAO_API_KEY 未配置，请检查.env文件")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat_single(self, prompt: str, temperature: str = 0.7, system_prompt: Optional[str] = None) -> Dict:
        """
        单轮对话调用大模型
        :param prompt: 用户提问内容
        :param system_prompt: 系统角色设定
        :param temperature : 固定一个模型内部，选词的随机程度, 取值区间: 0 ~ 2, 
        :return: 字典: content 回答文本 / total_tokens 消耗token / status 状态
        """

        messages :List[Dict] = []
        if system_prompt:
            messages.append({"role": "user", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_body = {
            "model": self.model_name,
            "temperature": 0.7, 
            "messages": messages,
        }

        try:
            resp = requests.post(
                url = self.endpoint,
                headers=self.headers,
                json= request_body,
                timeout=self.timeout
            )
            resp.raise_for_status() # 状态码非200直接抛异常
            resp_data = resp.json()

            content = resp_data["choices"][0]["message"]["content"]
            token_usage = resp_data["usage"]

            return {
                "status": "success",
                "content": content,
                "prompt_tokens": token_usage["prompt_tokens"],
                "completion_tokens": token_usage["completion_tokens"],
                "total_tokens": token_usage["total_tokens"]
            }
        except requests.exceptions.Timeout:
            return {"status": "timeout", "content": "请求大模型超时，请稍后重试"}
        except requests.exceptions.ConnectionError:
            return {"status": "conn_error", "content": "网络连接失败，无法访问模型接口"}
        except requests.exceptions.HTTPError as http_err:
            return {"status": "http_err", "content": f"接口异常：{str(http_err)}，响应内容：{resp.text if 'resp' in locals() else ''}"}
        except Exception as e:
            return {"status": "unknown_err", "content": f"未知异常：{str(e)}"}