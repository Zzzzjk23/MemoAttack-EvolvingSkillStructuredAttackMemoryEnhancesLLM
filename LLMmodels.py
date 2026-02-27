from litai import LLM
from global_context import global_context
from system_prompts import get_attacker_system_prompt
from openai import OpenAI
import json
import os

LLM_choices = {
    'attacker':'Qwen/Qwen3-Coder-480B-A35B-Instruct',
    'evaluator':'moonshotai/Kimi-K2-Instruct',
    'target':'openai/gpt-oss-20b'
}
client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key="v1.CmQKHHN0YXRpY2tleS1lMDBxMDN4MHl3d3FlZ2Q4cDkSIXNlcnZpY2VhY2NvdW50LWUwMHdwcWpmYWcwNXp6YTRrZzIMCMrxgcoGEPyfrr0DOgwIy_SZlQcQgIyNngJAAloDZTAw.AAAAAAAAAAFZJjga8wP8uscjRXekw5wFdE2ySYvpwKLD-pH8KlRyKUYypqrhqdunKX2HlZ7svbE61JCwPV0gTI3eXy-3Z7AG"
)

def convert_to_openai_messages(template):
    
    openai_messages = []
    if template.system_message:
        openai_messages.append({
            "role": "system",
            "content": template.system_message
        })
    for role, content in template.messages:
        if content:
            if role == template.roles[0]:
                # 如果content是字典，转换为JSON字符串
                if isinstance(content, dict):
                    content = json.dumps(content, ensure_ascii=False)
                openai_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": content
                        }
                    ]
                })
            elif role == template.roles[1]:
                # 如果content是字典，转换为JSON字符串
                if isinstance(content, dict):
                    content = json.dumps(content, ensure_ascii=False)
                openai_messages.append({
                    "role": "assistant",
                    "content": content 
                })
    return openai_messages

class AttackerLLM:

    
    def __init__(self, model_name: str, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 5000, goal: str = "", target_str: str = ""):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.llm = LLM(model=self.model_name, api_key="63cfa10b-6e54-42bd-8d87-dbd0ba83a291")
    def generate_prompt(self, conv) -> str:
        openai_messages = convert_to_openai_messages(conv)

        # 使用 function calling 强制返回 JSON
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "attack_result",
                    "description": "Return the jailbreak improvement analysis and the final prompt in strict JSON format.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "improvement": {
                                "type": "string",
                                "description": "A few sentences interpreting the model's response and explaining how the prompt should be modified to better achieve the goal."
                            },
                            "prompt": {
                                "type": "string",
                                "description": "The new adversarial jailbreaking prompt."
                            }
                        },
                        "required": ["improvement", "prompt"]
                    }
                }
            }
        ]

        response = client.chat.completions.create(
            model=LLM_choices['attacker'],
            messages=openai_messages,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {
                    "name": "attack_result"
                }
            },
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        # 函数调用的 JSON 参数在 tool_calls 里，以字符串形式给出
        tool_call = response.choices[0].message.tool_calls[0]
        arguments_str = tool_call.function.arguments

        try:
            return json.loads(arguments_str)
        except json.JSONDecodeError:
            # 理论上不会发生，如果发生就原样返回字符串，方便调试
            return {arguments_str}


class EvaluatorLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llm = LLM(model=self.model_name, api_key="63cfa10b-6e54-42bd-8d87-dbd0ba83a291")
    def out_of_target(self, off_target_system_prompt: str, off_target_prompt: str) -> str:
        response = client.chat.completions.create(
            model=LLM_choices['evaluator'],
            messages=[
                {
                    "role": "system",
                    "content": off_target_system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": off_target_prompt
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content
    def judge(self, judge_system_prompt: str, target_response: str) -> str:
        # 防止 target_response 为 None 导致 OpenAI 接口 422 错误
        if target_response is None:
            target_response = "I am sorry, but I can't help you with that."

        response = client.chat.completions.create(
            model=LLM_choices['evaluator'],
            messages=[
                {
                    "role": "system",
                    "content": judge_system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": target_response
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content
    def prompt_category(self, prompt: str) -> str:
        response = client.chat.completions.create(
            model=LLM_choices['evaluator'],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content
class TargetLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llm = LLM(model=self.model_name, api_key="63cfa10b-6e54-42bd-8d87-dbd0ba83a291")
    def target_response(self, prompt: str) -> str:
        response = client.chat.completions.create(
            model=LLM_choices['target'],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content