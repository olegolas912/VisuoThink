import json
import os
import sys
from datetime import datetime
from autogen.agentchat.contrib.img_utils import (
    gpt4v_formatter,
)
from autogen.oai.client import OpenAIWrapper
from config import llm_config
from utils_misc import print_error
from time import sleep
from copy import deepcopy
from utils_misc import print_error

TOKEN_G = 0
TOKEN_USED = 0
# print error message with red color and bold。                                                                                                                                                                                    

def chat_vlm(prompt: str, history_messages = None, temperature: float = 0., retry_times: int = 10):
    global TOKEN_USED, TOKEN_G
    call_config = deepcopy(llm_config)
    for s_config in call_config['config_list']:
        s_config['temperature'] = temperature

    interval = 1
    for i in range(retry_times):
        try:
            if history_messages is None:
                history_messages = []
            clean_messages = history_messages + [{"role": "user", "content":  prompt}]
            dirty_messages = [{'role': mdict['role'], 'content': gpt4v_formatter(mdict['content'])} for mdict in clean_messages]
            
            client = OpenAIWrapper(**call_config)
            response = client.create(
                messages=dirty_messages,
                timeout=600,
            )
            messages = clean_messages + [{"role": "assistant", "content": response.choices[0].message.content}]
            print(response.usage)
            # TOKEN_USED += response.usage.total_tokens
            # TOKEN_G += response.usage.completion_tokens
            # print_error(f'[Token Gen] {id(TOKEN_G)} {TOKEN_G - response.usage.completion_tokens} -> {TOKEN_G}')
            # print_error(f'[Token Used] {id(TOKEN_USED)} {TOKEN_USED - response.usage.total_tokens} -> {TOKEN_USED}')
            return response.choices[0].message.content, messages
        except Exception as e:
            if 'limit' in str(e):
                sleep(interval)
                interval = min(interval * 2, 60)
            print_error(e)
            if i >= (retry_times - 1):
                raise e

if __name__ == "__main__":
    # print(llm_config)
    print(chat_vlm('Hello.', temperature=0.8)[0])
