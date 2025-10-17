import json
import os
import sys
from pathlib import Path

from autogen.agentchat.contrib.img_utils import gpt4v_formatter
from autogen.oai.client import OpenAIWrapper

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _PROJECT_ROOT / "visual-navigation"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

from config import llm_config  # noqa: E402

def chat_gpt4o(prompt: str, history_messages = None):
    if history_messages is None:
        history_messages = []
    clean_messages = history_messages + [{"role": "user", "content":  prompt}]
    dirty_messages = [{'role': mdict['role'], 'content': gpt4v_formatter(mdict['content'])} for mdict in clean_messages]
    client = OpenAIWrapper(**llm_config)
    response = client.create(
        messages=dirty_messages,
        temperature=0.8,
    )
    messages = clean_messages + [{"role": "assistant", "content": response.choices[0].message.content}]
    return response.choices[0].message.content, messages
