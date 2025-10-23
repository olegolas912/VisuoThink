import json
import os
import re
import sys
from pathlib import Path
from time import sleep
from typing import Dict, List, Tuple

from autogen.agentchat.contrib.img_utils import gpt4v_formatter
from autogen.oai.client import OpenAIWrapper
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'visual-navigation'))
from config import llm_config
from copy import deepcopy

if __package__ is None or __package__ == "":  # pragma: no cover - script execution fallback
    _PKG_DIR = os.path.dirname(os.path.abspath(__file__))
    if _PKG_DIR not in sys.path:
        sys.path.insert(0, _PKG_DIR)
    from utils_misc import print_error  # type: ignore
    from hf_client import HuggingFaceChatClient  # type: ignore
else:
    from .utils_misc import print_error
    from .hf_client import HuggingFaceChatClient

TOKEN_G = 0
TOKEN_USED = 0
_HF_CLIENT_CACHE: Dict[str, "HuggingFaceChatClient"] = {}


def _supports_image_messages(call_config: dict) -> bool:
    config_list = call_config.get("config_list", [])
    if not config_list:
        return True

    for cfg in config_list:
        api_type = (cfg or {}).get("api_type")
        if api_type and api_type.lower().startswith("ollama"):
            return False
        base_url = (cfg or {}).get("base_url", "")
        if isinstance(base_url, str) and "11434" in base_url:
            return False
    return True


def _is_ollama_config(call_config: dict) -> bool:
    return any((cfg or {}).get("api_type", "").lower().startswith("ollama") for cfg in call_config.get("config_list", []))


def _is_hf_config(call_config: dict) -> bool:
    return any((cfg or {}).get("api_type", "").lower() == "huggingface" for cfg in call_config.get("config_list", []))


_IMG_TAG_RE = re.compile(r"<img\s+[^>]*src=['\"]([^'\"]+)['\"][^>]*>", re.IGNORECASE)


def _extract_text_and_images(content: str) -> Tuple[str, List[str]]:
    image_paths: List[str] = []

    def _repl(match: re.Match) -> str:
        raw_path = match.group(1)
        path = Path(raw_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path.exists():
            image_paths.append(str(path))
            return ""
        return f"[Missing image: {raw_path}]"

    text = _IMG_TAG_RE.sub(_repl, content).strip()
    if image_paths:
        filenames = ", ".join(Path(p).name for p in image_paths)
        if text:
            text = f"{text}\n[Attached image(s): {filenames}]"
        else:
            text = f"[Attached image(s): {filenames}]"
    return text, image_paths


def chat_vlm(prompt: str, history_messages=None, temperature: float = 0.0, retry_times: int = 10):
    global TOKEN_USED, TOKEN_G
    call_config = deepcopy(llm_config)
    for s_config in call_config['config_list']:
        s_config['temperature'] = temperature

    use_ollama = _is_ollama_config(call_config)
    use_hf = _is_hf_config(call_config)
    use_image_formatter = not (use_ollama or use_hf) and _supports_image_messages(call_config)

    interval = 1
    for _ in range(retry_times):
        try:
            if history_messages is None:
                history_messages = []
            clean_messages = history_messages + [{"role": "user", "content": prompt}]

            if use_ollama or use_hf:
                dirty_messages = []
                for mdict in clean_messages:
                    text, images = _extract_text_and_images(mdict['content'])
                    payload = {'role': mdict['role'], 'content': text or ""}
                    if images:
                        payload['images'] = images
                    dirty_messages.append(payload)
            elif use_image_formatter:
                dirty_messages = [{'role': mdict['role'], 'content': gpt4v_formatter(mdict['content'])} for mdict in clean_messages]
            else:
                dirty_messages = clean_messages

            if use_ollama:
                try:
                    import ollama
                except ImportError as exc:  # pragma: no cover - dependency guard
                    raise ImportError("The 'ollama' package is required for Ollama API usage") from exc

                cfg = call_config['config_list'][0]
                host = cfg.get('base_url')
                client = ollama.Client(host=host) if host else ollama.Client()
                options = {}
                if 'temperature' in cfg:
                    options['temperature'] = cfg['temperature']
                stream = cfg.get('stream', False)
                response = client.chat(
                    model=cfg['model'],
                    messages=dirty_messages,
                    stream=stream,
                    options=options or None,
                )
                if stream:
                    content_parts: List[str] = []
                    for chunk in response:
                        if 'message' in chunk and 'content' in chunk['message']:
                            content_parts.append(chunk['message']['content'])
                    reply_content = ''.join(content_parts)
                else:
                    reply_content = response['message']['content']
                messages = clean_messages + [{"role": "assistant", "content": reply_content}]
                return reply_content, messages
            elif use_hf:
                cfg = call_config['config_list'][0]
                cache_key = json.dumps(cfg, sort_keys=True)
                client = _HF_CLIENT_CACHE.get(cache_key)
                if client is None:
                    client = HuggingFaceChatClient(cfg)
                    _HF_CLIENT_CACHE[cache_key] = client
                reply_content = client.generate(dirty_messages)
                messages = clean_messages + [{"role": "assistant", "content": reply_content}]
                return reply_content, messages

            client = OpenAIWrapper(**call_config)
            response = client.create(
                messages=dirty_messages,
                timeout=600,
            )
            messages = clean_messages + [{"role": "assistant", "content": response.choices[0].message.content}]
            print(response.usage)
            return response.choices[0].message.content, messages
        except Exception as e:
            if 'limit' in str(e):
                sleep(interval)
                interval = min(interval * 2, 60)
            print_error(e)
    raise RuntimeError("chat_vlm failed after maximum retry attempts")


if __name__ == "__main__":
    print(chat_vlm('Hello.', temperature=0.8)[0])
