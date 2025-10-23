import os
import re
import sys
from pathlib import Path
from time import sleep
from typing import Dict, List, Tuple

import torch
from PIL import Image

from .utils_misc import print_error

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_GEOMETRY_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = _PROJECT_ROOT / "visual-navigation"
for _path in (str(_GEOMETRY_DIR), str(_CONFIG_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from config import (  # noqa: E402
    HF_DEVICE,
    HF_MAX_NEW_TOKENS,
    HF_MODEL_ID,
    HF_REPETITION_PENALTY,
    HF_TEMPERATURE,
    HF_TOP_P,
    HF_TRUST_REMOTE_CODE,
)

try:
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
except Exception as exc:  # pragma: no cover - handled at runtime
    AutoProcessor = None  # type: ignore
    Qwen2VLForConditionalGeneration = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


IMG_TAG_PATTERN = re.compile(r"<img\s+src=['\"]([^'\"]+)['\"][^>]*>", re.IGNORECASE)

_processor = None
_model = None


def _select_device() -> str:
    pref = (HF_DEVICE or "").strip()
    if not pref or pref.lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def _resolve_image_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return path

    candidates = [
        Path.cwd() / path,
        _PROJECT_ROOT / path,
        Path(_PROJECT_ROOT / "geometry") / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _load_image(path_str: str) -> Image.Image | None:
    resolved = _resolve_image_path(path_str)
    if not resolved.exists():
        print_error(f"[HF-Qwen] image not found: {path_str}")
        return None
    try:
        with Image.open(resolved) as img:
            return img.convert("RGB")
    except Exception as exc:  # pragma: no cover - I/O errors at runtime
        print_error(f"[HF-Qwen] failed to open image {resolved}: {exc}")
        return None


def _ensure_model():
    global _processor, _model, _device, _dtype

    if _processor is not None and _model is not None:
        return

    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "Failed to import transformers/Qwen2VL. "
            "Install transformers>=4.41 and accelerate together with torch."
        ) from _IMPORT_ERROR

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    model_device = _select_device()
    model_device_lower = model_device.lower()
    use_cuda = model_device_lower.startswith("cuda") and torch.cuda.is_available()
    torch_dtype = torch.float16 if use_cuda else torch.float32

    _processor = AutoProcessor.from_pretrained(
        HF_MODEL_ID,
        trust_remote_code=HF_TRUST_REMOTE_CODE,
    )
    load_kwargs: Dict[str, object] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": HF_TRUST_REMOTE_CODE,
    }
    if use_cuda:
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = None

    _model = Qwen2VLForConditionalGeneration.from_pretrained(
        HF_MODEL_ID,
        **load_kwargs,
    )
    if not use_cuda:
        _model.to("cpu")


def _content_to_messages(messages: List[Dict[str, str]]) -> Tuple[List[Dict[str, object]], List[Image.Image]]:
    multimodal_messages: List[Dict[str, object]] = []
    collected_images: List[Image.Image] = []

    for msg in messages:
        raw = msg["content"]
        items: List[Dict[str, object]] = []
        last_idx = 0

        for match in IMG_TAG_PATTERN.finditer(raw):
            start, end = match.span()
            text_chunk = raw[last_idx:start]
            if text_chunk:
                items.append({"type": "text", "text": text_chunk})

            img_src = match.group(1)
            image = _load_image(img_src)
            if image is not None:
                items.append({"type": "image"})
                collected_images.append(image)
            else:
                # Preserve the tag text when image loading fails.
                items.append({"type": "text", "text": match.group(0)})
            last_idx = end

        tail_text = raw[last_idx:]
        if tail_text or not items:
            items.append({"type": "text", "text": tail_text})

        # Reduce to string if there is only one plain-text entry.
        if len(items) == 1 and items[0].get("type") == "text":
            content = items[0]["text"]
        else:
            content = items

        multimodal_messages.append({"role": msg["role"], "content": content})

    return multimodal_messages, collected_images


def _generate_with_qwen(clean_messages: List[Dict[str, str]], temperature: float) -> str:
    _ensure_model()
    assert _processor is not None and _model is not None  # for type checkers

    formatted_messages, images = _content_to_messages(clean_messages)
    chat_text = _processor.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    processor_kwargs: Dict[str, object] = {"text": [chat_text]}
    if images:
        processor_kwargs["images"] = [images]

    inputs = _processor(
        **processor_kwargs,
        return_tensors="pt",
    )

    torch_device = torch.device(_select_device() if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(torch_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    gen_kwargs: Dict[str, object] = {
        "max_new_tokens": HF_MAX_NEW_TOKENS,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = HF_TOP_P
    else:
        gen_kwargs["do_sample"] = False

    if HF_REPETITION_PENALTY != 1.0:
        gen_kwargs["repetition_penalty"] = HF_REPETITION_PENALTY

    with torch.no_grad():
        generated = _model.generate(**inputs, **gen_kwargs)

    input_length = inputs["input_ids"].shape[-1]
    response_ids = generated[:, input_length:]
    decoded = _processor.batch_decode(response_ids, skip_special_tokens=True)[0]
    return decoded.strip()


def chat_vlm(prompt: str, history_messages=None, temperature: float = 0.0, retry_times: int = 3):
    if history_messages is None:
        history_messages = []

    clean_messages = history_messages + [{"role": "user", "content": prompt}]

    interval = 1
    for attempt in range(retry_times):
        try:
            response_content = _generate_with_qwen(clean_messages, temperature)
            messages = clean_messages + [{"role": "assistant", "content": response_content}]
            return response_content, messages
        except Exception as exc:  # pragma: no cover - runtime robustness
            print_error(f"[HF-Qwen] generation failed (attempt {attempt + 1}/{retry_times}): {exc}")
            if attempt >= retry_times - 1:
                raise
            sleep(interval)
            interval = min(interval * 2, 60)


if __name__ == "__main__":
    demo_prompt = "Hello! Introduce yourself briefly."
    print(chat_vlm(demo_prompt, temperature=0.2)[0])
