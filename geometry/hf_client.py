"""Hugging Face backend for geometry solver chat interactions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


def _resolve_dtype(dtype_str: Optional[str]):
    if not dtype_str or dtype_str.lower() == "auto":
        return None

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Hugging Face backend requires the 'torch' package. Install it via pip (e.g. pip install torch)."
        ) from exc

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "double": torch.float64,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_str.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported HF_DTYPE value '{dtype_str}'.")
    return mapping[key]


def _bool_from_config(value: Any, default: Optional[bool] = None) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return default


@lru_cache(maxsize=1)
def _flash_attention_available() -> bool:
    try:
        import flash_attn_2_cuda  # noqa: F401

        return True
    except ImportError:
        try:
            import flash_attn  # noqa: F401

            return True
        except ImportError:
            return False


@dataclass
class _HFSettings:
    model_id: str
    model_display: str
    temperature: float
    max_new_tokens: int
    do_sample: bool
    top_p: Optional[float]
    top_k: Optional[int]
    repetition_penalty: Optional[float]
    device_pref: str
    dtype: Any
    trust_remote_code: bool
    load_in_8bit: bool
    load_in_4bit: bool
    bnb_4bit_compute_dtype: Any
    bnb_4bit_quant_type: Optional[str]
    bnb_4bit_use_double_quant: bool
    llm_int8_threshold: Optional[float]
    chat_template_pref: str
    hf_token: Optional[str]
    revision: Optional[str]
    local_files_only: bool
    attn_implementation: Optional[str]


class HuggingFaceChatClient:
    """Simple wrapper around ``transformers`` chat-style generation."""

    def __init__(self, config: Dict[str, Any]) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "Hugging Face backend requires the 'torch' package. Install it via pip (e.g. pip install torch)."
            ) from exc
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "Hugging Face backend requires the 'transformers' package. Install it via pip install transformers."
            ) from exc

        self._torch = torch
        self._AutoModelForCausalLM = AutoModelForCausalLM
        self._AutoProcessor = AutoProcessor
        self._AutoTokenizer = AutoTokenizer

        model_id = config.get("model_path") or config.get("model")
        if not model_id:
            raise ValueError("Hugging Face config must include a 'model' or 'model_path' value.")
        temperature = float(config.get("temperature", 0.0) or 0.0)
        max_new_tokens = int(config.get("max_new_tokens", 512) or 512)
        do_sample_override = _bool_from_config(config.get("do_sample"))
        do_sample = do_sample_override if do_sample_override is not None else temperature > 0

        top_p = config.get("top_p")
        top_k = config.get("top_k")
        repetition_penalty = config.get("repetition_penalty")

        dtype = _resolve_dtype(config.get("dtype"))
        load_in_8bit = bool(config.get("load_in_8bit", False))
        load_in_4bit = bool(config.get("load_in_4bit", False))
        bnb_compute_dtype = _resolve_dtype(config.get("bnb_4bit_compute_dtype"))
        if bnb_compute_dtype is None:
            if bool(config.get("bnb_4bit_compute_fp32", False)):
                bnb_compute_dtype = torch.float32
            else:
                bnb_compute_dtype = torch.bfloat16
        bnb_quant_type = config.get("bnb_4bit_quant_type") or "nf4"
        bnb_use_double_quant = bool(config.get("bnb_4bit_use_double_quant", True))
        llm_int8_threshold = config.get("llm_int8_threshold")
        if llm_int8_threshold is not None:
            try:
                llm_int8_threshold = float(llm_int8_threshold)
            except (TypeError, ValueError):
                llm_int8_threshold = None

        self._settings = _HFSettings(
            model_id=model_id,
            model_display=str(config.get("model", model_id)),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=float(top_p) if top_p is not None else None,
            top_k=int(top_k) if top_k is not None else None,
            repetition_penalty=float(repetition_penalty) if repetition_penalty is not None else None,
            device_pref=(config.get("device") or "auto").lower(),
            dtype=dtype,
            trust_remote_code=bool(config.get("trust_remote_code", True)),
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=bnb_compute_dtype,
            bnb_4bit_quant_type=bnb_quant_type,
            bnb_4bit_use_double_quant=bnb_use_double_quant,
            llm_int8_threshold=llm_int8_threshold,
            chat_template_pref=(config.get("chat_template") or "auto").lower(),
            hf_token=config.get("hf_token"),
            revision=config.get("revision"),
            local_files_only=bool(config.get("local_files_only", False)),
            attn_implementation=config.get("attn_implementation"),
        )

        if self._settings.load_in_8bit and self._settings.load_in_4bit:
            raise ValueError("Set only one of HF_LOAD_IN_8BIT or HF_LOAD_IN_4BIT (not both).")

        self._tokenizer = None
        self._processor = None
        self._model = None
        self._device_for_generation = None

        self._load_processor()
        self._load_tokenizer()
        self._load_model()
        self._prepare_generation_device()

        logger.info(
            "Loaded Hugging Face model '%s' (device=%s, dtype=%s, do_sample=%s)",
            self._settings.model_display,
            self._device_for_generation,
            getattr(self._settings.dtype, "name", self._settings.dtype),
            self._settings.do_sample,
        )

    # --------------------------------------------------------------------- #
    # Loading helpers
    # --------------------------------------------------------------------- #
    def _common_kwargs(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "trust_remote_code": self._settings.trust_remote_code,
        }
        if self._settings.hf_token:
            params["token"] = self._settings.hf_token
        if self._settings.revision:
            params["revision"] = self._settings.revision
        if self._settings.local_files_only:
            params["local_files_only"] = True
        return params

    def _load_processor(self) -> None:
        try:
            self._processor = self._AutoProcessor.from_pretrained(
                self._settings.model_id,
                **self._common_kwargs(),
            )
        except Exception as exc:
            logger.debug("Processor not available for %s: %s", self._settings.model_id, exc)
            self._processor = None

    def _load_tokenizer(self) -> None:
        if self._processor is not None and hasattr(self._processor, "tokenizer"):
            tokenizer = getattr(self._processor, "tokenizer")
            if tokenizer is not None:
                self._tokenizer = tokenizer
                return

        try:
            tokenizer = self._AutoTokenizer.from_pretrained(
                self._settings.model_id,
                **self._common_kwargs(),
            )
        except Exception as exc:
            raise RuntimeError(f"Unable to load tokenizer for '{self._settings.model_id}': {exc}") from exc

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self._tokenizer = tokenizer

    def _resolve_device_map(self) -> Any:
        device_pref = self._settings.device_pref
        torch = self._torch

        if device_pref in {"auto", ""}:
            return "auto" if torch.cuda.is_available() else "cpu"

        if device_pref == "cuda" and not torch.cuda.is_available():
            logger.warning("HF_DEVICE=cuda requested but CUDA is unavailable. Falling back to CPU.")
            return "cpu"

        return device_pref

    def _load_model(self) -> None:
        model_kwargs: Dict[str, Any] = self._common_kwargs()
        quantization_config = None

        device_map = self._resolve_device_map()

        if self._settings.load_in_4bit or self._settings.load_in_8bit:
            try:
                import bitsandbytes as _  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "HF_LOAD_IN_4BIT/HF_LOAD_IN_8BIT requires the 'bitsandbytes' package. "
                    "Install it with `pip install bitsandbytes`."
                ) from exc
            if not self._torch.cuda.is_available():
                raise RuntimeError(
                    "4-bit/8-bit quantisation via bitsandbytes requires a CUDA-enabled GPU. "
                    "Disable HF_LOAD_IN_4BIT/HF_LOAD_IN_8BIT or provide a GPU."
                )
            from transformers import BitsAndBytesConfig

            if self._settings.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self._settings.bnb_4bit_compute_dtype,
                    bnb_4bit_use_double_quant=self._settings.bnb_4bit_use_double_quant,
                    bnb_4bit_quant_type=self._settings.bnb_4bit_quant_type,
                )
            else:
                int8_kwargs: Dict[str, Any] = {"load_in_8bit": True}
                if self._settings.llm_int8_threshold is not None:
                    int8_kwargs["llm_int8_threshold"] = self._settings.llm_int8_threshold
                quantization_config = BitsAndBytesConfig(**int8_kwargs)

            if device_map == "cuda":
                device_map = "auto"

        model_kwargs["device_map"] = device_map

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        elif self._settings.dtype is not None:
            model_kwargs["torch_dtype"] = self._settings.dtype

        attn_impl = self._settings.attn_implementation
        if attn_impl:
            if "flash" in attn_impl.lower() and not _flash_attention_available():
                logger.warning(
                    "Requested attention implementation '%s' but flash-attn is not installed. Falling back to 'sdpa'.",
                    attn_impl,
                )
                attn_impl = "sdpa"
            model_kwargs["attn_implementation"] = attn_impl

        try:
            model = self._AutoModelForCausalLM.from_pretrained(
                self._settings.model_id,
                **model_kwargs,
            )
        except Exception as primary_exc:
            try:
                from transformers import AutoModelForVision2Seq

                model = AutoModelForVision2Seq.from_pretrained(
                    self._settings.model_id,
                    **model_kwargs,
                )
            except Exception as secondary_exc:
                raise RuntimeError(
                    f"Unable to load Hugging Face model '{self._settings.model_id}'. "
                    f"Tried AutoModelForCausalLM and AutoModelForVision2Seq. "
                    f"Errors: {primary_exc} | {secondary_exc}"
                ) from secondary_exc

        model.eval()
        self._model = model

    def _prepare_generation_device(self) -> None:
        torch = self._torch
        if torch.cuda.is_available():
            self._device_for_generation = torch.device("cuda")
        else:
            self._device_for_generation = torch.device("cpu")

    # --------------------------------------------------------------------- #
    # Prompt preparation helpers
    # --------------------------------------------------------------------- #
    def _format_plain_conversation(self, messages: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for message in messages:
            role = message.get("role", "user").capitalize()
            content = (message.get("content") or "").strip()
            parts.append(f"{role}: {content}".strip())
        parts.append("Assistant:")
        return "\n".join(parts)

    def _apply_chat_template(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        preference = self._settings.chat_template_pref
        if preference == "off":
            return None

        template_fn = None
        if self._processor is not None and hasattr(self._processor, "apply_chat_template"):
            template_fn = getattr(self._processor, "apply_chat_template")
        elif self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            template_fn = getattr(self._tokenizer, "apply_chat_template")

        if template_fn is None:
            return None

        try:
            return template_fn(messages, tokenize=False, add_generation_prompt=True)
        except Exception as exc:
            logger.debug("apply_chat_template failed for %s: %s", self._settings.model_display, exc)
            return None

    def _collect_images(self, messages: List[Dict[str, Any]]) -> List[Any]:
        images: List[Any] = []
        for message in messages:
            for image_path in message.get("images", []):
                try:
                    with Image.open(image_path) as img:
                        images.append(img.convert("RGB"))
                except Exception as exc:
                    logger.warning("Failed to load image '%s': %s", image_path, exc)
        return images

    def _prepare_inputs(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not messages:
            raise ValueError("At least one message is required for Hugging Face generation.")

        text_only_messages = [{"role": m.get("role"), "content": m.get("content", "")} for m in messages]
        prompt = self._apply_chat_template(text_only_messages)
        if prompt is None:
            prompt = self._format_plain_conversation(text_only_messages)

        processor_inputs: Dict[str, Any]
        images = self._collect_images(messages)

        if images and self._processor is None:
            logger.warning(
                "Model %s received %d image(s) but no processor is available. Proceeding without visual inputs.",
                self._settings.model_display,
                len(images),
            )

        include_images = bool(images) and self._processor is not None and hasattr(self._processor, "image_processor")
        if self._processor is not None and images and not include_images:
            logger.warning(
                "Processor for %s does not expose an image processor. Ignoring %d image(s).",
                self._settings.model_display,
                len(images),
            )

        if self._processor is not None:
            processor_inputs = {"text": prompt, "return_tensors": "pt"}
            if include_images:
                processor_inputs["images"] = images
            inputs = self._processor(**processor_inputs)
        else:
            inputs = self._tokenizer(prompt, return_tensors="pt")

        torch = self._torch
        moved_inputs: Dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                moved_inputs[key] = value.to(self._device_for_generation)
            else:
                moved_inputs[key] = value
        return moved_inputs

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def generate(self, messages: List[Dict[str, Any]]) -> str:
        inputs = self._prepare_inputs(messages)
        torch = self._torch

        tokenizer = self._tokenizer
        if tokenizer is None and self._processor is not None and hasattr(self._processor, "tokenizer"):
            tokenizer = getattr(self._processor, "tokenizer")

        pad_token_id = None
        if tokenizer is not None and tokenizer.pad_token_id is not None:
            pad_token_id = tokenizer.pad_token_id

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self._settings.max_new_tokens,
            "pad_token_id": pad_token_id,
        }

        if self._settings.do_sample:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = max(self._settings.temperature, 1e-5)
            if self._settings.top_p is not None:
                generation_kwargs["top_p"] = self._settings.top_p
            if self._settings.top_k is not None:
                generation_kwargs["top_k"] = self._settings.top_k
        else:
            generation_kwargs["do_sample"] = False

        if self._settings.repetition_penalty is not None:
            generation_kwargs["repetition_penalty"] = self._settings.repetition_penalty

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **generation_kwargs)

        input_length = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        generated = outputs[0][input_length:] if input_length > 0 else outputs[0]
        if hasattr(generated, "cpu"):
            generated = generated.cpu()

        if tokenizer is not None:
            text = tokenizer.decode(generated, skip_special_tokens=True)
        elif hasattr(self._processor, "batch_decode"):
            text = self._processor.batch_decode(generated.unsqueeze(0), skip_special_tokens=True)[0]
        else:
            text = ""

        return text.strip()
