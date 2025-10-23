from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

try:
    from transformers import CLIPModel, CLIPProcessor
except Exception:  # pragma: no cover - optional dep
    CLIPModel = None
    CLIPProcessor = None

from .modules import AutoencoderSelector


class CLIPFeatureExtractor(nn.Module):
    """
    Thin wrapper to extract patch tokens from CLIP-like encoders.
    Returns sequence of visual tokens excluding CLS where applicable.
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        if CLIPModel is None:
            raise ImportError("transformers[CLIP] is required for CLIPFeatureExtractor")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

    @torch.no_grad()
    def forward(self, pil_image: Image.Image, device: Optional[str] = None) -> torch.Tensor:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        pixel = inputs["pixel_values"].to(device)
        out = self.model.vision_model(pixel)
        # out.last_hidden_state: [B, 1+N, D] (CLS + patches)
        tokens = out.last_hidden_state[:, 1:, :].contiguous()
        return tokens  # [B, N, D]


class ATRClipWrapper(nn.Module):
    """
    Applies ATR selection to CLIP tokens and returns reduced token set.
    """

    def __init__(self, dim: int = 768, lambda_reg: float = 0.05):
        super().__init__()
        self.selector = AutoencoderSelector(dim=dim, lambda_reg=lambda_reg)

    @torch.no_grad()
    def reduce(self, tokens: torch.Tensor, retention: float) -> Dict[str, torch.Tensor]:
        kept, mask = self.selector.select(tokens, retention=retention)
        return {"kept_tokens": kept, "mask": mask}


