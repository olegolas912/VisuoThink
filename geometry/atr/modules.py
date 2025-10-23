import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelBinarySampler(nn.Module):
    """
    Gumbel-Softmax binary sampler producing differentiable hard masks.
    Input logits are of shape [B, N, 2] -> mask in {0,1}^{B,N} (straight-through).
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    @staticmethod
    def _gumbel_noise_like(x: torch.Tensor) -> torch.Tensor:
        eps = 1e-9
        u = torch.empty_like(x).uniform_(0, 1)
        return -torch.log(-torch.log(u + eps) + eps)

    def forward(self, logits_2: torch.Tensor, hard: bool = True) -> torch.Tensor:
        # logits_2: [B, N, 2]
        g = self._gumbel_noise_like(logits_2)
        y = F.softmax((logits_2 + g) / self.temperature, dim=-1)  # [B, N, 2]
        # take the keep-channel (idx=1) as probability
        y_keep = y[..., 1]  # [B, N]
        if hard:
            with torch.no_grad():
                hard_mask = (y_keep >= 0.5).float()
            # straight-through estimator
            y_keep = y_keep + (hard_mask - y_keep).detach()
        return y_keep  # in [0,1], treated as binary during inference


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.drop_path1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)
        self.drop_path2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = self.drop_path1(x)
        x = x + h

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path2(x)
        x = x + h
        return x


class TransformerFeatureSelector(nn.Module):
    """
    Selector: 3 transformer blocks + linear(2) -> Gumbel-Softmax -> binary mask.
    """

    def __init__(self, dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.blocks = nn.Sequential(
            TransformerBlock(dim, num_heads),
            TransformerBlock(dim, num_heads),
            TransformerBlock(dim, num_heads),
        )
        self.head = nn.Linear(dim, 2)
        self.gumbel = GumbelBinarySampler(temperature=1.0)

    def forward(self, tokens: torch.Tensor, hard: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokens: [B, N, D]
        x = self.blocks(tokens)
        logits2 = self.head(x)  # [B, N, 2]
        mask = self.gumbel(logits2, hard=hard)  # [B, N]
        return mask, logits2


class TransformerFeatureReconstructor(nn.Module):
    """
    Reconstructor: 3 transformer blocks. Takes selected tokens and tries to
    reconstruct original tokens through cross-attention-like expansion.
    For simplicity, we project selected tokens to N positions with learned queries.
    """

    def __init__(self, dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.dim = dim
        self.blocks = nn.Sequential(
            TransformerBlock(dim, num_heads),
            TransformerBlock(dim, num_heads),
            TransformerBlock(dim, num_heads),
        )
        self.query_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, kept_tokens: torch.Tensor, target_len: int) -> torch.Tensor:
        # kept_tokens: [B, K, D]
        x = self.blocks(kept_tokens)
        # Create learnable queries per forward (parameter-free): use mean token as seed
        mean_token = x.mean(dim=1, keepdim=True)  # [B,1,D]
        queries = mean_token.repeat(1, target_len, 1)  # [B,N,D]
        queries = self.query_proj(queries)
        # Attend queries to kept tokens (scaled dot-product via MultiheadAttention emulation)
        # Using a single MHA call by concatenating queries and keys requires larger plumbing;
        # approximate with residual fusion
        expanded = queries + x.mean(dim=1, keepdim=True)
        expanded = self.out_proj(expanded)
        return expanded  # [B,N,D]


class AutoencoderSelector(nn.Module):
    """
    Autoencoder with selector + reconstructor.
    Loss: reconstruction + lambda * fraction_kept
    """

    def __init__(self, dim: int = 768, num_heads: int = 12, lambda_reg: float = 0.05):
        super().__init__()
        self.selector = TransformerFeatureSelector(dim=dim, num_heads=num_heads)
        self.reconstructor = TransformerFeatureReconstructor(dim=dim, num_heads=num_heads)
        self.lambda_reg = lambda_reg

    def forward(self, tokens: torch.Tensor, retention: Optional[float] = None, hard: bool = True):
        # tokens: [B,N,D]
        B, N, D = tokens.shape
        mask, logits2 = self.selector(tokens, hard=hard)  # [B,N]

        if retention is not None:
            # enforce exact retention by thresholding top-k
            k = max(1, int(N * retention))
            topk = torch.topk(mask, k=k, dim=1).values[:, -1:]
            hard_mask = (mask >= topk).float()
            mask = mask + (hard_mask - mask).detach()

        kept = mask.unsqueeze(-1) * tokens  # [B,N,D]; zeros masked tokens
        # compact kept tokens into contiguous sequence (gather)
        # to avoid non-differentiable gather on hard mask, keep dense and average
        # compute kept mean and broadcast; for training, this is sufficient
        # for inference we will hard-select top-k separately
        kept_tokens = kept  # [B,N,D]

        recon = self.reconstructor(kept_tokens.mean(dim=1, keepdim=True).repeat(1, N, 1), target_len=N)
        recon_loss = F.mse_loss(recon, tokens)
        sparsity = mask.mean()
        total = recon_loss + self.lambda_reg * sparsity
        return {
            "loss": total,
            "recon_loss": recon_loss,
            "sparsity": sparsity,
            "mask": mask,
            "logits2": logits2,
            "reconstructed": recon,
        }

    @torch.no_grad()
    def select(self, tokens: torch.Tensor, retention: float) -> Tuple[torch.Tensor, torch.Tensor]:
        # Inference: produce hard binary mask with exact retention
        B, N, D = tokens.shape
        mask_soft, _ = self.selector(tokens, hard=False)
        k = max(1, int(N * retention))
        topk_vals, topk_idx = torch.topk(mask_soft, k=k, dim=1)
        hard_mask = torch.zeros_like(mask_soft)
        hard_mask.scatter_(1, topk_idx, 1.0)
        kept_tokens = torch.gather(tokens, 1, topk_idx.unsqueeze(-1).expand(-1, -1, D))
        return kept_tokens, hard_mask


