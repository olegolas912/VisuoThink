"""
Heuristic-based Token Selector for Adaptive Token Reduction.

This module implements a no-training-required approach to token selection
using visual importance heuristics instead of learned weights.

Strategies:
1. Variance-based: Select tokens with high variance (more information)
2. Attention-based: Use CLIP's attention weights
3. Edge-based: Detect edges and high-frequency content
4. Saliency-based: Combined approach using multiple metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Literal
from PIL import Image


class VarianceBasedSelector:
    """
    Selects tokens based on variance - tokens with higher variance contain
    more information and are more likely to be important.
    """
    
    @staticmethod
    @torch.no_grad()
    def select(tokens: torch.Tensor, retention: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select tokens based on variance.
        
        Args:
            tokens: [B, N, D] feature tokens
            retention: fraction of tokens to keep (e.g., 0.3 for 30%)
            
        Returns:
            kept_tokens: [B, K, D] where K = int(N * retention)
            mask: [B, N] binary mask
        """
        B, N, D = tokens.shape
        
        # Compute variance across feature dimension for each token
        variance = tokens.var(dim=-1)  # [B, N]
        
        # Select top-k tokens with highest variance
        k = max(1, int(N * retention))
        topk_vals, topk_idx = torch.topk(variance, k=k, dim=1)
        
        # Create binary mask
        mask = torch.zeros_like(variance)
        mask.scatter_(1, topk_idx, 1.0)
        
        # Gather selected tokens
        kept_tokens = torch.gather(tokens, 1, topk_idx.unsqueeze(-1).expand(-1, -1, D))
        
        return kept_tokens, mask


class NormBasedSelector:
    """
    Selects tokens based on L2 norm - tokens with larger norms are typically
    more activated and contain more semantic information.
    """
    
    @staticmethod
    @torch.no_grad()
    def select(tokens: torch.Tensor, retention: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select tokens based on L2 norm.
        
        Args:
            tokens: [B, N, D] feature tokens
            retention: fraction of tokens to keep
            
        Returns:
            kept_tokens: [B, K, D]
            mask: [B, N] binary mask
        """
        B, N, D = tokens.shape
        
        # Compute L2 norm for each token
        norms = torch.norm(tokens, p=2, dim=-1)  # [B, N]
        
        # Select top-k tokens with highest norms
        k = max(1, int(N * retention))
        topk_vals, topk_idx = torch.topk(norms, k=k, dim=1)
        
        # Create binary mask
        mask = torch.zeros_like(norms)
        mask.scatter_(1, topk_idx, 1.0)
        
        # Gather selected tokens
        kept_tokens = torch.gather(tokens, 1, topk_idx.unsqueeze(-1).expand(-1, -1, D))
        
        return kept_tokens, mask


class EntropyBasedSelector:
    """
    Selects tokens based on information entropy - tokens with higher entropy
    contain more diverse information.
    """
    
    @staticmethod
    @torch.no_grad()
    def select(tokens: torch.Tensor, retention: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select tokens based on entropy of normalized features.
        
        Args:
            tokens: [B, N, D] feature tokens
            retention: fraction of tokens to keep
            
        Returns:
            kept_tokens: [B, K, D]
            mask: [B, N] binary mask
        """
        B, N, D = tokens.shape
        
        # Normalize tokens to [0, 1] for entropy calculation
        tokens_norm = (tokens - tokens.min(dim=-1, keepdim=True)[0])
        tokens_norm = tokens_norm / (tokens_norm.max(dim=-1, keepdim=True)[0] + 1e-8)
        
        # Compute entropy: -sum(p * log(p))
        eps = 1e-8
        entropy = -(tokens_norm * torch.log(tokens_norm + eps)).sum(dim=-1)  # [B, N]
        
        # Select top-k tokens with highest entropy
        k = max(1, int(N * retention))
        topk_vals, topk_idx = torch.topk(entropy, k=k, dim=1)
        
        # Create binary mask
        mask = torch.zeros_like(entropy)
        mask.scatter_(1, topk_idx, 1.0)
        
        # Gather selected tokens
        kept_tokens = torch.gather(tokens, 1, topk_idx.unsqueeze(-1).expand(-1, -1, D))
        
        return kept_tokens, mask


class CombinedSaliencySelector:
    """
    Combines multiple heuristics to compute a comprehensive saliency score.
    This is the recommended selector for general use.
    """
    
    def __init__(
        self, 
        variance_weight: float = 0.4,
        norm_weight: float = 0.4,
        entropy_weight: float = 0.2
    ):
        """
        Initialize combined selector with weighted heuristics.
        
        Args:
            variance_weight: Weight for variance-based score
            norm_weight: Weight for norm-based score
            entropy_weight: Weight for entropy-based score
        """
        self.variance_weight = variance_weight
        self.norm_weight = norm_weight
        self.entropy_weight = entropy_weight
        
        # Normalize weights
        total = variance_weight + norm_weight + entropy_weight
        self.variance_weight /= total
        self.norm_weight /= total
        self.entropy_weight /= total
    
    @torch.no_grad()
    def select(self, tokens: torch.Tensor, retention: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select tokens using combined saliency score.
        
        Args:
            tokens: [B, N, D] feature tokens
            retention: fraction of tokens to keep
            
        Returns:
            kept_tokens: [B, K, D]
            mask: [B, N] binary mask
        """
        B, N, D = tokens.shape
        
        # Compute variance score
        variance = tokens.var(dim=-1)  # [B, N]
        variance_norm = (variance - variance.min(dim=1, keepdim=True)[0])
        variance_norm = variance_norm / (variance_norm.max(dim=1, keepdim=True)[0] + 1e-8)
        
        # Compute norm score
        norms = torch.norm(tokens, p=2, dim=-1)  # [B, N]
        norms_norm = (norms - norms.min(dim=1, keepdim=True)[0])
        norms_norm = norms_norm / (norms_norm.max(dim=1, keepdim=True)[0] + 1e-8)
        
        # Compute entropy score
        tokens_norm = (tokens - tokens.min(dim=-1, keepdim=True)[0])
        tokens_norm = tokens_norm / (tokens_norm.max(dim=-1, keepdim=True)[0] + 1e-8)
        eps = 1e-8
        entropy = -(tokens_norm * torch.log(tokens_norm + eps)).sum(dim=-1)  # [B, N]
        entropy_norm = (entropy - entropy.min(dim=1, keepdim=True)[0])
        entropy_norm = entropy_norm / (entropy_norm.max(dim=1, keepdim=True)[0] + 1e-8)
        
        # Combine scores
        saliency = (
            self.variance_weight * variance_norm +
            self.norm_weight * norms_norm +
            self.entropy_weight * entropy_norm
        )
        
        # Select top-k tokens with highest saliency
        k = max(1, int(N * retention))
        topk_vals, topk_idx = torch.topk(saliency, k=k, dim=1)
        
        # Create binary mask
        mask = torch.zeros_like(saliency)
        mask.scatter_(1, topk_idx, 1.0)
        
        # Gather selected tokens
        kept_tokens = torch.gather(tokens, 1, topk_idx.unsqueeze(-1).expand(-1, -1, D))
        
        return kept_tokens, mask


class SpatialAwareSelector:
    """
    Selects tokens while considering spatial distribution to avoid
    clustering all selected tokens in one region.
    """
    
    def __init__(self, spatial_weight: float = 0.3):
        """
        Initialize spatial-aware selector.
        
        Args:
            spatial_weight: Weight for spatial diversity penalty
        """
        self.spatial_weight = spatial_weight
        self.base_selector = CombinedSaliencySelector()
    
    @torch.no_grad()
    def select(self, tokens: torch.Tensor, retention: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select tokens with spatial diversity consideration.
        
        Args:
            tokens: [B, N, D] feature tokens
            retention: fraction of tokens to keep
            
        Returns:
            kept_tokens: [B, K, D]
            mask: [B, N] binary mask
        """
        B, N, D = tokens.shape
        
        # Get base saliency scores
        variance = tokens.var(dim=-1)
        norms = torch.norm(tokens, p=2, dim=-1)
        
        variance_norm = (variance - variance.min(dim=1, keepdim=True)[0])
        variance_norm = variance_norm / (variance_norm.max(dim=1, keepdim=True)[0] + 1e-8)
        
        norms_norm = (norms - norms.min(dim=1, keepdim=True)[0])
        norms_norm = norms_norm / (norms_norm.max(dim=1, keepdim=True)[0] + 1e-8)
        
        base_score = 0.5 * variance_norm + 0.5 * norms_norm
        
        # Infer spatial grid (assume square for CLIP)
        grid_size = int(np.sqrt(N))
        if grid_size * grid_size != N:
            # Fallback to base selector for non-square grids
            return self.base_selector.select(tokens, retention)
        
        # Create spatial diversity bonus
        # Tokens near center get slight penalty to encourage boundary selection
        positions = torch.arange(N, device=tokens.device).view(grid_size, grid_size)
        center = grid_size // 2
        
        y_coords = torch.arange(grid_size, device=tokens.device)
        x_coords = torch.arange(grid_size, device=tokens.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Distance from center
        center_dist = torch.sqrt(((y_grid - center) ** 2 + (x_grid - center) ** 2).float())
        center_dist = center_dist.reshape(-1)
        center_dist_norm = center_dist / (center_dist.max() + 1e-8)
        
        # Add slight bonus to non-center tokens
        spatial_bonus = center_dist_norm * self.spatial_weight
        spatial_bonus = spatial_bonus.unsqueeze(0).expand(B, -1)
        
        # Combine scores
        final_score = base_score + spatial_bonus
        
        # Select top-k
        k = max(1, int(N * retention))
        topk_vals, topk_idx = torch.topk(final_score, k=k, dim=1)
        
        # Create binary mask
        mask = torch.zeros_like(final_score)
        mask.scatter_(1, topk_idx, 1.0)
        
        # Gather selected tokens
        kept_tokens = torch.gather(tokens, 1, topk_idx.unsqueeze(-1).expand(-1, -1, D))
        
        return kept_tokens, mask


class HeuristicATRSelector:
    """
    Main interface for heuristic-based Adaptive Token Reduction.
    Provides multiple selection strategies without requiring training.
    """
    
    STRATEGIES = {
        'variance': VarianceBasedSelector,
        'norm': NormBasedSelector,
        'entropy': EntropyBasedSelector,
        'combined': CombinedSaliencySelector,
        'spatial': SpatialAwareSelector,
    }
    
    def __init__(
        self, 
        strategy: Literal['variance', 'norm', 'entropy', 'combined', 'spatial'] = 'combined'
    ):
        """
        Initialize heuristic ATR selector.
        
        Args:
            strategy: Selection strategy to use
                - 'variance': Select tokens with high variance
                - 'norm': Select tokens with high L2 norm
                - 'entropy': Select tokens with high entropy
                - 'combined': Use weighted combination (RECOMMENDED)
                - 'spatial': Combined with spatial diversity
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(self.STRATEGIES.keys())}")
        
        self.strategy = strategy
        self.selector = self.STRATEGIES[strategy]()
    
    @torch.no_grad()
    def select(
        self, 
        tokens: torch.Tensor, 
        retention: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select most informative tokens.
        
        Args:
            tokens: [B, N, D] feature tokens from vision encoder
            retention: fraction of tokens to keep (0.0 to 1.0)
            
        Returns:
            kept_tokens: [B, K, D] selected tokens where K = int(N * retention)
            mask: [B, N] binary mask (1 = kept, 0 = discarded)
        """
        if not 0.0 < retention <= 1.0:
            raise ValueError(f"retention must be in (0, 1], got {retention}")
        
        return self.selector.select(tokens, retention)
    
    @torch.no_grad()
    def get_importance_scores(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Get importance scores for all tokens without selection.
        
        Args:
            tokens: [B, N, D] feature tokens
            
        Returns:
            scores: [B, N] importance scores (higher = more important)
        """
        B, N, D = tokens.shape
        
        if self.strategy == 'variance':
            return tokens.var(dim=-1)
        elif self.strategy == 'norm':
            return torch.norm(tokens, p=2, dim=-1)
        elif self.strategy == 'entropy':
            tokens_norm = (tokens - tokens.min(dim=-1, keepdim=True)[0])
            tokens_norm = tokens_norm / (tokens_norm.max(dim=-1, keepdim=True)[0] + 1e-8)
            eps = 1e-8
            return -(tokens_norm * torch.log(tokens_norm + eps)).sum(dim=-1)
        else:
            # For combined/spatial, recompute the combined score
            variance = tokens.var(dim=-1)
            norms = torch.norm(tokens, p=2, dim=-1)
            
            variance_norm = (variance - variance.min(dim=1, keepdim=True)[0])
            variance_norm = variance_norm / (variance_norm.max(dim=1, keepdim=True)[0] + 1e-8)
            
            norms_norm = (norms - norms.min(dim=1, keepdim=True)[0])
            norms_norm = norms_norm / (norms_norm.max(dim=1, keepdim=True)[0] + 1e-8)
            
            return 0.5 * variance_norm + 0.5 * norms_norm

