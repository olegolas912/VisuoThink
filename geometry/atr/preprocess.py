"""
DEPRECATED: This module uses the learned selector (requires training).
Please use preprocess_heuristic.py instead for the heuristic-based approach.

This file is kept for backward compatibility but will use the heuristic
approach as a fallback since no trained weights are available.
"""

import os
import warnings
from typing import Optional

# Import the heuristic-based implementation
from .preprocess_heuristic import preprocess_image_with_atr as _preprocess_heuristic


def preprocess_image_with_atr(
    image_path: str,
    retention: float = 0.3,
    crop: bool = False,
    device: Optional[str] = None,
) -> str:
    """
    Apply ATR preprocessing to an image.
    
    DEPRECATED: This function now uses the heuristic-based approach instead
    of the learned selector (which requires training).
    
    For new code, please use:
        from atr.preprocess_heuristic import preprocess_image_with_atr
    
    Args:
        image_path: Path to input image
        retention: Fraction of tokens to keep (0.0 to 1.0)
        crop: If True, crop to bounding box; if False, darken background
        device: Device to use ('cuda', 'cpu', or None for auto)
        
    Returns:
        Path to the processed image (with .atr.png extension)
    """
    warnings.warn(
        "preprocess.preprocess_image_with_atr is deprecated. "
        "Using heuristic-based approach instead. "
        "For new code, use: from atr.preprocess_heuristic import preprocess_image_with_atr",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Use heuristic-based implementation with 'combined' strategy
    return _preprocess_heuristic(
        image_path=image_path,
        retention=retention,
        crop=crop,
        device=device,
        strategy='combined'  # Use combined strategy by default
    )


