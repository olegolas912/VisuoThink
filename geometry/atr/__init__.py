"""
Adaptive Token Reduction (ATR) package.

This package provides two approaches:

1. HEURISTIC-BASED (RECOMMENDED - No training required):
   - Uses visual importance metrics (variance, norm, entropy, etc.)
   - Works out-of-the-box without any training
   - Fast and effective for most use cases
   
2. LEARNED (Paper-compliant - Requires training):
   - Feature Selector: transformer blocks + Gumbel-Softmax
   - Feature Reconstructor: transformer blocks to reconstruct full token set
   - Requires training on 100K COCO images (as per paper)
   - Currently NOT RECOMMENDED (no pretrained weights available)

For most users, use the heuristic-based approach:
    from atr import preprocess_image_with_atr
    output = preprocess_image_with_atr("image.png", retention=0.3)
"""

# Heuristic-based approach (RECOMMENDED)
from .heuristic_selector import (
    HeuristicATRSelector,
    VarianceBasedSelector,
    NormBasedSelector,
    EntropyBasedSelector,
    CombinedSaliencySelector,
    SpatialAwareSelector,
)
from .preprocess_heuristic import (
    preprocess_image_with_atr,
    batch_preprocess_images,
    get_token_importance_map,
)

# Learned approach (requires training - not recommended without weights)
from .modules import (
    GumbelBinarySampler,
    TransformerFeatureSelector,
    TransformerFeatureReconstructor,
    AutoencoderSelector,
)
from .clip_adapter import CLIPFeatureExtractor, ATRClipWrapper

__all__ = [
    # Main interface (heuristic-based)
    "preprocess_image_with_atr",
    "batch_preprocess_images",
    "get_token_importance_map",
    "HeuristicATRSelector",
    
    # Heuristic selectors
    "VarianceBasedSelector",
    "NormBasedSelector",
    "EntropyBasedSelector",
    "CombinedSaliencySelector",
    "SpatialAwareSelector",
    
    # Learned components (for research/training)
    "GumbelBinarySampler",
    "TransformerFeatureSelector",
    "TransformerFeatureReconstructor",
    "AutoencoderSelector",
    "CLIPFeatureExtractor",
    "ATRClipWrapper",
]


