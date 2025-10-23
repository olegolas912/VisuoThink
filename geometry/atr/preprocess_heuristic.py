"""
Heuristic-based Image Preprocessing with Adaptive Token Reduction.

This module provides image preprocessing using heuristic token selection
(no training required) instead of the learned selector approach from the paper.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Literal
from PIL import Image

from .clip_adapter import CLIPFeatureExtractor
from .heuristic_selector import HeuristicATRSelector


def _infer_grid(num_tokens: int) -> Tuple[int, int]:
    """
    Infer spatial grid dimensions from number of tokens.
    
    Args:
        num_tokens: Number of visual tokens
        
    Returns:
        (height, width) of the spatial grid
    """
    # Try square grid first
    side = int(round(num_tokens ** 0.5))
    if side * side == num_tokens:
        return side, side
    
    # Try rectangular factors
    for h in range(1, side + 2):
        if num_tokens % h == 0:
            w = num_tokens // h
            return h, w
    
    # Fallback
    return side, side


@torch.no_grad()
def preprocess_image_with_atr(
    image_path: str,
    retention: float = 0.3,
    crop: bool = False,
    device: Optional[str] = None,
    strategy: Literal['variance', 'norm', 'entropy', 'combined', 'spatial'] = 'combined',
    model_name: str = "openai/clip-vit-large-patch14",
    output_path: Optional[str] = None,
    visualize: bool = False,
) -> str:
    """
    Apply heuristic-based ATR to produce a token-reduced image.
    
    This function:
    1. Loads the image
    2. Extracts CLIP visual tokens
    3. Selects important tokens using heuristic scoring
    4. Creates a spatial mask from selected tokens
    5. Applies mask to image (darken or crop background)
    6. Saves the processed image
    
    Args:
        image_path: Path to input image
        retention: Fraction of tokens to keep (0.0 to 1.0)
            - 0.1: Very aggressive (10% tokens)
            - 0.3: Balanced (30% tokens) - RECOMMENDED
            - 0.5: Conservative (50% tokens)
        crop: If True, crop to bounding box of selected tokens
              If False, darken background (set to black)
        device: Device to use ('cuda', 'cpu', or None for auto)
        strategy: Token selection strategy
            - 'variance': High variance tokens
            - 'norm': High L2 norm tokens
            - 'entropy': High entropy tokens
            - 'combined': Weighted combination (RECOMMENDED)
            - 'spatial': Combined with spatial diversity
        model_name: CLIP model to use for feature extraction
        output_path: Output path (default: original_name.atr.png)
        visualize: If True, also save visualization of selected tokens
        
    Returns:
        Path to the processed image
        
    Example:
        >>> output = preprocess_image_with_atr(
        ...     "diagram.png",
        ...     retention=0.3,
        ...     crop=True,
        ...     strategy='combined'
        ... )
        >>> # Output saved to "diagram.atr.png"
    """
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load image
    pil_image = Image.open(image_path).convert("RGB")
    original_size = pil_image.size
    
    # Extract CLIP features
    try:
        extractor = CLIPFeatureExtractor(model_name=model_name)
        tokens = extractor(pil_image, device=device)  # [1, N, D]
    except Exception as e:
        print(f"Warning: CLIP feature extraction failed: {e}")
        print("Falling back to basic image processing...")
        # Fallback: just return the original image
        output_path = output_path or os.path.splitext(image_path)[0] + ".atr.png"
        pil_image.save(output_path)
        return output_path
    
    B, N, D = tokens.shape
    
    # Select important tokens using heuristic
    selector = HeuristicATRSelector(strategy=strategy)
    kept_tokens, mask = selector.select(tokens, retention=retention)
    
    # Convert mask to numpy
    mask_np = mask[0].cpu().numpy()  # [N]
    
    # Infer spatial grid
    Ht, Wt = _infer_grid(N)
    
    try:
        mask_grid = mask_np.reshape(Ht, Wt)
    except ValueError:
        print(f"Warning: Cannot reshape {N} tokens to grid. Using original image.")
        output_path = output_path or os.path.splitext(image_path)[0] + ".atr.png"
        pil_image.save(output_path)
        return output_path
    
    # Upscale mask to original image size
    mask_img = Image.fromarray((mask_grid > 0.5).astype(np.uint8) * 255)
    mask_img = mask_img.resize(original_size, resample=Image.NEAREST)
    mask_full = np.array(mask_img).astype(np.uint8)
    
    # Apply mask to image
    img_np = np.array(pil_image)
    
    if crop:
        # Find bounding box of selected regions
        ys, xs = np.where(mask_full > 0)
        
        if ys.size > 0 and xs.size > 0:
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            
            # Add padding
            pad = int(0.02 * max(original_size))
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(img_np.shape[1] - 1, x1 + pad)
            y1 = min(img_np.shape[0] - 1, y1 + pad)
            
            # Crop to bounding box
            masked_img = img_np[y0:y1+1, x0:x1+1]
        else:
            # No tokens selected, return original
            masked_img = img_np
    else:
        # Darken background (set to black)
        background = np.zeros_like(img_np)
        masked_img = np.where(mask_full[..., None] > 0, img_np, background)
    
    # Save processed image
    output_path = output_path or os.path.splitext(image_path)[0] + ".atr.png"
    Image.fromarray(masked_img).save(output_path)
    
    # Optionally save visualization
    if visualize:
        viz_path = os.path.splitext(output_path)[0] + ".viz.png"
        _save_visualization(pil_image, mask_grid, tokens, selector, retention, viz_path)
    
    return output_path


def _save_visualization(
    original_image: Image.Image,
    mask_grid: np.ndarray,
    tokens: torch.Tensor,
    selector: HeuristicATRSelector,
    retention: float,
    output_path: str
) -> None:
    """
    Save a visualization showing original image, mask, and importance scores.
    
    Args:
        original_image: Original PIL image
        mask_grid: Binary mask in grid format [H, W]
        tokens: Feature tokens [1, N, D]
        selector: The selector used
        retention: Retention ratio
        output_path: Where to save visualization
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization")
        return
    
    # Get importance scores
    importance = selector.get_importance_scores(tokens)[0].cpu().numpy()  # [N]
    
    Ht, Wt = mask_grid.shape
    importance_grid = importance.reshape(Ht, Wt)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Importance heatmap
    im = axes[1].imshow(importance_grid, cmap='hot', interpolation='nearest')
    axes[1].set_title(f"Token Importance Scores\n({selector.strategy} strategy)")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Selected tokens mask
    axes[2].imshow(mask_grid, cmap='gray', interpolation='nearest')
    axes[2].set_title(f"Selected Tokens\n(Retention: {retention*100:.0f}%)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def batch_preprocess_images(
    image_paths: list,
    retention: float = 0.3,
    crop: bool = False,
    device: Optional[str] = None,
    strategy: str = 'combined',
    model_name: str = "openai/clip-vit-large-patch14",
    output_dir: Optional[str] = None,
    visualize: bool = False,
) -> list:
    """
    Batch process multiple images with ATR.
    
    Args:
        image_paths: List of input image paths
        retention: Fraction of tokens to keep
        crop: Whether to crop or darken background
        device: Device to use
        strategy: Token selection strategy
        model_name: CLIP model name
        output_dir: Output directory (default: same as input)
        visualize: Whether to create visualizations
        
    Returns:
        List of output paths
    """
    output_paths = []
    
    print(f"Processing {len(image_paths)} images with ATR...")
    print(f"  Strategy: {strategy}")
    print(f"  Retention: {retention*100:.0f}%")
    print(f"  Crop: {crop}")
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"  [{i}/{len(image_paths)}] {Path(image_path).name}...", end=' ')
        
        try:
            if output_dir:
                output_path = os.path.join(
                    output_dir,
                    Path(image_path).stem + ".atr.png"
                )
            else:
                output_path = None
            
            result_path = preprocess_image_with_atr(
                image_path=image_path,
                retention=retention,
                crop=crop,
                device=device,
                strategy=strategy,
                model_name=model_name,
                output_path=output_path,
                visualize=visualize,
            )
            
            output_paths.append(result_path)
            print("✓")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            output_paths.append(None)
    
    success_count = sum(1 for p in output_paths if p is not None)
    print(f"\nCompleted: {success_count}/{len(image_paths)} images processed successfully")
    
    return output_paths


def get_token_importance_map(
    image_path: str,
    strategy: str = 'combined',
    device: Optional[str] = None,
    model_name: str = "openai/clip-vit-large-patch14",
) -> np.ndarray:
    """
    Get importance scores for all tokens without applying reduction.
    Useful for analysis and debugging.
    
    Args:
        image_path: Path to input image
        strategy: Token selection strategy
        device: Device to use
        model_name: CLIP model name
        
    Returns:
        importance_map: [H, W] array of importance scores
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load and extract features
    pil_image = Image.open(image_path).convert("RGB")
    extractor = CLIPFeatureExtractor(model_name=model_name)
    tokens = extractor(pil_image, device=device)  # [1, N, D]
    
    # Get importance scores
    selector = HeuristicATRSelector(strategy=strategy)
    importance = selector.get_importance_scores(tokens)[0].cpu().numpy()  # [N]
    
    # Reshape to grid
    N = len(importance)
    Ht, Wt = _infer_grid(N)
    importance_map = importance.reshape(Ht, Wt)
    
    return importance_map

