#!/usr/bin/env python3
"""
Comprehensive test suite for Heuristic-based ATR implementation.

This test suite validates all components of the ATR system without
requiring any trained weights or datasets.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile

# Add geometry module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_heuristic_selectors():
    """Test all heuristic selector strategies."""
    print("=" * 70)
    print("TEST 1: Heuristic Selectors")
    print("=" * 70)
    
    try:
        from atr.heuristic_selector import (
            VarianceBasedSelector,
            NormBasedSelector,
            EntropyBasedSelector,
            CombinedSaliencySelector,
            SpatialAwareSelector,
            HeuristicATRSelector
        )
        
        print("✓ All selector classes imported successfully\n")
        
        # Create dummy tokens
        B, N, D = 1, 256, 768  # Batch=1, 256 tokens (16x16), 768 dims (CLIP-L)
        tokens = torch.randn(B, N, D)
        retention = 0.3
        
        selectors = {
            'Variance': VarianceBasedSelector(),
            'Norm': NormBasedSelector(),
            'Entropy': EntropyBasedSelector(),
            'Combined': CombinedSaliencySelector(),
            'Spatial': SpatialAwareSelector(),
        }
        
        for name, selector in selectors.items():
            print(f"Testing {name}Selector...")
            kept_tokens, mask = selector.select(tokens, retention)
            
            expected_k = int(N * retention)
            actual_k = kept_tokens.shape[1]
            
            # Validate shapes
            assert kept_tokens.shape == (B, expected_k, D), \
                f"Wrong kept_tokens shape: {kept_tokens.shape}, expected {(B, expected_k, D)}"
            assert mask.shape == (B, N), \
                f"Wrong mask shape: {mask.shape}, expected {(B, N)}"
            
            # Validate mask values
            assert torch.all((mask == 0) | (mask == 1)), "Mask should be binary"
            assert mask.sum().item() == expected_k, \
                f"Mask sum {mask.sum().item()} != expected {expected_k}"
            
            print(f"  ✓ Shape validation passed")
            print(f"  ✓ Selected {actual_k}/{N} tokens ({retention*100:.0f}%)")
            print(f"  ✓ Mask is binary and correct\n")
        
        # Test unified interface
        print("Testing HeuristicATRSelector unified interface...")
        for strategy in ['variance', 'norm', 'entropy', 'combined', 'spatial']:
            selector = HeuristicATRSelector(strategy=strategy)
            kept_tokens, mask = selector.select(tokens, retention)
            scores = selector.get_importance_scores(tokens)
            
            assert scores.shape == (B, N), f"Importance scores shape mismatch: {scores.shape}"
            print(f"  ✓ Strategy '{strategy}' works correctly")
        
        print("\n✅ All selector tests PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Selector tests FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_clip_feature_extraction():
    """Test CLIP feature extraction (if transformers available)."""
    print("=" * 70)
    print("TEST 2: CLIP Feature Extraction")
    print("=" * 70)
    
    try:
        from atr.clip_adapter import CLIPFeatureExtractor
        
        # Create dummy image
        dummy_img = Image.new('RGB', (224, 224), color='white')
        
        print("Initializing CLIP feature extractor...")
        try:
            extractor = CLIPFeatureExtractor()
            print("✓ CLIP model loaded successfully\n")
            
            print("Extracting features from dummy image...")
            tokens = extractor(dummy_img, device='cpu')
            
            print(f"  Token shape: {tokens.shape}")
            print(f"  Expected shape: [1, N, D] where N=196 or 256, D=768 or 1024")
            
            B, N, D = tokens.shape
            assert B == 1, f"Batch size should be 1, got {B}"
            assert N > 0, f"Number of tokens should be > 0, got {N}"
            assert D > 0, f"Feature dimension should be > 0, got {D}"
            
            print(f"  ✓ Shape validation passed")
            print(f"  ✓ Extracted {N} tokens with {D} dimensions\n")
            
            print("✅ CLIP feature extraction test PASSED\n")
            return True
            
        except ImportError as e:
            print(f"⚠️  CLIP test skipped (transformers not installed): {e}")
            print("    Install with: pip install transformers\n")
            return True  # Not a failure, just skipped
            
    except Exception as e:
        print(f"\n❌ CLIP test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_image_preprocessing():
    """Test end-to-end image preprocessing."""
    print("=" * 70)
    print("TEST 3: Image Preprocessing")
    print("=" * 70)
    
    try:
        from atr.preprocess_heuristic import preprocess_image_with_atr
        
        # Create test image
        test_img = Image.new('RGB', (400, 300), color=(128, 128, 128))
        # Add some content (white square in center)
        import numpy as np
        img_array = np.array(test_img)
        img_array[100:200, 150:250] = [255, 255, 255]
        test_img = Image.fromarray(img_array)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_img.save(tmp.name)
            temp_path = tmp.name
        
        try:
            print("Testing preprocessing with different strategies...\n")
            
            strategies = ['variance', 'norm', 'combined']
            
            for strategy in strategies:
                print(f"  Strategy: {strategy}")
                try:
                    output_path = preprocess_image_with_atr(
                        image_path=temp_path,
                        retention=0.3,
                        crop=False,
                        strategy=strategy,
                        device='cpu'
                    )
                    
                    assert os.path.exists(output_path), f"Output file not created: {output_path}"
                    
                    # Verify output is a valid image
                    output_img = Image.open(output_path)
                    assert output_img.mode == 'RGB', "Output should be RGB"
                    
                    print(f"    ✓ Generated: {Path(output_path).name}")
                    print(f"    ✓ Output size: {output_img.size}")
                    
                    # Clean up
                    os.unlink(output_path)
                    
                except ImportError as e:
                    if 'transformers' in str(e) or 'CLIP' in str(e):
                        print(f"    ⚠️  Skipped (CLIP not available)")
                        continue
                    raise
            
            print("\n✅ Image preprocessing test PASSED\n")
            return True
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
    except Exception as e:
        print(f"\n❌ Preprocessing test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_integration_helpers():
    """Test integration helper functions."""
    print("=" * 70)
    print("TEST 4: Integration Helpers")
    print("=" * 70)
    
    try:
        from atr.integration import (
            apply_atr_if_enabled,
            get_atr_config_from_env,
            set_atr_config,
            print_atr_status
        )
        
        print("Testing configuration functions...\n")
        
        # Test set_atr_config
        print("  Setting ATR config...")
        set_atr_config(
            enable=True,
            retention=0.5,
            crop=True,
            strategy='spatial'
        )
        
        # Test get_atr_config_from_env
        print("  Reading ATR config...")
        config = get_atr_config_from_env()
        
        assert config['atr_enable'] == True, "Config enable mismatch"
        assert config['atr_retention'] == 0.5, "Config retention mismatch"
        assert config['atr_crop'] == True, "Config crop mismatch"
        assert config['atr_strategy'] == 'spatial', "Config strategy mismatch"
        
        print("  ✓ Configuration read/write works correctly\n")
        
        # Test print_atr_status
        print("  Testing status display...")
        print_atr_status()
        print("  ✓ Status display works\n")
        
        # Reset to defaults
        set_atr_config(enable=False)
        
        print("✅ Integration helpers test PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processing():
    """Test batch image processing."""
    print("=" * 70)
    print("TEST 5: Batch Processing")
    print("=" * 70)
    
    try:
        from atr.preprocess_heuristic import batch_preprocess_images
        
        # Create multiple test images
        temp_dir = tempfile.mkdtemp()
        temp_images = []
        
        try:
            for i in range(3):
                img = Image.new('RGB', (200, 200), color=(i*80, i*80, i*80))
                img_path = os.path.join(temp_dir, f"test_image_{i}.png")
                img.save(img_path)
                temp_images.append(img_path)
            
            print(f"Created {len(temp_images)} test images\n")
            
            # Try batch processing (may fail if CLIP not available)
            try:
                output_paths = batch_preprocess_images(
                    image_paths=temp_images,
                    retention=0.3,
                    strategy='combined',
                    output_dir=temp_dir
                )
                
                print(f"\n✓ Processed {len(output_paths)} images")
                
                # Verify outputs
                for out_path in output_paths:
                    if out_path and os.path.exists(out_path):
                        print(f"  ✓ Generated: {Path(out_path).name}")
                
                print("\n✅ Batch processing test PASSED\n")
                return True
                
            except ImportError as e:
                if 'transformers' in str(e) or 'CLIP' in str(e):
                    print("⚠️  Batch processing test skipped (CLIP not available)\n")
                    return True
                raise
                
        finally:
            # Clean up
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"\n❌ Batch processing test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_retention_variations():
    """Test different retention ratios."""
    print("=" * 70)
    print("TEST 6: Retention Ratio Variations")
    print("=" * 70)
    
    try:
        from atr.heuristic_selector import HeuristicATRSelector
        
        B, N, D = 1, 196, 768
        tokens = torch.randn(B, N, D)
        
        selector = HeuristicATRSelector(strategy='combined')
        
        retentions = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        
        print("Testing different retention ratios...\n")
        
        for retention in retentions:
            kept_tokens, mask = selector.select(tokens, retention)
            expected_k = max(1, int(N * retention))
            actual_k = kept_tokens.shape[1]
            
            assert actual_k == expected_k, \
                f"Retention {retention}: got {actual_k} tokens, expected {expected_k}"
            
            print(f"  Retention {retention*100:>4.0f}%: {actual_k:>3}/{N} tokens kept ✓")
        
        print("\n✅ Retention variation test PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Retention test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and generate report."""
    print("\n" + "="*70)
    print("HEURISTIC ATR TEST SUITE")
    print("="*70)
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    try:
        import torch
        print(f"  ✓ PyTorch: {torch.__version__}")
    except ImportError:
        print("  ✗ PyTorch not found - please install: pip install torch")
        return
    
    try:
        import PIL
        print(f"  ✓ Pillow: {PIL.__version__}")
    except ImportError:
        print("  ✗ Pillow not found - please install: pip install pillow")
        return
    
    try:
        import transformers
        print(f"  ✓ Transformers: {transformers.__version__}")
    except ImportError:
        print("  ⚠️  Transformers not found (CLIP tests will be skipped)")
        print("      Install with: pip install transformers")
    
    print()
    
    # Run tests
    tests = [
        ("Heuristic Selectors", test_heuristic_selectors),
        ("CLIP Feature Extraction", test_clip_feature_extraction),
        ("Image Preprocessing", test_image_preprocessing),
        ("Integration Helpers", test_integration_helpers),
        ("Batch Processing", test_batch_processing),
        ("Retention Variations", test_retention_variations),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed with exception: {e}\n")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test_name}")
    
    print()
    print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nThe heuristic-based ATR implementation is working correctly.")
        print("\nQuick Start:")
        print("  1. Enable ATR: export ATR_ENABLE=true")
        print("  2. Set retention: export ATR_RETENTION=0.3")
        print("  3. Choose strategy: export ATR_STRATEGY=combined")
        print("  4. Run solver: python geometry/solver.py")
        print()
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Please check the error messages above.")
        print()


if __name__ == "__main__":
    run_all_tests()

