#!/usr/bin/env python3
"""
Test script to verify that the cache invalidation fix works correctly.
This tests that changing scale_to_min_max and invert_strengths toggles 
produces different cache keys.
"""

import sys
import os

# Add the current directory to Python path to find the shared modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from shared.cache_management import compute_merge_hash

def test_toggle_cache_invalidation():
    """Test that changing toggle parameters invalidates the cache"""
    print("Testing cache invalidation with toggle parameters...")
    
    # Create dummy models for testing
    test_models = []
    for i in range(2):
        class _TestModel:
            def state_dict(self): return {}
            def named_parameters(self): return iter([])
        m = _TestModel()
        setattr(m, "_is_filler", True)
        test_models.append(m)
    
    # Base parameters
    base_params = {
        'merge_strength': 1.0,
        'min_strength': 0.0,
        'max_strength': 1.0,
        'importance_threshold': 1.0,
        'importance_boost': 1.0,
        'rank_sensitivity': 2.0,
        'skip_threshold': 0.0,
        'normalization_mode': 'magnitude',
        'lora_stack': None
    }
    
    # Test 1: Both toggles False (default)
    hash1 = compute_merge_hash(test_models, **base_params, scale_to_min_max=False, invert_strengths=False)
    print(f"Hash 1 (both False): {hash1}")
    
    # Test 2: scale_to_min_max True, invert_strengths False
    hash2 = compute_merge_hash(test_models, **base_params, scale_to_min_max=True, invert_strengths=False)
    print(f"Hash 2 (scale_to_min_max=True): {hash2}")
    
    # Test 3: scale_to_min_max False, invert_strengths True
    hash3 = compute_merge_hash(test_models, **base_params, scale_to_min_max=False, invert_strengths=True)
    print(f"Hash 3 (invert_strengths=True): {hash3}")
    
    # Test 4: Both toggles True
    hash4 = compute_merge_hash(test_models, **base_params, scale_to_min_max=True, invert_strengths=True)
    print(f"Hash 4 (both True): {hash4}")
    
    # Verify all hashes are different
    hashes = [hash1, hash2, hash3, hash4]
    unique_hashes = set(hashes)
    
    print(f"\nResults:")
    print(f"Total hashes: {len(hashes)}")
    print(f"Unique hashes: {len(unique_hashes)}")
    print(f"Cache invalidation working: {len(hashes) == len(unique_hashes)}")
    
    if len(hashes) == len(unique_hashes):
        print("✅ SUCCESS: All toggle combinations produce different cache keys!")
        return True
    else:
        print("❌ FAILURE: Some toggle combinations produce identical cache keys!")
        print("This means cache invalidation is not working properly.")
        return False

def test_identical_parameters():
    """Test that identical parameters still produce identical hashes"""
    print("\nTesting that identical parameters produce identical hashes...")
    
    # Create dummy models for testing
    test_models = []
    for i in range(2):
        class _TestModel:
            def state_dict(self): return {}
            def named_parameters(self): return iter([])
        m = _TestModel()
        setattr(m, "_is_filler", True)
        test_models.append(m)
    
    # Base parameters
    base_params = {
        'merge_strength': 1.0,
        'min_strength': 0.0,
        'max_strength': 1.0,
        'importance_threshold': 1.0,
        'importance_boost': 1.0,
        'rank_sensitivity': 2.0,
        'skip_threshold': 0.0,
        'normalization_mode': 'magnitude',
        'lora_stack': None,
        'scale_to_min_max': True,
        'invert_strengths': False
    }
    
    # Generate two identical hashes
    hash1 = compute_merge_hash(test_models, **base_params)
    hash2 = compute_merge_hash(test_models, **base_params)
    
    print(f"Hash 1: {hash1}")
    print(f"Hash 2: {hash2}")
    print(f"Identical: {hash1 == hash2}")
    
    if hash1 == hash2:
        print("✅ SUCCESS: Identical parameters produce identical hashes!")
        return True
    else:
        print("❌ FAILURE: Identical parameters produce different hashes!")
        return False

if __name__ == "__main__":
    print("="*60)
    print("CACHE INVALIDATION FIX TEST")
    print("="*60)
    
    test1_passed = test_toggle_cache_invalidation()
    test2_passed = test_identical_parameters()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Toggle differentiation test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Identical parameters test: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! Cache invalidation fix is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 SOME TESTS FAILED! Cache invalidation fix needs more work.")
        sys.exit(1)