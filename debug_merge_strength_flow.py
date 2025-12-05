#!/usr/bin/env python3
"""
Debug script to test if merge_strength parameter flows correctly
"""

import sys
import os

# Add the project directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from shared.cache_management import compute_merge_hash, analyze_hash_differences
    from shared.merge_strength import _compatibility_to_merge_strength
except ImportError:
    print("Could not import modules - run this from the ComfyUI-DonutNodes directory")
    sys.exit(1)

def test_merge_strength_flow():
    """Test if different merge_strength values produce different hashes and results"""
    
    print("="*60)
    print("DEBUGGING MERGE_STRENGTH FLOW")
    print("="*60)
    
    # Create dummy models (same as cache debug test)
    test_models = []
    for i in range(2):
        class _TestModel:
            def state_dict(self): return {}
            def named_parameters(self): return iter([])
        m = _TestModel()
        setattr(m, "_is_filler", True)
        test_models.append(m)
    
    # Test parameters - identical except merge_strength
    base_params = {
        'min_strength': 0.0,
        'max_strength': 1.0, 
        'importance_threshold': 1.0,
        'importance_boost': 1.0,
        'rank_sensitivity': 1.0,
        'skip_threshold': 0.0,
        'normalization_mode': 'magnitude_enhanced_widen',
        'lora_stack': None
    }
    
    # Test different merge_strength values
    test_values = [0.1, 0.5, 1.0]
    
    print("\n1. TESTING CACHE KEY GENERATION:")
    print("-" * 40)
    
    hashes = {}
    for merge_strength in test_values:
        hash_key = compute_merge_hash(test_models, merge_strength, **base_params)
        hashes[merge_strength] = hash_key
        print(f"merge_strength={merge_strength}: {hash_key}")
    
    # Check if hashes are different
    unique_hashes = set(hashes.values())
    print(f"\nUnique hashes: {len(unique_hashes)} out of {len(test_values)}")
    
    if len(unique_hashes) == len(test_values):
        print("✅ GOOD: Different merge_strength values produce different cache keys")
    else:
        print("❌ BUG: Same cache keys for different merge_strength values!")
        print("This means cache will return identical results!")
    
    print("\n2. TESTING STRENGTH CALCULATION:")
    print("-" * 40)
    
    # Test the strength calculation function directly
    test_compatibility = 0.01  # Typical compatibility score
    
    for merge_strength in test_values:
        strength = _compatibility_to_merge_strength(
            compatibility_score=test_compatibility,
            merge_strength=merge_strength,
            min_strength=base_params['min_strength'],
            max_strength=base_params['max_strength'],
            sensitivity=base_params['rank_sensitivity']
        )
        print(f"merge_strength={merge_strength}, compatibility={test_compatibility} → final_strength={strength:.6f}")
    
    print("\n3. DETAILED HASH COMPARISON:")
    print("-" * 40)
    
    # Compare 0.1 vs 0.5 specifically
    params_01 = (0.1,) + tuple(base_params.values())
    params_05 = (0.5,) + tuple(base_params.values())
    
    hash1, hash2, differences = analyze_hash_differences(test_models, params_01, test_models, params_05)
    
    print(f"Hash for 0.1: {hash1}")
    print(f"Hash for 0.5: {hash2}")
    print(f"Hashes different: {hash1 != hash2}")
    
    if differences:
        print("Parameter differences:")
        for diff in differences:
            print(f"  {diff}")
    else:
        print("❌ NO PARAMETER DIFFERENCES DETECTED!")

if __name__ == "__main__":
    test_merge_strength_flow()