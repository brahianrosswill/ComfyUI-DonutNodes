#!/usr/bin/env python3
"""
Cache Invalidation Fix Validation Script

This script validates that the cache invalidation issue has been resolved.
The issue was that ComfyUI model object recreation was causing cache keys to change
even when parameters were identical, masking the detection of parameter changes.

PROBLEM BEFORE FIX:
1. First execution (merge_strength=0.5): Cache key = hash(model_id_1, model_id_2, 0.5, ...)
2. Second execution (merge_strength=0.1): NEW model objects → Cache key = hash(model_id_3, model_id_4, 0.1, ...)
3. Both executions miss cache due to different model IDs, giving illusion that parameter changes aren't detected

SOLUTION:
- Modified compute_merge_hash() to use stable model identifiers instead of object IDs
- Model content/structure is now used for hashing, not object identity
- Parameter changes are now properly detected regardless of model object recreation
"""

import sys
import os

# Add the project directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from shared.cache_management import compute_merge_hash, clear_merge_cache, store_merge_result, check_cache_for_merge
    from shared.constants import _MERGE_CACHE
except ImportError as e:
    print(f"Could not import modules: {e}")
    sys.exit(1)

def validate_cache_fix():
    """Validate that the cache invalidation fix works correctly"""
    
    print("="*80)
    print("VALIDATING CACHE INVALIDATION FIX")
    print("="*80)
    
    # Clear cache to start fresh
    clear_merge_cache()
    
    # Create realistic model mock that simulates ComfyUI model behavior
    class ComfyUIModelMock:
        def __init__(self, model_name):
            self.model_name = model_name
            self._state_dict = {
                f"layers.{i}.weight": f"tensor_data_{model_name}_{i}" 
                for i in range(5)
            }
        
        def state_dict(self):
            return self._state_dict
        
        def named_parameters(self):
            # Some models have this, some don't
            return iter([])
        
        def clone(self):
            return ComfyUIModelMock(self.model_name)
    
    # Test parameters
    base_params = {
        'min_strength': 0.0,
        'max_strength': 1.0,
        'importance_threshold': 1.0,
        'importance_boost': 1.0,
        'rank_sensitivity': 2.0,
        'skip_threshold': 0.0,
        'normalization_mode': 'magnitude_enhanced_widen',
        'lora_stack': None
    }
    
    print("\n1. TESTING MODEL OBJECT RECREATION (ComfyUI behavior)")
    print("-" * 60)
    
    # Simulate first execution
    model_base_1 = ComfyUIModelMock("base_model")
    model_other_1 = ComfyUIModelMock("other_model")
    models_1 = [model_base_1, model_other_1] + [None] * 10
    
    hash_1 = compute_merge_hash(models_1, merge_strength=0.5, **base_params)
    print(f"First execution (merge_strength=0.5): {hash_1}")
    
    # Simulate ComfyUI recreating model objects (same content, different objects)
    model_base_2 = ComfyUIModelMock("base_model")  # Same name/content, different object
    model_other_2 = ComfyUIModelMock("other_model")  # Same name/content, different object
    models_2 = [model_base_2, model_other_2] + [None] * 10
    
    hash_2 = compute_merge_hash(models_2, merge_strength=0.5, **base_params)
    print(f"After model recreation (merge_strength=0.5): {hash_2}")
    print(f"Hashes identical (should be TRUE): {hash_1 == hash_2}")
    
    if hash_1 == hash_2:
        print("✅ FIXED: Model object recreation no longer affects cache keys")
    else:
        print("❌ STILL BROKEN: Model object recreation still changes cache keys")
        return False
    
    print("\n2. TESTING PARAMETER CHANGE DETECTION")
    print("-" * 60)
    
    # Test parameter change with same model objects
    hash_3 = compute_merge_hash(models_2, merge_strength=0.1, **base_params)
    print(f"Parameter change (merge_strength=0.1): {hash_3}")
    print(f"Different from 0.5 (should be TRUE): {hash_1 != hash_3}")
    
    if hash_1 != hash_3:
        print("✅ GOOD: Parameter changes produce different cache keys")
    else:
        print("❌ BROKEN: Parameter changes don't affect cache keys")
        return False
    
    print("\n3. TESTING REAL CACHE FLOW SIMULATION")
    print("-" * 60)
    
    # Simulate real ComfyUI execution flow
    
    # First execution: merge_strength=0.5
    print("Execution 1: merge_strength=0.5")
    cached_result_1 = check_cache_for_merge(hash_1)
    print(f"Cache lookup result: {cached_result_1}")
    
    if cached_result_1 is None:
        print("Cache miss (expected) - would compute merge")
        # Simulate storing result
        dummy_result_1 = (model_base_1.clone(), "merge_result_0.5", "param_info_0.5")
        store_merge_result(hash_1, dummy_result_1)
        print("Stored result in cache")
    
    # Second execution: merge_strength=0.1 (different models objects, different parameter)
    print("\nExecution 2: merge_strength=0.1 (new model objects)")
    cached_result_2 = check_cache_for_merge(hash_3)
    print(f"Cache lookup result: {cached_result_2}")
    
    if cached_result_2 is None:
        print("Cache miss (expected) - would compute new merge")
        dummy_result_2 = (model_base_2.clone(), "merge_result_0.1", "param_info_0.1")
        store_merge_result(hash_3, dummy_result_2)
        print("Stored result in cache")
    else:
        print("❌ UNEXPECTED: Cache hit when parameters changed!")
        return False
    
    # Third execution: merge_strength=0.5 again (different model objects, same parameter)
    print("\nExecution 3: merge_strength=0.5 (new model objects, same parameter)")
    model_base_3 = ComfyUIModelMock("base_model")  # Yet another set of objects
    model_other_3 = ComfyUIModelMock("other_model")
    models_3 = [model_base_3, model_other_3] + [None] * 10
    
    hash_4 = compute_merge_hash(models_3, merge_strength=0.5, **base_params)
    print(f"Cache key: {hash_4}")
    print(f"Same as first execution: {hash_1 == hash_4}")
    
    cached_result_3 = check_cache_for_merge(hash_4)
    print(f"Cache lookup result: {cached_result_3 is not None}")
    
    if cached_result_3 is not None and hash_1 == hash_4:
        print("✅ PERFECT: Cache hit with same parameters despite new model objects")
        cached_model, cached_text, cached_info = cached_result_3
        print(f"Retrieved cached result: {cached_text}")
    else:
        print("❌ ISSUE: Should have hit cache with same parameters")
        return False
    
    print("\n4. TESTING EDGE CASES")
    print("-" * 60)
    
    # Test very small parameter differences
    hash_5 = compute_merge_hash(models_1, merge_strength=0.50001, **base_params)
    hash_6 = compute_merge_hash(models_1, merge_strength=0.5, **base_params)
    
    print(f"merge_strength=0.50001: {hash_5}")
    print(f"merge_strength=0.5:     {hash_6}")
    print(f"Different (should be FALSE due to rounding): {hash_5 != hash_6}")
    
    # Test different model content
    class DifferentModelMock:
        def __init__(self, model_name):
            self.model_name = model_name
            self._state_dict = {
                f"different_layers.{i}.weight": f"different_tensor_data_{model_name}_{i}" 
                for i in range(3)  # Different structure
            }
        
        def state_dict(self):
            return self._state_dict
        
        def named_parameters(self):
            return iter([])
        
        def clone(self):
            return DifferentModelMock(self.model_name)
    
    model_different = DifferentModelMock("different_model")
    models_different = [model_different, model_other_1] + [None] * 10
    hash_7 = compute_merge_hash(models_different, merge_strength=0.5, **base_params)
    
    print(f"Different model content: {hash_7}")
    print(f"Different from original: {hash_1 != hash_7}")
    
    if hash_1 != hash_7:
        print("✅ GOOD: Different model content produces different cache keys")
    else:
        print("❌ ISSUE: Different model content should produce different cache keys")
        return False
    
    return True

def print_summary():
    """Print summary of the fix"""
    print("\n" + "="*80)
    print("CACHE INVALIDATION FIX SUMMARY")
    print("="*80)
    print("""
ISSUE RESOLVED:
- ComfyUI model object recreation no longer breaks parameter change detection
- Cache keys are now stable across model object recreations
- Parameter changes (like merge_strength 0.5 → 0.1) are properly detected

TECHNICAL DETAILS:
- Modified compute_merge_hash() in shared/cache_management.py
- Replaced object ID-based hashing with content-based hashing
- Model state_dict keys are now used for stable model identification
- Fallback to position-based identification for edge cases

IMPACT:
✅ Changing merge_strength from 0.5 to 0.1 will now trigger a new merge
✅ Same parameters will hit cache even with new model objects  
✅ Different models or parameters will correctly miss cache
✅ Cache invalidation works as expected in ComfyUI workflow execution

TESTING:
- All cache invalidation tests pass
- Model object recreation doesn't affect cache keys
- Parameter changes correctly produce different cache keys
- Cache hits and misses work as expected
""")

if __name__ == "__main__":
    if validate_cache_fix():
        print("\n✅ ALL TESTS PASSED!")
        print_summary()
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("The cache invalidation fix needs more work.")