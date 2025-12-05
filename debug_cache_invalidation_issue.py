#!/usr/bin/env python3
"""
Debug script to investigate cache invalidation issues when changing merge_strength
from 0.5 to 0.1 in the actual ComfyUI node execution context.
"""

import sys
import os

# Add the project directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from shared.cache_management import (
        compute_merge_hash, check_cache_for_merge_with_bypass, 
        store_merge_result, clear_merge_cache, enable_cache_debug_logging,
        inspect_cache, _MERGE_CACHE
    )
    from shared.constants import _CACHE_MAX_SIZE
except ImportError as e:
    print(f"Could not import modules: {e}")
    print("Run this from the ComfyUI-DonutNodes directory")
    sys.exit(1)

def simulate_comfyui_execution():
    """Simulate the actual ComfyUI node execution to find the cache invalidation bug"""
    
    print("="*80)
    print("DEBUGGING CACHE INVALIDATION ISSUE IN COMFYUI EXECUTION")
    print("="*80)
    
    # Enable debug logging
    enable_cache_debug_logging()
    
    # Clear cache to start fresh
    clear_merge_cache()
    print(f"Starting with empty cache")
    
    # Create dummy models that simulate ComfyUI model objects
    def create_dummy_model():
        class _DummyModel:
            def state_dict(self): return {}
            def named_parameters(self): return iter([])
            def clone(self): return create_dummy_model()
        return _DummyModel()
    
    # Simulate the models array as it would appear in ComfyUI
    model_base = create_dummy_model()
    model_other = create_dummy_model()
    all_models = [model_base, model_other, None, None, None, None, None, None, None, None, None, None]
    
    # First execution with merge_strength=0.5
    print("\n" + "="*50)
    print("STEP 1: First execution with merge_strength=0.5")
    print("="*50)
    
    params_1 = {
        'merge_strength': 0.5,
        'min_strength': 0.0,
        'max_strength': 1.0,
        'importance_threshold': 1.0,
        'importance_boost': 1.0,
        'rank_sensitivity': 2.0,
        'skip_threshold': 0.0,
        'normalization_mode': 'magnitude_enhanced_widen',
        'lora_stack': None
    }
    
    cache_key_1 = compute_merge_hash(all_models, **params_1)
    print(f"Generated cache key: {cache_key_1}")
    
    # Check cache (should be miss)
    cached_result_1 = check_cache_for_merge_with_bypass(cache_key_1, force_fresh=False)
    print(f"Cached result: {cached_result_1}")
    
    if cached_result_1 is None:
        print("✅ EXPECTED: Cache miss on first run")
        # Simulate storing a result
        dummy_result = (create_dummy_model(), "result_text_1", "param_info_1")
        store_merge_result(cache_key_1, dummy_result)
        print("Stored result in cache")
    else:
        print("❌ UNEXPECTED: Cache hit on first run!")
    
    print(f"Cache size after first execution: {len(_MERGE_CACHE)}")
    inspect_cache()
    
    # Second execution with merge_strength=0.1 
    # This should generate a different cache key and miss the cache
    print("\n" + "="*50)
    print("STEP 2: Second execution with merge_strength=0.1")
    print("="*50)
    
    params_2 = params_1.copy()
    params_2['merge_strength'] = 0.1
    
    cache_key_2 = compute_merge_hash(all_models, **params_2)
    print(f"Generated cache key: {cache_key_2}")
    print(f"Keys are different: {cache_key_1 != cache_key_2}")
    
    if cache_key_1 == cache_key_2:
        print("❌ BUG FOUND: Same cache key for different merge_strength!")
        print("This explains why parameter changes don't trigger new merges!")
    else:
        print("✅ GOOD: Different cache keys for different merge_strength")
    
    # Check cache (should be miss since key is different)
    cached_result_2 = check_cache_for_merge_with_bypass(cache_key_2, force_fresh=False)
    print(f"Cached result: {cached_result_2}")
    
    if cached_result_2 is None:
        print("✅ EXPECTED: Cache miss with different parameters")
    else:
        print("❌ BUG: Cache hit with different parameters!")
        print("This means the cache lookup logic is broken!")
    
    print(f"Cache size after second execution: {len(_MERGE_CACHE)}")
    inspect_cache()
    
    # Third execution - repeat merge_strength=0.1 to test cache hit
    print("\n" + "="*50)
    print("STEP 3: Repeat execution with merge_strength=0.1 (should hit cache)")
    print("="*50)
    
    cache_key_3 = compute_merge_hash(all_models, **params_2)
    print(f"Generated cache key: {cache_key_3}")
    print(f"Same as previous key: {cache_key_2 == cache_key_3}")
    
    cached_result_3 = check_cache_for_merge_with_bypass(cache_key_3, force_fresh=False)
    print(f"Cached result: {cached_result_3}")
    
    if cached_result_3 is not None and cache_key_2 == cache_key_3:
        print("✅ EXPECTED: Cache hit with identical parameters")
    else:
        print("❌ BUG: Expected cache hit but got miss!")
    
    # Test potential edge cases
    print("\n" + "="*50)
    print("STEP 4: Testing edge cases")
    print("="*50)
    
    # Test floating point precision issues
    params_4 = params_1.copy()
    params_4['merge_strength'] = 0.5000000001  # Tiny difference
    
    cache_key_4 = compute_merge_hash(all_models, **params_4)
    print(f"merge_strength=0.5000000001 key: {cache_key_4}")
    print(f"Different from original 0.5: {cache_key_1 != cache_key_4}")
    
    # Test if the issue is in the model comparison
    print("\n" + "="*50)
    print("STEP 5: Testing model identity issues")
    print("="*50)
    
    # Create new model instances (different object IDs)
    new_model_base = create_dummy_model()
    new_model_other = create_dummy_model()
    new_all_models = [new_model_base, new_model_other, None, None, None, None, None, None, None, None, None, None]
    
    cache_key_5 = compute_merge_hash(new_all_models, **params_1)
    print(f"Same params, different model objects: {cache_key_5}")
    print(f"Different from original: {cache_key_1 != cache_key_5}")
    
    if cache_key_1 == cache_key_5:
        print("✅ GOOD: Model identity doesn't affect cache key (should be deterministic)")
    else:
        print("⚠️  NOTICE: Different model objects produce different cache keys")
        print("This could cause issues if ComfyUI creates new model objects")
    
    return {
        'first_key': cache_key_1,
        'second_key': cache_key_2, 
        'keys_different': cache_key_1 != cache_key_2,
        'cache_working': cached_result_1 is None and cached_result_2 is None
    }

def test_comfyui_model_behavior():
    """Test if the issue is related to how ComfyUI handles model objects"""
    
    print("\n" + "="*80)
    print("TESTING COMFYUI MODEL OBJECT BEHAVIOR")
    print("="*80)
    
    # This simulates what might happen in ComfyUI when parameters change
    # The question is: does ComfyUI reuse the same model objects or create new ones?
    
    class ComfyUIModelWrapper:
        def __init__(self, name):
            self.name = name
            self.model = self._create_inner_model()
        
        def _create_inner_model(self):
            class _InnerModel:
                def state_dict(self): return {}
                def named_parameters(self): return iter([])
            return _InnerModel()
        
        def clone(self):
            return ComfyUIModelWrapper(self.name)
    
    # Test 1: Same wrapper objects
    print("Test 1: Same wrapper objects (parameter change only)")
    wrapper1 = ComfyUIModelWrapper("base")
    wrapper2 = ComfyUIModelWrapper("other")
    
    models1 = [wrapper1, wrapper2] + [None] * 10
    models2 = [wrapper1, wrapper2] + [None] * 10  # Same objects
    
    key1 = compute_merge_hash(models1, 0.5, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, "magnitude_enhanced_widen", None)
    key2 = compute_merge_hash(models2, 0.1, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, "magnitude_enhanced_widen", None)
    
    print(f"Same objects, different merge_strength: {key1 != key2}")
    
    # Test 2: Different wrapper objects (simulating ComfyUI reload)
    print("\nTest 2: Different wrapper objects (simulating model reload)")
    wrapper3 = ComfyUIModelWrapper("base")
    wrapper4 = ComfyUIModelWrapper("other")
    
    models3 = [wrapper3, wrapper4] + [None] * 10  # Different objects, same name
    
    key3 = compute_merge_hash(models3, 0.5, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, "magnitude_enhanced_widen", None)
    
    print(f"Different objects, same merge_strength: {key1 != key3}")
    
    if key1 == key3:
        print("✅ GOOD: Model object identity doesn't affect hash (stable hashing)")
    else:
        print("❌ POTENTIAL ISSUE: Different model objects produce different hashes")
        print("This could cause cache misses when ComfyUI reloads models")

if __name__ == "__main__":
    result = simulate_comfyui_execution()
    test_comfyui_model_behavior()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if result['keys_different'] and result['cache_working']:
        print("✅ CACHE SYSTEM APPEARS TO BE WORKING CORRECTLY")
        print("The issue may be elsewhere:")
        print("1. ComfyUI might not be calling the node with new parameters")
        print("2. ComfyUI might be reusing outputs from its own cache layer")
        print("3. There might be a caching layer above the node level")
        print("4. The node might not be getting re-executed at all")
    else:
        print("❌ CACHE SYSTEM HAS ISSUES:")
        if not result['keys_different']:
            print("- Different parameters produce same cache key")
        if not result['cache_working']:
            print("- Cache lookup logic is broken")
    
    print("\nNext steps:")
    print("1. Enable cache debug logging in ComfyUI UI")
    print("2. Check if the node execute() method is being called when parameters change")
    print("3. Look for ComfyUI-level caching that might be interfering")
    print("4. Verify that parameter changes are actually reaching the node")