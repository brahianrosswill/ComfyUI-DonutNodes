#!/usr/bin/env python3
"""
Fix for cache invalidation issue: Model object identity should not affect cache keys.

The issue is that ComfyUI creates new model wrapper objects on each execution,
causing cache keys to change even when the underlying models are identical.
This masks parameter change detection.
"""

import sys
import os

# Add the project directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def fix_compute_merge_hash():
    """Fix the compute_merge_hash function to be stable across model object recreations"""
    
    print("="*80)
    print("FIXING CACHE MODEL IDENTITY ISSUE")
    print("="*80)
    
    # Read the current cache_management.py
    cache_file = os.path.join(current_dir, "shared", "cache_management.py")
    
    with open(cache_file, 'r') as f:
        content = f.read()
    
    # The problematic lines are around 61 and 65 where we use id(model)
    # We need to replace this with a more stable approach
    
    old_fallback_code = '''                else:
                    # Deterministic fallback - use model class and id
                    model_class = type(model).__name__
                    hasher.update(f"{model_class}_{id(model)}".encode())
            except Exception:
                # Deterministic ultimate fallback
                model_class = type(model).__name__ if hasattr(model, '__class__') else 'unknown'
                hasher.update(f"{model_class}_{id(model)}".encode())'''
    
    new_fallback_code = '''                else:
                    # Stable fallback - use model class and a stable identifier
                    # Try to get a stable hash based on model structure/content
                    model_class = type(model).__name__
                    try:
                        # Try to use model state_dict keys as a stable identifier
                        if hasattr(model, 'state_dict'):
                            state_keys = sorted(model.state_dict().keys())
                            if state_keys:
                                keys_str = "_".join(state_keys[:5])  # First 5 keys for hash
                                hasher.update(f"{model_class}_{keys_str}".encode())
                            else:
                                # Empty state dict - use class name and position
                                hasher.update(f"{model_class}_empty_{i}".encode())
                        else:
                            # No state dict - use class name and position
                            hasher.update(f"{model_class}_no_state_{i}".encode())
                    except Exception:
                        # Ultimate stable fallback - use class name and position
                        hasher.update(f"{model_class}_fallback_{i}".encode())
            except Exception:
                # Deterministic ultimate fallback - avoid object id
                model_class = type(model).__name__ if hasattr(model, '__class__') else 'unknown'
                # Use model position in list instead of object id for stability
                hasher.update(f"{model_class}_position_{i}".encode())'''
    
    if old_fallback_code in content:
        new_content = content.replace(old_fallback_code, new_fallback_code)
        
        # Write the fixed version
        with open(cache_file, 'w') as f:
            f.write(new_content)
        
        print("✅ FIXED: Updated compute_merge_hash to use stable model identifiers")
        print("✅ Model object recreation will no longer affect cache keys")
        print("✅ Parameter changes will now be properly detected")
        
        return True
    else:
        print("❌ Could not find the problematic code to replace")
        print("The cache_management.py file may have been modified")
        return False

def test_fix():
    """Test that the fix works"""
    print("\n" + "="*50)
    print("TESTING THE FIX")
    print("="*50)
    
    try:
        from shared.cache_management import compute_merge_hash
        
        # Create dummy models
        class _TestModel:
            def state_dict(self): return {"layer1.weight": "dummy", "layer2.bias": "dummy"}
            def named_parameters(self): return iter([])
        
        # Test 1: Same model content, different objects
        model1a = _TestModel() 
        model1b = _TestModel()  # Different object, same content
        
        models_a = [model1a, None, None, None, None, None, None, None, None, None, None, None]
        models_b = [model1b, None, None, None, None, None, None, None, None, None, None, None]
        
        params = {
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
        
        hash_a = compute_merge_hash(models_a, **params)
        hash_b = compute_merge_hash(models_b, **params)
        
        print(f"Same model content, different objects:")
        print(f"Hash A: {hash_a}")
        print(f"Hash B: {hash_b}")
        print(f"Hashes identical: {hash_a == hash_b}")
        
        if hash_a == hash_b:
            print("✅ GOOD: Model object identity no longer affects cache")
        else:
            print("❌ STILL BROKEN: Different objects produce different hashes")
        
        # Test 2: Different parameters, same models
        params_2 = params.copy()
        params_2['merge_strength'] = 0.1
        
        hash_c = compute_merge_hash(models_a, **params_2)
        
        print(f"\nDifferent parameters:")
        print(f"Hash with 0.5: {hash_a}")
        print(f"Hash with 0.1: {hash_c}")
        print(f"Hashes different: {hash_a != hash_c}")
        
        if hash_a != hash_c:
            print("✅ GOOD: Parameter changes still produce different hashes")
        else:
            print("❌ BROKEN: Parameter changes don't affect hash")
        
        return hash_a == hash_b and hash_a != hash_c
        
    except Exception as e:
        print(f"❌ Error testing fix: {e}")
        return False

if __name__ == "__main__":
    if fix_compute_merge_hash():
        if test_fix():
            print("\n" + "="*80)
            print("✅ FIX SUCCESSFUL!")
            print("Cache invalidation should now work correctly:")
            print("1. Same models with different parameters → different cache keys")
            print("2. Same models/parameters with different object IDs → same cache keys")
            print("3. ComfyUI model recreation will no longer break parameter detection")
            print("="*80)
        else:
            print("\n❌ Fix applied but tests failed!")
    else:
        print("\n❌ Could not apply fix!")