"""
Cache management functionality for DonutWidenMerge operations.

This module provides functions for managing the merge result cache, including:
- Computing hash keys for merge parameters
- Checking and retrieving cached results  
- Storing new merge results with memory monitoring
- Clearing the global cache

The cache helps avoid redundant merge operations by storing results based on
a hash of the input models and merge parameters.
"""

import hashlib
import gc
from .constants import _MERGE_CACHE, _CACHE_MAX_SIZE
from .logging_config import diagnostic_logger


def compute_merge_hash(models, merge_strength, min_strength, max_strength, importance_threshold, importance_boost, rank_sensitivity, skip_threshold, normalization_mode, lora_stack=None, scale_to_min_max=False, invert_strengths=False):
    """Compute hash of merge parameters to detect changes - FIXED: More robust hashing"""
    hasher = hashlib.sha256()  # Changed from md5 to sha256 for better collision resistance
    
    # DEBUG: Log all parameters being hashed
    diagnostic_logger.debug(f"[Cache Debug] Computing hash for parameters:")
    diagnostic_logger.debug(f"  merge_strength: {merge_strength}")
    diagnostic_logger.debug(f"  min_strength: {min_strength}")
    diagnostic_logger.debug(f"  max_strength: {max_strength}")
    diagnostic_logger.debug(f"  importance_threshold: {importance_threshold}")
    diagnostic_logger.debug(f"  importance_boost: {importance_boost}")
    diagnostic_logger.debug(f"  rank_sensitivity: {rank_sensitivity}")
    diagnostic_logger.debug(f"  skip_threshold: {skip_threshold}")
    diagnostic_logger.debug(f"  normalization_mode: {normalization_mode}")
    diagnostic_logger.debug(f"  scale_to_min_max: {scale_to_min_max}")
    diagnostic_logger.debug(f"  invert_strengths: {invert_strengths}")
    diagnostic_logger.debug(f"  lora_stack: {lora_stack is not None}")
    diagnostic_logger.debug(f"  num_models: {len([m for m in models if m is not None and not getattr(m, '_is_filler', False)])}")

    # Hash model inputs - FIXED: Use model state checksum instead of object ID
    for i, model in enumerate(models):
        if model is not None and not getattr(model, "_is_filler", False):
            try:
                # Create a more stable hash based on model parameters
                model_params = list(model.named_parameters()) if hasattr(model, 'named_parameters') else []
                if model_params:
                    # Use first and last parameter shapes and a few sample values for hash
                    first_param = model_params[0][1] if model_params else None
                    last_param = model_params[-1][1] if len(model_params) > 1 else first_param

                    if first_param is not None:
                        hasher.update(str(first_param.shape).encode())
                        # Round parameter values to avoid floating point precision issues
                        first_values = [round(float(x), 6) for x in first_param.flatten()[:10].tolist()]
                        hasher.update(str(first_values).encode())
                    if last_param is not None and last_param is not first_param:
                        hasher.update(str(last_param.shape).encode())
                        # Round parameter values to avoid floating point precision issues
                        last_values = [round(float(x), 6) for x in last_param.flatten()[:10].tolist()]
                        hasher.update(str(last_values).encode())
                else:
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
                hasher.update(f"{model_class}_position_{i}".encode())

    # Hash merge parameters - FIXED: Include ALL parameters that affect merge with proper floating point precision
    # Round floating point values to avoid precision issues that could cause cache misses
    precision = 8  # 8 decimal places should be sufficient for UI parameters
    param_string = f"{round(merge_strength, precision)}_{round(min_strength, precision)}_{round(max_strength, precision)}_{round(importance_threshold, precision)}_{round(importance_boost, precision)}_{round(rank_sensitivity, precision)}_{round(skip_threshold, precision)}_{normalization_mode}_{scale_to_min_max}_{invert_strengths}"
    diagnostic_logger.debug(f"[Cache Debug] Parameter string for hash: {param_string}")
    hasher.update(param_string.encode())
    
    # Hash LoRA stack if present
    if lora_stack is not None:
        try:
            if hasattr(lora_stack, '__iter__'):
                for idx, lora_item in enumerate(lora_stack):
                    if isinstance(lora_item, tuple) and len(lora_item) >= 4:
                        # Hash lora name, model_weight, clip_weight, block_vector
                        lora_string = f"lora_{idx}_{lora_item[0]}_{lora_item[1]}_{lora_item[2]}_{lora_item[3]}"
                        diagnostic_logger.debug(f"[Cache Debug] LoRA {idx}: {lora_string}")
                        hasher.update(lora_string.encode())
                    else:
                        lora_string = f"lora_{idx}_{str(lora_item)}"
                        diagnostic_logger.debug(f"[Cache Debug] LoRA {idx} (fallback): {lora_string}")
                        hasher.update(lora_string.encode())
            else:
                lora_string = f"lora_single_{str(lora_stack)}"
                diagnostic_logger.debug(f"[Cache Debug] Single LoRA: {lora_string}")
                hasher.update(lora_string.encode())
        except Exception as e:
            # Fallback for unparseable lora_stack
            lora_string = f"lora_fallback_{id(lora_stack)}"
            diagnostic_logger.debug(f"[Cache Debug] LoRA fallback: {lora_string} (error: {e})")
            hasher.update(lora_string.encode())

    final_hash = hasher.hexdigest()
    diagnostic_logger.debug(f"[Cache Debug] Final computed hash: {final_hash}")
    return final_hash


def check_cache_for_merge(cache_key):
    """Check if we have a cached result for this merge"""
    diagnostic_logger.debug(f"[Cache Debug] Looking for cache key: {cache_key}")
    diagnostic_logger.debug(f"[Cache Debug] Current cache contains {len(_MERGE_CACHE)} entries")
    
    if cache_key in _MERGE_CACHE:
        print("[Cache] Found cached merge result - skipping processing")
        diagnostic_logger.info(f"[Cache Debug] *** CACHE HIT *** for key: {cache_key}")
        cached_result = _MERGE_CACHE[cache_key]
        # Clone the cached model to prevent mutation of cached result
        cached_model, results_text, parameter_info = cached_result
        fresh_model = cached_model.clone()
        return (fresh_model, results_text, parameter_info)
    else:
        diagnostic_logger.info(f"[Cache Debug] *** CACHE MISS *** for key: {cache_key}")
        if len(_MERGE_CACHE) > 0:
            diagnostic_logger.debug("[Cache Debug] Existing cache keys:")
            for existing_key in list(_MERGE_CACHE.keys())[:5]:  # Show first 5 keys
                diagnostic_logger.debug(f"  {existing_key}")
            if len(_MERGE_CACHE) > 5:
                diagnostic_logger.debug(f"  ... and {len(_MERGE_CACHE) - 5} more")
        
    return None


def store_merge_result(cache_key, result):
    """Store merge result in cache with memory monitoring"""
    global _MERGE_CACHE

    diagnostic_logger.debug(f"[Cache Debug] Storing result for key: {cache_key}")

    # Clear old entries if cache is full
    if len(_MERGE_CACHE) >= _CACHE_MAX_SIZE:
        oldest_key = next(iter(_MERGE_CACHE))
        diagnostic_logger.debug(f"[Cache Debug] Cache full, removing oldest key: {oldest_key}")
        del _MERGE_CACHE[oldest_key]
        # Light cleanup when removing old cache entries
        gc.collect()
        print(f"[Cache] Removed oldest entry, cache size: {len(_MERGE_CACHE)}")

    _MERGE_CACHE[cache_key] = result
    print(f"[Cache] Stored merge result, cache size: {len(_MERGE_CACHE)}")
    diagnostic_logger.info(f"[Cache Debug] *** STORED *** result for key: {cache_key}")


def clear_merge_cache():
    """Clear the model merge cache"""
    global _MERGE_CACHE
    cache_size = len(_MERGE_CACHE)
    _MERGE_CACHE.clear()
    print(f"[Cache] Cleared all cached merge results ({cache_size} entries removed)")
    diagnostic_logger.info(f"[Cache Debug] Cache manually cleared, removed {cache_size} entries")


def enable_cache_debug_logging():
    """Enable detailed cache debugging logging"""
    from .logging_config import diagnostic_logger, configure_widen_logging
    configure_widen_logging("DEBUG", enable_diagnostic_debug=True)
    diagnostic_logger.info("[Cache Debug] Cache debugging enabled - will show detailed cache operations")


def inspect_cache():
    """Debug function to inspect current cache contents"""
    diagnostic_logger.info(f"[Cache Debug] Current cache contains {len(_MERGE_CACHE)} entries:")
    for i, cache_key in enumerate(list(_MERGE_CACHE.keys())):
        diagnostic_logger.info(f"  [{i}] {cache_key}")
    return list(_MERGE_CACHE.keys())


def check_cache_for_merge_with_bypass(cache_key, force_fresh=False):
    """Check cache with option to bypass for testing"""
    if force_fresh:
        diagnostic_logger.info(f"[Cache Debug] *** BYPASSING CACHE *** (force_fresh=True) for key: {cache_key}")
        return None
    return check_cache_for_merge(cache_key)


def analyze_hash_differences(models1, params1, models2, params2):
    """Debug tool to analyze why two similar parameter sets might produce different hashes"""
    diagnostic_logger.info("[Cache Debug] Analyzing hash differences between two parameter sets:")
    
    # Unpack parameters - updated to handle new toggle parameters
    if len(params1) == 11:  # New format with toggle parameters
        merge_strength1, min_strength1, max_strength1, importance_threshold1, importance_boost1, rank_sensitivity1, skip_threshold1, normalization_mode1, lora_stack1, scale_to_min_max1, invert_strengths1 = params1
        merge_strength2, min_strength2, max_strength2, importance_threshold2, importance_boost2, rank_sensitivity2, skip_threshold2, normalization_mode2, lora_stack2, scale_to_min_max2, invert_strengths2 = params2
    else:  # Legacy format without toggle parameters
        merge_strength1, min_strength1, max_strength1, importance_threshold1, importance_boost1, rank_sensitivity1, skip_threshold1, normalization_mode1, lora_stack1 = params1
        merge_strength2, min_strength2, max_strength2, importance_threshold2, importance_boost2, rank_sensitivity2, skip_threshold2, normalization_mode2, lora_stack2 = params2
        scale_to_min_max1 = scale_to_min_max2 = False
        invert_strengths1 = invert_strengths2 = False
    
    # Compare each parameter
    param_differences = []
    if merge_strength1 != merge_strength2:
        param_differences.append(f"merge_strength: {merge_strength1} vs {merge_strength2}")
    if min_strength1 != min_strength2:
        param_differences.append(f"min_strength: {min_strength1} vs {min_strength2}")
    if max_strength1 != max_strength2:
        param_differences.append(f"max_strength: {max_strength1} vs {max_strength2}")
    if importance_threshold1 != importance_threshold2:
        param_differences.append(f"importance_threshold: {importance_threshold1} vs {importance_threshold2}")
    if importance_boost1 != importance_boost2:
        param_differences.append(f"importance_boost: {importance_boost1} vs {importance_boost2}")
    if rank_sensitivity1 != rank_sensitivity2:
        param_differences.append(f"rank_sensitivity: {rank_sensitivity1} vs {rank_sensitivity2}")
    if skip_threshold1 != skip_threshold2:
        param_differences.append(f"skip_threshold: {skip_threshold1} vs {skip_threshold2}")
    if normalization_mode1 != normalization_mode2:
        param_differences.append(f"normalization_mode: {normalization_mode1} vs {normalization_mode2}")
    if scale_to_min_max1 != scale_to_min_max2:
        param_differences.append(f"scale_to_min_max: {scale_to_min_max1} vs {scale_to_min_max2}")
    if invert_strengths1 != invert_strengths2:
        param_differences.append(f"invert_strengths: {invert_strengths1} vs {invert_strengths2}")
    
    if param_differences:
        diagnostic_logger.info("[Cache Debug] Parameter differences found:")
        for diff in param_differences:
            diagnostic_logger.info(f"  {diff}")
    else:
        diagnostic_logger.warning("[Cache Debug] *** NO PARAMETER DIFFERENCES FOUND *** - This may indicate a floating point precision issue!")
    
    # Generate hashes for comparison
    hash1 = compute_merge_hash(models1, *params1)
    hash2 = compute_merge_hash(models2, *params2)
    
    diagnostic_logger.info(f"[Cache Debug] Hash 1: {hash1}")
    diagnostic_logger.info(f"[Cache Debug] Hash 2: {hash2}")
    diagnostic_logger.info(f"[Cache Debug] Hashes are {'IDENTICAL' if hash1 == hash2 else 'DIFFERENT'}")
    
    return hash1, hash2, param_differences


def create_merge_tracer(enable_tracing=False):
    """Create a merge decision tracer that can be passed to merge functions"""
    if not enable_tracing:
        return None
    
    tracer_data = {
        'parameters_merged': [],
        'parameters_skipped': [],
        'cache_hits': [],
        'total_parameters': 0,
        'merge_decisions': {}
    }
    
    def trace_decision(param_name, decision, reason, extra_info=None):
        """Trace a merge decision"""
        decision_info = {
            'parameter': param_name,
            'decision': decision,  # 'merged', 'skipped', 'cached'
            'reason': reason,
            'extra_info': extra_info
        }
        
        if decision == 'merged':
            tracer_data['parameters_merged'].append(decision_info)
        elif decision == 'skipped':
            tracer_data['parameters_skipped'].append(decision_info)
        elif decision == 'cached':
            tracer_data['cache_hits'].append(decision_info)
            
        tracer_data['merge_decisions'][param_name] = decision_info
        tracer_data['total_parameters'] += 1
        
        diagnostic_logger.debug(f"[Merge Trace] {param_name}: {decision} - {reason}")
    
    tracer_data['trace'] = trace_decision
    return tracer_data


def print_merge_trace_summary(tracer_data):
    """Print a summary of merge decisions"""
    if not tracer_data:
        return "No trace data available"
    
    summary = f"""
[Merge Trace Summary]
Total Parameters: {tracer_data['total_parameters']}
Parameters Merged: {len(tracer_data['parameters_merged'])}
Parameters Skipped: {len(tracer_data['parameters_skipped'])}
Cache Hits: {len(tracer_data['cache_hits'])}

Merge Rate: {len(tracer_data['parameters_merged']) / max(1, tracer_data['total_parameters']) * 100:.1f}%
Skip Rate: {len(tracer_data['parameters_skipped']) / max(1, tracer_data['total_parameters']) * 100:.1f}%
"""
    
    # Show recent skipped parameters if any
    if tracer_data['parameters_skipped']:
        summary += "\nRecent Skip Reasons:\n"
        for skip_info in tracer_data['parameters_skipped'][-5:]:  # Last 5
            summary += f"  {skip_info['parameter']}: {skip_info['reason']}\n"
    
    diagnostic_logger.info(summary)
    return summary


def debug_cache_invalidation_test():
    """
    Test function to help debug cache invalidation issues.
    This creates two identical parameter sets and checks if they produce the same hash.
    """
    diagnostic_logger.info("[Cache Debug] Running cache invalidation test...")
    
    # Create test parameters
    test_params = {
        'merge_strength': 1.0,
        'min_strength': 0.0, 
        'max_strength': 1.0,
        'importance_threshold': 1.0,
        'importance_boost': 1.0,
        'rank_sensitivity': 2.0,
        'skip_threshold': 0.0,
        'normalization_mode': 'magnitude',
        'lora_stack': None,
        'scale_to_min_max': False,
        'invert_strengths': False
    }
    
    # Test with dummy models (filler models)
    test_models = []
    for i in range(2):
        class _TestModel:
            def state_dict(self): return {}
            def named_parameters(self): return iter([])
        m = _TestModel()
        setattr(m, "_is_filler", True)
        test_models.append(m)
    
    # Generate two identical hashes
    hash1 = compute_merge_hash(test_models, **test_params)
    hash2 = compute_merge_hash(test_models, **test_params)
    
    diagnostic_logger.info(f"[Cache Debug] Test hash 1: {hash1}")
    diagnostic_logger.info(f"[Cache Debug] Test hash 2: {hash2}")
    diagnostic_logger.info(f"[Cache Debug] Hashes identical: {hash1 == hash2}")
    
    # Test with slightly different merge_strength
    test_params_2 = test_params.copy()
    test_params_2['merge_strength'] = 0.01
    hash3 = compute_merge_hash(test_models, **test_params_2)
    
    diagnostic_logger.info(f"[Cache Debug] Test hash 3 (merge_strength=0.01): {hash3}")
    diagnostic_logger.info(f"[Cache Debug] Hash 1 vs Hash 3 different: {hash1 != hash3}")
    
    return hash1 == hash2 and hash1 != hash3