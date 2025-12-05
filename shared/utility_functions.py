"""
Utility Functions for Enhanced WIDEN Merge Operations

This module contains utility functions extracted from DonutWidenMerge.py for better modularity
and code organization. These functions support advanced model merging operations including:

- Parameter magnitude and direction computation
- Batch computation optimization
- Compatibility analysis and threshold recommendation
- Tensor ranking and importance scoring
- Memory-efficient tensor operations

All functions maintain their original implementation logic for compatibility.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Import shared modules
try:
    # Try relative imports first (when running as part of ComfyUI)
    from .alignment import safe_stack
    from .logging_config import performance_logger
except ImportError:
    # Fallback to absolute imports (when running standalone)
    from alignment import safe_stack
    from logging_config import performance_logger


# Shared pooling kernels to reduce allocations
_POOLING_KERNELS = {}


def _transpose_token_embeddings(param_dict):
    """Transpose token embeddings"""
    for param_name in param_dict:
        if param_name == "model.embed_tokens.weight":
            param_dict[param_name] = param_dict[param_name].transpose(dim0=0, dim1=1)


def _compute_param_magnitude_direction(param_dict, module_dict):
    """Compute magnitude vector and direction matrix for parameters"""
    param_magnitude_dict, param_direction_dict = {}, {}
    
    for param_name in param_dict:
        param_last_name = param_name.split(".")[-1]
        module_name = param_name[:-len(f".{param_last_name}")]
        
        if param_dict[param_name].dim() == 1:
            # Handle 1D parameters (bias, norm) - treat each element as a feature
            param_tensor = param_dict[param_name]
            magnitude_vector = torch.abs(param_tensor)  # For 1D, magnitude is just absolute value
            direction_matrix = torch.sign(param_tensor)  # For 1D, direction is just the sign
            
            param_magnitude_dict[param_name] = magnitude_vector
            param_direction_dict[param_name] = direction_matrix
            
        elif param_dict[param_name].dim() == 2:
            # Compute magnitude and direction for 2D parameters
            magnitude_vector = torch.norm(param_dict[param_name], p=2, dim=0)
            direction_matrix = param_dict[param_name] / (magnitude_vector + 1e-8)
            
            param_magnitude_dict[param_name] = magnitude_vector
            param_direction_dict[param_name] = direction_matrix
            
        elif param_dict[param_name].dim() > 2:
            # Handle higher dimensional parameters (conv layers, etc.)
            # Preserve per-channel structure by averaging only spatial dims
            tensor = param_dict[param_name]
            
            # Collapse only dims beyond the first two, preserving [out_features, in_features]
            if tensor.ndim > 2:
                # Mean over all spatial/extra dims, keep channel structure
                flat = tensor.mean(dim=tuple(range(2, tensor.ndim)))
            else:
                flat = tensor
            
            magnitude_vector = torch.norm(flat, p=2, dim=0)
            direction_matrix = flat / (magnitude_vector + 1e-8)
            
            param_magnitude_dict[param_name] = magnitude_vector
            param_direction_dict[param_name] = direction_matrix
    
    return param_magnitude_dict, param_direction_dict


def _compute_param_magnitude_direction_differences(pretrained_param_magnitude_dict, pretrained_param_direction_dict,
                                                 finetuned_param_magnitude_dict, finetuned_param_direction_dict, module_dict):
    """Compute magnitude and direction differences"""
    param_magnitude_diff_dict, param_direction_diff_dict = {}, {}
    
    for param_name in pretrained_param_magnitude_dict:
        # Ensure tensors are on the same device
        pretrained_mag = pretrained_param_magnitude_dict[param_name]
        finetuned_mag = finetuned_param_magnitude_dict[param_name]
        pretrained_dir = pretrained_param_direction_dict[param_name]
        finetuned_dir = finetuned_param_direction_dict[param_name]
        
        # Move to same device (prefer CUDA if available)
        target_device = pretrained_mag.device
        if finetuned_mag.device != target_device:
            finetuned_mag = finetuned_mag.to(target_device)
        if finetuned_dir.device != target_device:
            finetuned_dir = finetuned_dir.to(target_device)
        if pretrained_dir.device != target_device:
            pretrained_dir = pretrained_dir.to(target_device)
        
        # Compute magnitude difference
        param_magnitude_diff = torch.abs(finetuned_mag - pretrained_mag)
        
        # Compute direction difference
        param_direction_diff = 1.0 - torch.cosine_similarity(
            finetuned_dir,
            pretrained_dir,
            dim=0
        )
        
        param_magnitude_diff_dict[param_name] = param_magnitude_diff
        param_direction_diff_dict[param_name] = param_direction_diff
    
    return param_magnitude_diff_dict, param_direction_diff_dict


def _get_pooling_kernel(target_size):
    """Get or create shared adaptive pooling kernel"""
    key = tuple(target_size) if isinstance(target_size, (list, tuple)) else target_size
    if key not in _POOLING_KERNELS:
        _POOLING_KERNELS[key] = torch.nn.AdaptiveAvgPool2d(target_size)
    return _POOLING_KERNELS[key]


def _batch_compute_magnitude_direction_diffs(task_vectors, param_names, skip_threshold=0.0):
    """
    Batch compute magnitude and direction differences for multiple task vectors.
    Replaces O(N*M) individual operations with batched O(M) operations.
    Includes early-exit optimization for low-magnitude parameters.
    Uses shared pooling kernels for efficiency.
    """
    result_tuples = []
    skipped_params = 0
    
    # Group parameters by name and stack deltas across models
    param_delta_stacks = {}
    
    # First pass: collect and stack deltas for each parameter with early exit
    for param_name in param_names:
        deltas = []
        max_magnitude = 0.0
        
        for task_vector in task_vectors:
            if param_name in task_vector.task_vector_param_dict:
                delta = task_vector.task_vector_param_dict[param_name]
                deltas.append(delta)
                # Quick magnitude check for early exit
                if skip_threshold > 0.0:
                    param_magnitude = delta.abs().max().item()
                    max_magnitude = max(max_magnitude, param_magnitude)
            else:
                # Create zero tensor with proper shape
                if deltas:
                    deltas.append(torch.zeros_like(deltas[0]))
                # If no deltas exist yet, skip this parameter
        
        # Early exit for low-magnitude parameters (saves 80%+ of processing)
        if skip_threshold > 0.0 and max_magnitude < skip_threshold:
            skipped_params += 1
            param_delta_stacks[param_name] = "SKIPPED"
            continue
            
        if deltas:
            try:
                # Use safe_stack for robust alignment
                param_delta_stacks[param_name] = safe_stack(deltas, dim=0)  # Shape: [num_models, ...]
            except Exception as e:
                print(f"[WARNING] Failed to stack deltas for {param_name}: {e}")
                # Fallback to individual processing for this parameter
                param_delta_stacks[param_name] = None
    
    # Second pass: batch compute magnitude and direction for each parameter
    for task_vector_idx in range(len(task_vectors)):
        magnitude_diffs = {}
        direction_diffs = {}
        
        for param_name, stacked_deltas in param_delta_stacks.items():
            if stacked_deltas == "SKIPPED":
                # Skip low-magnitude parameters entirely
                continue
            elif stacked_deltas is None:
                # Fallback to individual processing
                if param_name in task_vectors[task_vector_idx].task_vector_param_dict:
                    delta = task_vectors[task_vector_idx].task_vector_param_dict[param_name]
                    magnitude_diffs[param_name], direction_diffs[param_name] = _compute_single_param_diffs(delta, use_gpu_compute=True, use_mixed_precision=True)
                continue
                
            # Extract this model's delta from the stack
            delta = stacked_deltas[task_vector_idx]
            magnitude_diffs[param_name], direction_diffs[param_name] = _compute_single_param_diffs(delta, use_gpu_compute=True, use_mixed_precision=True)
        
        result_tuples.append((magnitude_diffs, direction_diffs))
    
    # Performance reporting
    if skipped_params > 0:
        total_params = len(param_names)
        performance_logger.info(f"Early-exit optimization: {skipped_params}/{total_params} parameters skipped ({100*skipped_params/total_params:.1f}%)")
    
    return result_tuples


def _analyze_compatibility_patterns_and_recommend_threshold(compatibility_scores, current_skip_threshold=0.0):
    """
    Analyze compatibility score distribution and recommend optimal skip threshold.
    Only suggests new thresholds when current setting is clearly suboptimal.
    """
    if not compatibility_scores:
        return current_skip_threshold, "No compatibility scores available for analysis"
    
    import numpy as np
    
    # Debug: Track function calls to detect duplicate calls
    if not hasattr(_analyze_compatibility_patterns_and_recommend_threshold, '_call_count'):
        _analyze_compatibility_patterns_and_recommend_threshold._call_count = 0
        _analyze_compatibility_patterns_and_recommend_threshold._last_scores = None
    
    _analyze_compatibility_patterns_and_recommend_threshold._call_count += 1
    call_num = _analyze_compatibility_patterns_and_recommend_threshold._call_count
    
    # Convert to numpy for statistical analysis
    scores = np.array(list(compatibility_scores.values()))
    
    # Analyze compatibility score patterns
    
    _analyze_compatibility_patterns_and_recommend_threshold._last_scores = scores.copy()
    
    # Remove extreme outliers (beyond 3 standard deviations)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    filtered_scores = scores[np.abs(scores - mean_score) <= 3 * std_score]
    
    # Statistical analysis
    q10, q25, q50, q75, q90 = np.percentile(filtered_scores, [10, 25, 50, 75, 90])
    min_score, max_score = np.min(filtered_scores), np.max(filtered_scores)
    
    # Detect uniform score problem (like the 0.001463 issue)
    score_range = max_score - min_score
    relative_variation = score_range / mean_score if mean_score > 0 else 0
    
    # Don't recommend changes if user has set a reasonable threshold
    if current_skip_threshold > 0.000001:  # User has set a custom threshold
        current_effectiveness = np.sum(filtered_scores <= current_skip_threshold) / len(filtered_scores)
        if 0.05 <= current_effectiveness <= 0.8:  # Current threshold is working reasonably (5-80% filtered)
            return current_skip_threshold, f"Using current threshold: {current_skip_threshold:.6f} (filtering {current_effectiveness:.1%} of parameters)"
    
    # Recommendation logic based on distribution analysis
    recommendations = []
    
    if relative_variation < 0.1:  # Less than 10% variation
        # Uniform scores detected - recommend conservative threshold
        recommended_threshold = max(q75, current_skip_threshold)
        recommendations.append(f"⚠️  UNIFORM SCORES DETECTED (variation: {relative_variation:.1%})")
        recommendations.append(f"   Many parameters have nearly identical compatibility (~{mean_score:.6f})")
        recommendations.append(f"   Recommended skip_threshold: {recommended_threshold:.6f} (75th percentile)")
        recommendations.append(f"   This would skip {np.sum(filtered_scores <= recommended_threshold)/len(filtered_scores):.1%} of low-impact parameters")
        
    elif q25 > 0.01:  # Most scores are reasonably high
        # Normal distribution - recommend based on lower quartile
        recommended_threshold = max(q10, current_skip_threshold)
        recommendations.append(f"✅ HEALTHY SCORE DISTRIBUTION (range: {min_score:.6f} - {max_score:.6f})")
        recommendations.append(f"   Recommended skip_threshold: {recommended_threshold:.6f} (10th percentile)")
        recommendations.append(f"   This would skip {np.sum(filtered_scores <= recommended_threshold)/len(filtered_scores):.1%} of lowest-impact parameters")
        
    else:  # Many very low scores
        # Heavy low-end distribution - recommend more aggressive filtering
        recommended_threshold = max(q50, current_skip_threshold * 2)
        recommendations.append(f"📊 HEAVY LOW-END DISTRIBUTION detected")
        recommendations.append(f"   {np.sum(filtered_scores <= 0.01)/len(filtered_scores):.1%} of parameters have compatibility < 0.01")
        recommendations.append(f"   Recommended skip_threshold: {recommended_threshold:.6f} (median)")
        recommendations.append(f"   This would skip {np.sum(filtered_scores <= recommended_threshold)/len(filtered_scores):.1%} of low-impact parameters")
    
    # Performance impact estimate
    potential_savings = np.sum(filtered_scores <= recommended_threshold) / len(filtered_scores)
    if potential_savings > 0.5:
        recommendations.append(f"🚀 PERFORMANCE BOOST: {potential_savings:.1%} parameter reduction possible")
    elif potential_savings > 0.2:
        recommendations.append(f"⚡ MODERATE SPEEDUP: {potential_savings:.1%} parameter reduction possible")
    
    # Memory impact for very low scores (like the 0.001463 pattern)
    very_low_count = np.sum(filtered_scores <= 0.005)  # Threshold for "very low impact"
    if very_low_count > len(filtered_scores) * 0.3:  # More than 30% very low
        recommendations.append(f"💾 MEMORY OPTIMIZATION: {very_low_count}/{len(filtered_scores)} parameters have minimal impact")
        recommendations.append(f"   Consider skip_threshold ≥ 0.005 for significant memory savings")
    
    analysis_summary = "\n".join([
        f"[SKIP THRESHOLD ANALYSIS]",
        f"  Analyzed {len(filtered_scores)} compatibility scores",
        f"  Distribution: min={min_score:.6f}, q25={q25:.6f}, median={q50:.6f}, q75={q75:.6f}, max={max_score:.6f}",
        f"  Relative variation: {relative_variation:.1%} ({'LOW' if relative_variation < 0.1 else 'NORMAL' if relative_variation < 0.5 else 'HIGH'})",
        ""
    ] + recommendations)
    
    return recommended_threshold, analysis_summary


def _compute_single_param_diffs(delta: torch.Tensor, use_gpu_compute=True, use_mixed_precision=True):
    """Single parameter diff computation with smart device placement and mixed precision"""
    
    # Smart device placement: compute on GPU if available, then move small result to CPU
    original_device = delta.device
    compute_device = delta.device
    
    if use_gpu_compute and torch.cuda.is_available() and not delta.is_cuda:
        compute_device = torch.device('cuda')
        delta = delta.to(compute_device)
    
    # Mixed precision for memory efficiency (norms only need ~1e-4 precision)
    if use_mixed_precision and delta.dtype == torch.float32:
        delta = delta.half()
    
    if delta.dim() == 1:
        # 1D: Keep as-is, no need to pool
        magnitude_diff = torch.abs(delta)
        direction_diff = delta
    elif delta.dim() == 2:
        # 2D (Linear weights): Pool along one axis to preserve structure
        if delta.size(0) <= delta.size(1):
            magnitude_diff = torch.norm(delta, p=2, dim=0)  # Shape: [features]
            direction_diff = delta.mean(dim=0)              # Shape: [features]
        else:
            magnitude_diff = torch.norm(delta, p=2, dim=1)  # Shape: [features]
            direction_diff = delta.mean(dim=1)              # Shape: [features]
    elif delta.dim() == 4:
        # 4D (Conv weights): Pool spatial dims but preserve channel structure
        # Use shared pooling kernel for efficiency
        pooling_kernel = _get_pooling_kernel((1, 1))
        spatial_pooled = pooling_kernel(delta).squeeze(-1).squeeze(-1)
        magnitude_diff = torch.norm(spatial_pooled, p=2, dim=1)  # Shape: [C_out]
        direction_diff = spatial_pooled.mean(dim=1)               # Shape: [C_out]
    else:
        # Higher-D: Pool to preserve primary structure
        pooled_dims = tuple(range(2, delta.dim()))
        pooled_delta = delta.mean(dim=pooled_dims)
        
        # Add dimension check before norm calculation
        if pooled_delta.dim() > 1:
            magnitude_diff = torch.norm(pooled_delta, p=2, dim=1)  # Shape: [dim0]
            direction_diff = pooled_delta.mean(dim=1)               # Shape: [dim0]
        else:
            # 1D case: use absolute value instead of norm
            magnitude_diff = torch.abs(pooled_delta)               # Shape: [dim0]
            direction_diff = pooled_delta                          # Shape: [dim0]
    
    # Convert back to float32 and move small results to original device for compatibility
    if use_mixed_precision and magnitude_diff.dtype == torch.float16:
        magnitude_diff = magnitude_diff.float()
        direction_diff = direction_diff.float()
    
    if compute_device != original_device:
        magnitude_diff = magnitude_diff.to(original_device)
        direction_diff = direction_diff.to(original_device)
    
    return magnitude_diff, direction_diff


# Fast tensor ranking (non-JIT for performance)
def _fast_tensor_ranking(tensor: torch.Tensor) -> torch.Tensor:
    """Fast ranking for tensors that fit in memory (non-JIT)"""
    num_models, num_features = tensor.shape
    device = tensor.device
    
    # Sort and create ranks
    sort_indices = torch.argsort(tensor, dim=1, descending=False, stable=True)
    ranks = torch.arange(num_features, device=device, dtype=torch.float32) / num_features
    ranks = ranks.unsqueeze(0).expand(num_models, -1)
    
    # Apply ranks
    result = torch.zeros_like(tensor)
    result.scatter_(1, sort_indices, ranks)
    return result


def _chunked_rank_computation(tensor: torch.Tensor, chunk_size: int = 32) -> torch.Tensor:
    """Non-JIT chunked ranking to reduce memory spikes"""
    device = tensor.device
    num_models, num_features = tensor.shape
    
    if num_features <= chunk_size:
        # Small tensor - use non-JIT fast version
        return _fast_tensor_ranking(tensor)
    
    # Large tensor - process in chunks
    result = torch.zeros_like(tensor)
    for start_idx in range(0, num_features, chunk_size):
        end_idx = min(start_idx + chunk_size, num_features)
        chunk = tensor[:, start_idx:end_idx]
        chunk_size_actual = end_idx - start_idx
        
        sort_indices = torch.argsort(chunk, dim=1, descending=False, stable=True)
        ranks = torch.arange(chunk_size_actual, device=device, dtype=torch.float32) / chunk_size_actual
        ranks = ranks.unsqueeze(0).expand(num_models, -1)
        
        chunk_result = torch.zeros_like(chunk)
        chunk_result.scatter_(1, sort_indices, ranks)
        result[:, start_idx:end_idx] = chunk_result
        
    return result


def _rank_per_param_magnitude_or_direction_within_model(models_to_merge_param_diff):
    """Rank the magnitude or direction within model with memory optimization"""
    # Use chunked computation for large tensors to reduce memory spikes
    if models_to_merge_param_diff.numel() > 100_000:  # ~400KB threshold
        return _chunked_rank_computation(models_to_merge_param_diff)
    
    # Original implementation for smaller tensors
    device = models_to_merge_param_diff.device
    
    sort_indices = torch.argsort(models_to_merge_param_diff, dim=1, descending=False, stable=True)
    within_model_significance = (torch.arange(models_to_merge_param_diff.shape[1], device=device) / models_to_merge_param_diff.shape[1]).repeat(
        models_to_merge_param_diff.shape[0]
    ).reshape(models_to_merge_param_diff.shape)
    
    models_to_merge_param_within_model_significance = torch.zeros(within_model_significance.shape, device=device)
    models_to_merge_param_within_model_significance = torch.scatter(
        input=models_to_merge_param_within_model_significance,
        dim=1,
        index=sort_indices,
        src=within_model_significance
    )
    
    return models_to_merge_param_within_model_significance


# Core importance scoring (non-JIT for performance)
def _compute_importance_scores_core(input_significance_tensor: torch.Tensor, 
                                       above_average_value_ratio: float = 1.0, 
                                       score_calibration_value: float = 1.0):
    """Core importance scoring logic - pure tensor operations for vmap compatibility"""
    # Handle scalar or 1D tensors (e.g. clip.logit_scale) as uniform scores
    if input_significance_tensor.ndim <= 1:
        return torch.ones_like(input_significance_tensor), torch.tensor(True, device=input_significance_tensor.device)
    
    # Check if input is uniform (all values nearly identical) - use mask instead of if/else
    tensor_variance = torch.var(input_significance_tensor)
    is_uniform = tensor_variance < 1e-8
    
    # Uniform scores fallback (always computed, selected by mask)
    uniform_scores = torch.full_like(input_significance_tensor, 1.0 / input_significance_tensor.numel())
    
    # Non-uniform scores path - compute softmax scores
    # Handle dimension cases using tensor operations instead of if/else
    dim_0_softmax = torch.softmax(input_significance_tensor, dim=0)  # Multi-model case
    dim_1_softmax = torch.softmax(input_significance_tensor, dim=1)  # Single model case
    
    # Select appropriate softmax based on tensor dimensions (pure tensor operation)
    is_single_model = (input_significance_tensor.dim() == 2) and (input_significance_tensor.size(0) == 1)
    importance_scores = torch.where(
        torch.tensor(is_single_model, device=input_significance_tensor.device),
        dim_1_softmax,
        dim_0_softmax
    )
    
    # Apply above-average boost (pure tensor operations)
    if above_average_value_ratio != 1.0 and above_average_value_ratio > 0.0:
        mean_score = importance_scores.mean()
        above_avg_mask = importance_scores > mean_score
        importance_scores = torch.where(
            above_avg_mask,
            importance_scores * above_average_value_ratio,
            importance_scores
        )
        
        # Renormalize to maintain probability distribution
        total = importance_scores.sum()
        importance_scores = torch.where(
            total > 1e-8,
            importance_scores / total,
            importance_scores
        )
    
    # Apply score calibration (pure tensor operations)
    if score_calibration_value != 1.0 and score_calibration_value > 0.0:
        importance_scores = importance_scores * score_calibration_value
        
        # Renormalize
        total = importance_scores.sum()
        importance_scores = torch.where(
            total > 1e-8,
            importance_scores / total,
            importance_scores
        )
    
    # Select between uniform and non-uniform scores based on variance mask
    final_scores = torch.where(is_uniform, uniform_scores, importance_scores)
    
    return final_scores, is_uniform  # Return scores and uniform flag tensor


def _compute_importance_scores(input_significance_tensor, above_average_value_ratio=1.0, score_calibration_value=1.0):
    """
    Compute importance scores (fixed WIDEN algorithm) - Non-JIT optimized
    Returns: (importance_scores, is_uniform_flag)
    """
    # Use non-JIT core for maximum performance
    return _compute_importance_scores_core(input_significance_tensor, above_average_value_ratio, score_calibration_value)


def smart_device_management(tensor: torch.Tensor, target_device: torch.device) -> torch.Tensor:
    """
    Smart device management for tensor operations with memory optimization
    
    Args:
        tensor: Input tensor to move
        target_device: Target device to move tensor to
        
    Returns:
        Tensor moved to target device with proper memory management
    """
    if tensor.device == target_device:
        return tensor
    
    # Move tensor to target device with non-blocking transfer for efficiency
    try:
        if target_device.type == 'cuda':
            return tensor.to(target_device, non_blocking=True)
        else:
            return tensor.to(target_device)
    except RuntimeError as e:
        # Fallback for memory issues
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return tensor.to(target_device)


def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list = None):
    """
    Get the names of parameters that need to be merged
    
    Args:
        input_param_names: list, names of input parameters
        exclude_param_names_regex: list, regular expression patterns (strings or compiled) to exclude
        
    Returns:
        list of parameter names to merge
    """
    import re
    
    if not exclude_param_names_regex:
        return input_param_names.copy()
    
    # Compile regex patterns once for performance
    compiled_patterns = [
        re.compile(pattern) if isinstance(pattern, str) else pattern
        for pattern in exclude_param_names_regex
    ]
    
    param_names_to_merge = []
    for param_name in input_param_names:
        # Use search() instead of match() to be consistent (match only matches from beginning)
        exclude = any(pattern.search(param_name) for pattern in compiled_patterns)
        if not exclude:
            param_names_to_merge.append(param_name)
    
    return param_names_to_merge