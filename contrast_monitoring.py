"""
Energy and contrast monitoring utilities for merge quality assessment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .logging_config import diagnostic_logger

def compute_layer_energy_stats(model, layer_names: Optional[List[str]] = None) -> Dict:
    """
    Compute energy statistics for model layers.
    
    Measures:
    - Frobenius norm (overall energy)
    - Spectral norm (largest singular value)
    - Nuclear norm (sum of singular values)
    - Parameter variance
    
    Args:
        model: Model to analyze
        layer_names: Specific layers to analyze (all if None)
        
    Returns:
        Dictionary of energy statistics per layer
    """
    state_dict = model.state_dict() if hasattr(model, 'state_dict') else model
    
    if layer_names is None:
        layer_names = list(state_dict.keys())
    
    energy_stats = {}
    
    for name in layer_names:
        if name not in state_dict:
            continue
            
        param = state_dict[name]
        if not isinstance(param, torch.Tensor):
            continue
        
        stats = {}
        
        # Frobenius norm (L2 norm)
        stats['frobenius_norm'] = torch.norm(param, p='fro').item()
        
        # Parameter variance
        stats['variance'] = torch.var(param).item()
        
        # Parameter mean absolute value
        stats['mean_abs'] = torch.mean(torch.abs(param)).item()
        
        # For 2D+ tensors, compute spectral properties
        if param.ndim >= 2:
            # Reshape to matrix for SVD
            if param.ndim > 2:
                matrix = param.view(param.shape[0], -1)
            else:
                matrix = param
            
            try:
                # Compute singular values
                _, S, _ = torch.svd(matrix)
                
                # Spectral norm (largest singular value)
                stats['spectral_norm'] = S[0].item() if len(S) > 0 else 0.0
                
                # Nuclear norm (sum of singular values)
                stats['nuclear_norm'] = torch.sum(S).item()
                
                # Condition number (ratio of largest to smallest singular value)
                if len(S) > 1 and S[-1] > 1e-8:
                    stats['condition_number'] = (S[0] / S[-1]).item()
                else:
                    stats['condition_number'] = float('inf')
                    
                # Effective rank (number of significant singular values)
                threshold = 0.01 * S[0]  # 1% of largest singular value
                stats['effective_rank'] = torch.sum(S > threshold).item()
                
            except Exception as e:
                diagnostic_logger.debug(f"SVD failed for {name}: {e}")
                stats['spectral_norm'] = 0.0
                stats['nuclear_norm'] = 0.0
                stats['condition_number'] = 1.0
                stats['effective_rank'] = 1
        
        energy_stats[name] = stats
    
    return energy_stats

def compute_activation_contrast_stats(model, calibration_data, max_batches: int = 3) -> Dict:
    """
    Compute contrast statistics from model activations on calibration data.
    
    Measures:
    - Mean activation magnitude
    - Activation variance
    - Local contrast (gradient magnitude)
    - Dynamic range
    
    Args:
        model: Model to analyze
        calibration_data: Data to run through model
        max_batches: Maximum batches to process
        
    Returns:
        Dictionary of activation statistics per layer
    """
    diagnostic_logger.info("Computing activation contrast statistics")
    
    # Find layers to monitor
    target_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            target_layers[name] = module
    
    if not target_layers:
        diagnostic_logger.warning("No suitable layers found for activation monitoring")
        return {}
    
    activation_stats = {name: {'activations': []} for name in target_layers}
    
    # Hook to collect activations
    def make_activation_hook(layer_name):
        def hook(module, input, output):
            # Store activation statistics
            if isinstance(output, torch.Tensor):
                # Compute basic stats
                mean_act = torch.mean(torch.abs(output))
                var_act = torch.var(output)
                max_act = torch.max(torch.abs(output))
                min_act = torch.min(torch.abs(output))
                
                # Dynamic range
                dynamic_range = max_act - min_act if max_act > min_act else 0.0
                
                # Local contrast (gradient magnitude for spatial tensors)
                local_contrast = 0.0
                if output.ndim >= 3:  # Has spatial dimensions
                    try:
                        if output.ndim == 4:  # 2D spatial
                            grad_x = torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1])
                            grad_y = torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :])
                            local_contrast = torch.mean(grad_x) + torch.mean(grad_y)
                        elif output.ndim == 3:  # 1D spatial
                            grad = torch.abs(output[:, :, 1:] - output[:, :, :-1])
                            local_contrast = torch.mean(grad)
                    except:
                        pass
                
                stats = {
                    'mean_activation': mean_act.item(),
                    'activation_variance': var_act.item(),
                    'dynamic_range': dynamic_range.item(),
                    'local_contrast': local_contrast.item() if isinstance(local_contrast, torch.Tensor) else local_contrast
                }
                
                activation_stats[layer_name]['activations'].append(stats)
        
        return hook
    
    # Register hooks
    hooks = []
    for layer_name, layer in target_layers.items():
        hook = layer.register_forward_hook(make_activation_hook(layer_name))
        hooks.append(hook)
    
    # Run calibration data
    model.eval()
    with torch.no_grad():
        batch_count = 0
        for batch in calibration_data:
            if batch_count >= max_batches:
                break
                
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
                
            if not isinstance(x, torch.Tensor):
                continue
            
            try:
                model(x)
                batch_count += 1
            except Exception as e:
                diagnostic_logger.warning(f"Activation monitoring batch failed: {e}")
                continue
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Aggregate statistics
    aggregated_stats = {}
    for layer_name, data in activation_stats.items():
        if not data['activations']:
            continue
            
        # Average across all batches
        stats = {}
        for key in ['mean_activation', 'activation_variance', 'dynamic_range', 'local_contrast']:
            values = [act[key] for act in data['activations']]
            stats[f'avg_{key}'] = sum(values) / len(values) if values else 0.0
            stats[f'std_{key}'] = torch.std(torch.tensor(values)).item() if len(values) > 1 else 0.0
        
        aggregated_stats[layer_name] = stats
    
    diagnostic_logger.info(f"Activation contrast stats computed for {len(aggregated_stats)} layers")
    return aggregated_stats

def compare_model_energy_contrast(model_before, model_after, calibration_data=None, layer_names: Optional[List[str]] = None) -> Dict:
    """
    Compare energy and contrast before/after processing.
    
    Args:
        model_before: Model state before processing
        model_after: Model state after processing  
        calibration_data: Optional data for activation analysis
        layer_names: Layers to analyze
        
    Returns:
        Comprehensive comparison statistics
    """
    diagnostic_logger.info("Computing energy/contrast comparison")
    
    comparison = {}
    
    # Parameter energy comparison
    energy_before = compute_layer_energy_stats(model_before, layer_names)
    energy_after = compute_layer_energy_stats(model_after, layer_names)
    
    param_comparison = {}
    common_layers = set(energy_before.keys()) & set(energy_after.keys())
    
    for layer in common_layers:
        layer_comp = {}
        
        for metric in ['frobenius_norm', 'spectral_norm', 'variance', 'mean_abs']:
            if metric in energy_before[layer] and metric in energy_after[layer]:
                before_val = energy_before[layer][metric]
                after_val = energy_after[layer][metric]
                
                # Compute relative change
                if before_val > 1e-8:
                    relative_change = (after_val - before_val) / before_val
                else:
                    relative_change = 0.0
                
                layer_comp[f'{metric}_before'] = before_val
                layer_comp[f'{metric}_after'] = after_val
                layer_comp[f'{metric}_change'] = relative_change
        
        param_comparison[layer] = layer_comp
    
    comparison['parameter_energy'] = param_comparison
    
    # Activation contrast comparison (if calibration data provided)
    if calibration_data is not None:
        try:
            contrast_before = compute_activation_contrast_stats(model_before, calibration_data)
            contrast_after = compute_activation_contrast_stats(model_after, calibration_data)
            
            activation_comparison = {}
            common_act_layers = set(contrast_before.keys()) & set(contrast_after.keys())
            
            for layer in common_act_layers:
                layer_comp = {}
                
                for metric in ['avg_mean_activation', 'avg_local_contrast', 'avg_dynamic_range']:
                    if metric in contrast_before[layer] and metric in contrast_after[layer]:
                        before_val = contrast_before[layer][metric]
                        after_val = contrast_after[layer][metric]
                        
                        if before_val > 1e-8:
                            relative_change = (after_val - before_val) / before_val
                        else:
                            relative_change = 0.0
                        
                        layer_comp[f'{metric}_before'] = before_val
                        layer_comp[f'{metric}_after'] = after_val
                        layer_comp[f'{metric}_change'] = relative_change
                
                activation_comparison[layer] = layer_comp
            
            comparison['activation_contrast'] = activation_comparison
            
        except Exception as e:
            diagnostic_logger.warning(f"Activation contrast comparison failed: {e}")
    
    return comparison

def print_energy_contrast_summary(comparison_stats: Dict):
    """Print a human-readable summary of energy/contrast changes."""
    print("\n🔋 ENERGY & CONTRAST ANALYSIS:")
    
    if 'parameter_energy' in comparison_stats:
        param_stats = comparison_stats['parameter_energy']
        
        # Global statistics
        frobenius_changes = []
        contrast_changes = []
        
        for layer, stats in param_stats.items():
            if 'frobenius_norm_change' in stats:
                frobenius_changes.append(stats['frobenius_norm_change'])
            if 'variance_change' in stats:
                contrast_changes.append(stats['variance_change'])
        
        if frobenius_changes:
            avg_energy_change = sum(frobenius_changes) / len(frobenius_changes)
            print(f"  📊 Average Frobenius norm change: {avg_energy_change:+.2%}")
            
            if avg_energy_change > 0.1:
                print("  ⚡ INCREASED layer energy - may enhance detail")
            elif avg_energy_change < -0.1:
                print("  📉 DECREASED layer energy - may reduce noise but lose detail")
            else:
                print("  ⚖️ STABLE layer energy - good preservation")
        
        if contrast_changes:
            avg_contrast_change = sum(contrast_changes) / len(contrast_changes)
            print(f"  🎨 Average parameter variance change: {avg_contrast_change:+.2%}")
            
            if avg_contrast_change > 0.1:
                print("  ✨ INCREASED parameter contrast - enhanced detail preservation")
            elif avg_contrast_change < -0.1:
                print("  🌫️ DECREASED parameter contrast - potential detail loss")
    
    if 'activation_contrast' in comparison_stats:
        act_stats = comparison_stats['activation_contrast']
        
        local_contrast_changes = []
        dynamic_range_changes = []
        
        for layer, stats in act_stats.items():
            if 'avg_local_contrast_change' in stats:
                local_contrast_changes.append(stats['avg_local_contrast_change'])
            if 'avg_dynamic_range_change' in stats:
                dynamic_range_changes.append(stats['avg_dynamic_range_change'])
        
        if local_contrast_changes:
            avg_local_contrast = sum(local_contrast_changes) / len(local_contrast_changes)
            print(f"  🖼️ Average local contrast change: {avg_local_contrast:+.2%}")
        
        if dynamic_range_changes:
            avg_dynamic_range = sum(dynamic_range_changes) / len(dynamic_range_changes)
            print(f"  📈 Average dynamic range change: {avg_dynamic_range:+.2%}")

def monitor_refinement_quality(model_states: List[Tuple[str, any]], calibration_data=None) -> Dict:
    """
    Monitor quality metrics across multiple refinement steps.
    
    Args:
        model_states: List of (step_name, model_state) tuples
        calibration_data: Optional calibration data for activation analysis
        
    Returns:
        Dictionary tracking quality metrics across steps
    """
    if len(model_states) < 2:
        return {}
    
    diagnostic_logger.info(f"Monitoring refinement quality across {len(model_states)} steps")
    
    quality_progression = {}
    
    # Compare each step to the initial state
    initial_name, initial_model = model_states[0]
    
    for i, (step_name, current_model) in enumerate(model_states[1:], 1):
        comparison = compare_model_energy_contrast(initial_model, current_model, calibration_data)
        quality_progression[f"{i:02d}_{step_name}"] = comparison
        
        diagnostic_logger.info(f"Step {i} ({step_name}): quality metrics computed")
    
    return quality_progression