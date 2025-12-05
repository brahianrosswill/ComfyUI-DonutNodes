"""
WIDEN Merge Diagnostic Functions

This module contains diagnostic and hyperparameter tuning functions extracted from DonutWidenMerge.py.
These functions provide:

1. Merge quality metrics and sanity checks
2. Human-readable diagnostic output
3. Hyperparameter optimization (grid search and Optuna)
4. Utility functions for data sanitization

Functions:
- compute_merge_sanity_metrics(): Compute metrics to detect over-merging or catastrophic changes
- print_merge_diagnostics(): Print human-readable merge diagnostics
- tune_merge_hyperparameters(): Hyperparameter search for optimal merge settings
- sanitize_strength_distribution(): Sanitize strength distribution for display
"""

import torch
import itertools
from .constants import OPTUNA_AVAILABLE
from .logging_config import diagnostic_logger, widen_logger

# Optional dependency imports
if OPTUNA_AVAILABLE:
    import optuna


def compute_merge_sanity_metrics(base_model, merged_model, param_names=None):
    """
    Compute sanity check metrics to detect over-merging or catastrophic changes.
    
    Returns:
        dict: Metrics including L2 change ratio, per-layer changes, etc.
    """
    metrics = {
        'total_norm_old': 0.0,
        'total_norm_delta': 0.0,
        'max_param_change': 0.0,
        'layer_changes': {},
        'suspicious_layers': []
    }
    
    base_state = base_model.state_dict() if hasattr(base_model, 'state_dict') else base_model
    merged_state = merged_model.state_dict() if hasattr(merged_model, 'state_dict') else merged_model
    
    if param_names is None:
        param_names = list(base_state.keys())
    
    for name in param_names:
        if name not in base_state or name not in merged_state:
            continue
            
        base_param = base_state[name]
        merged_param = merged_state[name]
        
        if not isinstance(base_param, torch.Tensor) or not isinstance(merged_param, torch.Tensor):
            continue
            
        # Compute norms
        old_norm = base_param.norm().item()
        delta_norm = (merged_param - base_param).norm().item()
        
        metrics['total_norm_old'] += old_norm
        metrics['total_norm_delta'] += delta_norm
        
        # Track per-layer changes
        if old_norm > 0:
            relative_change = delta_norm / old_norm
            metrics['layer_changes'][name] = relative_change
            metrics['max_param_change'] = max(metrics['max_param_change'], relative_change)
            
            # Flag suspicious changes (>50% change in a layer)
            if relative_change > 0.5:
                metrics['suspicious_layers'].append((name, relative_change))
    
    # Compute global change ratio
    if metrics['total_norm_old'] > 0:
        metrics['global_change_ratio'] = metrics['total_norm_delta'] / metrics['total_norm_old']
    else:
        metrics['global_change_ratio'] = 0.0
    
    return metrics


def print_merge_diagnostics(metrics):
    """Print human-readable merge diagnostics"""
    ratio = metrics['global_change_ratio']
    print(f"\n🔍 MERGE DIAGNOSTICS:")
    print(f"  Global change ratio: Δ‖θ‖/‖θ‖ = {ratio:.3%}")
    
    if ratio > 1.0:
        print(f"  ⚠️  WARNING: Very high change ratio (>{100:.0%}) - possible over-merge")
    elif ratio > 0.5:
        print(f"  🟡 CAUTION: High change ratio (>{50:.0%}) - verify results")
    elif ratio > 0.1:
        print(f"  ✅ NORMAL: Moderate changes ({ratio:.1%})")
    else:
        print(f"  ℹ️  LOW: Minimal changes ({ratio:.1%}) - conservative merge")
    
    if metrics['suspicious_layers']:
        print(f"  🚨 {len(metrics['suspicious_layers'])} layers with >50% change:")
        for name, change in metrics['suspicious_layers'][:3]:  # Show first 3
            print(f"    {name}: {change:.1%}")
        if len(metrics['suspicious_layers']) > 3:
            print(f"    ... and {len(metrics['suspicious_layers']) - 3} more")


def tune_merge_hyperparameters(
    merge_function, 
    base_model, 
    models_to_merge,
    evaluate_function=None,
    validation_data=None,
    method="grid",  # "grid" or "optuna"
    n_trials=20,
    **fixed_kwargs
):
    """
    Hyperparameter search for optimal merge settings.
    
    Args:
        merge_function: The merge function to optimize
        base_model: Base model for merging
        models_to_merge: List of models to merge
        evaluate_function: Function that takes (model, val_data) -> score (higher=better)
        validation_data: Validation dataset for evaluation
        method: "grid" for grid search, "optuna" for Bayesian optimization
        n_trials: Number of trials for Optuna
        **fixed_kwargs: Fixed parameters to pass to merge function
    """
    if evaluate_function is None:
        diagnostic_logger.warning("No evaluation function provided - using dummy evaluation")
        def dummy_eval(model, data):
            # Dummy evaluation: return negative of global change ratio (prefer smaller changes)
            metrics = compute_merge_sanity_metrics(base_model, model)
            return -metrics['global_change_ratio']
        evaluate_function = dummy_eval
    
    best_result = {"score": float('-inf'), "params": None}
    
    if method == "grid":
        # Grid search with reasonable parameter ranges
        param_grid = {
            "merge_strength": [0.5, 1.0, 1.5],
            "importance_threshold": [0.5, 1.0, 2.0],
            "importance_boost": [1.5, 2.5, 3.5],
            "skip_threshold": [0.0, 1e-5, 1e-4, 1e-3],
        }
        
        print(f"🔍 Starting grid search with {len(list(itertools.product(*param_grid.values())))} combinations...")
        
        for i, (ms, it, ib, st) in enumerate(itertools.product(*param_grid.values())):
            params = {
                "merge_strength": ms,
                "importance_threshold": it, 
                "importance_boost": ib,
                "skip_threshold": st,
                **fixed_kwargs
            }
            
            try:
                merged_model, _ = merge_function(
                    merged_model=base_model,
                    models_to_merge=models_to_merge,
                    **params
                )
                
                score = evaluate_function(merged_model, validation_data)
                
                if score > best_result["score"]:
                    best_result = {"score": score, "params": params}
                    
                print(f"  Trial {i+1}: score={score:.4f}, params={params}")
                
            except Exception as e:
                diagnostic_logger.warning(f"Grid search trial {i+1} failed: {e}")
                continue
                
    elif method == "optuna" and OPTUNA_AVAILABLE:
        print(f"🔍 Starting Optuna optimization with {n_trials} trials...")
        
        def objective(trial):
            params = {
                "merge_strength": trial.suggest_float("merge_strength", 0.1, 2.0),
                "importance_threshold": trial.suggest_float("importance_threshold", 0.1, 5.0),
                "importance_boost": trial.suggest_float("importance_boost", 1.0, 5.0),
                "skip_threshold": trial.suggest_loguniform("skip_threshold", 1e-6, 1e-2),
                **fixed_kwargs
            }
            
            try:
                merged_model, _ = merge_function(
                    merged_model=base_model,
                    models_to_merge=models_to_merge, 
                    **params
                )
                
                score = evaluate_function(merged_model, validation_data)
                return score
                
            except Exception as e:
                diagnostic_logger.warning(f"Optuna trial failed: {e}")
                return float('-inf')
        
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_result = {
            "score": study.best_value,
            "params": {**study.best_params, **fixed_kwargs}
        }
        
    else:
        if method == "optuna":
            print("⚠️ Optuna not available, falling back to grid search")
        return tune_merge_hyperparameters(
            merge_function, base_model, models_to_merge, evaluate_function,
            validation_data, method="grid", **fixed_kwargs
        )
    
    print(f"\n🎯 BEST HYPERPARAMETERS:")
    print(f"  Score: {best_result['score']:.4f}")
    for key, value in best_result['params'].items():
        print(f"  {key}: {value}")
    
    return best_result


def sanitize_strength_distribution(strength_dist):
    """
    Centralized helper to sanitize strength distribution for display.
    Returns a clean dict for formatting, with N/A handling for empty data.
    """
    if strength_dist['count'] == 0:
        return {
            'min_used': None,  # Will display as "N/A"
            'max_used': None,  # Will display as "N/A"
            'mean': None,      # Will display as "N/A"
            'count': 0,
            'display_text': "N/A (no dynamic adjustments)"
        }
    return {
        'min_used': strength_dist['min_used'],
        'max_used': strength_dist['max_used'],
        'mean': strength_dist['mean'],
        'count': strength_dist['count'],
        'display_text': f"{strength_dist['min_used']:.3f}-{strength_dist['max_used']:.3f} (avg {strength_dist['mean']:.3f})"
    }