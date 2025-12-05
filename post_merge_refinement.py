"""
Post-merge refinement strategies to recover edge, detail, and contrast
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from .logging_config import diagnostic_logger, performance_logger
from .memory_management import MemoryEfficientContext, monitor_memory_usage

def frobenius_rescale_layers(merged_model, original_model, param_names: Optional[List[str]] = None):
    """
    Per-layer rescaling via Frobenius norm to preserve layer energy.
    
    For each weight tensor W, compute α = ||W_orig||_F / (||W_merged||_F + ε)
    and rescale: W_merged = W_merged * α
    
    This prevents "dulling" of high-frequency filters by under-scaling.
    
    Args:
        merged_model: The merged model to rescale
        original_model: Original model for reference norms
        param_names: Optional list of parameter names to process
    """
    diagnostic_logger.info("Applying Frobenius norm rescaling to preserve layer energy")
    
    merged_state = merged_model.state_dict() if hasattr(merged_model, 'state_dict') else merged_model
    original_state = original_model.state_dict() if hasattr(original_model, 'state_dict') else original_model
    
    if param_names is None:
        param_names = list(merged_state.keys())
    
    rescale_stats = {'rescaled_count': 0, 'avg_scale_factor': 0.0, 'scale_factors': []}
    
    with torch.no_grad():
        for name in param_names:
            if name not in merged_state or name not in original_state:
                continue
                
            merged_param = merged_state[name]
            original_param = original_state[name]
            
            if not isinstance(merged_param, torch.Tensor) or not isinstance(original_param, torch.Tensor):
                continue
                
            if merged_param.shape != original_param.shape:
                continue
            
            # Compute Frobenius norms
            orig_norm = torch.norm(original_param, p='fro')
            merged_norm = torch.norm(merged_param, p='fro')
            
            # Avoid division by zero
            eps = 1e-8
            if merged_norm > eps and orig_norm > eps:
                alpha = orig_norm / (merged_norm + eps)
                
                # Apply rescaling
                merged_state[name] = merged_param * alpha
                
                rescale_stats['rescaled_count'] += 1
                rescale_stats['scale_factors'].append(alpha.item())
                
                diagnostic_logger.debug(f"Rescaled {name}: α={alpha:.4f} (orig_norm={orig_norm:.4f}, merged_norm={merged_norm:.4f})")
    
    if rescale_stats['scale_factors']:
        rescale_stats['avg_scale_factor'] = sum(rescale_stats['scale_factors']) / len(rescale_stats['scale_factors'])
        
    diagnostic_logger.info(f"Frobenius rescaling complete: {rescale_stats['rescaled_count']} layers, avg scale factor: {rescale_stats['avg_scale_factor']:.4f}")
    
    return rescale_stats

def covariance_aware_conv_alignment(base_model, other_model, calibration_data, max_batches: int = 5):
    """
    Covariance-aware alignment for convolutional layers.
    
    For each conv output channel, records activations on calibration data
    and aligns covariance matrices using Procrustes transforms.
    
    Args:
        base_model: Base model for reference
        other_model: Model to align to base
        calibration_data: Data loader for computing activations
        max_batches: Maximum calibration batches to use
    
    Returns:
        Dictionary of alignment transforms per layer
    """
    diagnostic_logger.info("Computing covariance-aware conv alignment")
    
    # Find all conv layers
    conv_layers = {}
    for name, module in base_model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_layers[name] = module
    
    if not conv_layers:
        diagnostic_logger.warning("No conv layers found for covariance alignment")
        return {}
    
    alignment_transforms = {}
    
    with torch.no_grad():
        # Hook to collect activations
        activations = {'base': {}, 'other': {}}
        
        def make_hook(model_key, layer_name):
            def hook(module, input, output):
                if layer_name not in activations[model_key]:
                    activations[model_key][layer_name] = []
                # Store flattened channel activations
                batch_size = output.shape[0]
                channels = output.shape[1]
                spatial_dims = output.shape[2:]
                
                # Flatten spatial dimensions, keep channels separate
                flat_acts = output.view(batch_size, channels, -1)
                activations[model_key][layer_name].append(flat_acts.cpu())
            return hook
        
        # Register hooks
        hooks = []
        for layer_name in conv_layers:
            base_layer = dict(base_model.named_modules())[layer_name]
            other_layer = dict(other_model.named_modules())[layer_name]
            
            hooks.append(base_layer.register_forward_hook(make_hook('base', layer_name)))
            hooks.append(other_layer.register_forward_hook(make_hook('other', layer_name)))
        
        # Run calibration data
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
            
            # Forward through both models
            try:
                base_model(x)
                other_model(x)
                batch_count += 1
            except Exception as e:
                diagnostic_logger.warning(f"Calibration batch failed: {e}")
                continue
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    # Compute covariance alignment for each layer
    for layer_name in conv_layers:
        if layer_name not in activations['base'] or layer_name not in activations['other']:
            continue
            
        try:
            # Concatenate all batch activations
            base_acts = torch.cat(activations['base'][layer_name], dim=0)  # [total_samples, channels, spatial]
            other_acts = torch.cat(activations['other'][layer_name], dim=0)
            
            # Compute channel-wise covariance matrices
            base_cov = compute_channel_covariance(base_acts)
            other_cov = compute_channel_covariance(other_acts)
            
            # Compute Procrustes alignment transform
            alignment_transform = procrustes_alignment(other_cov, base_cov)
            alignment_transforms[layer_name] = alignment_transform
            
            diagnostic_logger.debug(f"Computed covariance alignment for {layer_name}")
            
        except Exception as e:
            diagnostic_logger.warning(f"Covariance alignment failed for {layer_name}: {e}")
            continue
    
    diagnostic_logger.info(f"Covariance alignment computed for {len(alignment_transforms)} layers")
    return alignment_transforms

def compute_channel_covariance(activations: torch.Tensor) -> torch.Tensor:
    """
    Compute channel-wise covariance matrix from activations.
    
    Args:
        activations: [samples, channels, spatial] tensor
        
    Returns:
        [channels, channels] covariance matrix
    """
    samples, channels, spatial = activations.shape
    
    # Flatten spatial and samples together
    flat_acts = activations.view(-1, channels)  # [samples * spatial, channels]
    
    # Center the data
    mean_acts = flat_acts.mean(dim=0, keepdim=True)
    centered = flat_acts - mean_acts
    
    # Compute covariance
    cov = torch.mm(centered.t(), centered) / (flat_acts.shape[0] - 1)
    
    return cov

def procrustes_alignment(source_cov: torch.Tensor, target_cov: torch.Tensor) -> torch.Tensor:
    """
    Compute Procrustes alignment transform to align source covariance to target.
    
    Args:
        source_cov: Source covariance matrix to transform
        target_cov: Target covariance matrix to match
        
    Returns:
        Orthogonal transformation matrix
    """
    # Use SVD-based Procrustes solution
    try:
        # Compute square roots of covariance matrices
        target_sqrt = matrix_sqrt(target_cov)
        source_sqrt = matrix_sqrt(source_cov)
        
        # Compute cross-covariance
        cross_cov = torch.mm(target_sqrt, source_sqrt)
        
        # SVD of cross-covariance
        U, S, V = torch.svd(cross_cov)
        
        # Optimal orthogonal transform
        R = torch.mm(U, V.t())
        
        return R
        
    except Exception as e:
        diagnostic_logger.warning(f"Procrustes alignment failed: {e}, using identity")
        return torch.eye(source_cov.shape[0], device=source_cov.device, dtype=source_cov.dtype)

def matrix_sqrt(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute matrix square root using eigendecomposition."""
    eigenvals, eigenvecs = torch.symeig(matrix, eigenvectors=True)
    eigenvals = torch.clamp(eigenvals, min=eps)  # Ensure positive
    sqrt_eigenvals = torch.sqrt(eigenvals)
    
    return torch.mm(torch.mm(eigenvecs, torch.diag(sqrt_eigenvals)), eigenvecs.t())

def low_rank_residual_injection(base_weights: torch.Tensor, other_weights: torch.Tensor, 
                               rank: int = 4, strength: float = 1.0) -> torch.Tensor:
    """
    Low-rank residual injection using SVD.
    
    Takes difference Δ = W_other - W_base, performs SVD, keeps only top-k
    singular vectors, and merges using only the most salient residual structure.
    
    Args:
        base_weights: Base weight tensor
        other_weights: Other weight tensor  
        rank: Number of top singular vectors to keep
        strength: Injection strength
        
    Returns:
        Merged weights with low-rank residual injection
    """
    if base_weights.shape != other_weights.shape:
        return base_weights
    
    # Compute residual
    delta = other_weights - base_weights
    
    # Reshape to matrix for SVD
    original_shape = delta.shape
    if delta.ndim > 2:
        out_features = delta.shape[0]
        delta_matrix = delta.view(out_features, -1)
    else:
        delta_matrix = delta
    
    try:
        # SVD decomposition
        U, S, V = torch.svd(delta_matrix)
        
        # Keep only top-k components
        k = min(rank, min(U.shape[1], V.shape[0]))
        
        # Reconstruct with reduced rank
        delta_low_rank = torch.mm(torch.mm(U[:, :k], torch.diag(S[:k])), V[:, :k].t())
        
        # Reshape back to original shape
        delta_low_rank = delta_low_rank.view(original_shape)
        
        # Inject with strength
        merged = base_weights + strength * delta_low_rank
        
        diagnostic_logger.debug(f"Low-rank injection: rank={k}, strength={strength}, delta_norm={torch.norm(delta):.4f}")
        
        return merged
        
    except Exception as e:
        diagnostic_logger.warning(f"Low-rank injection failed: {e}")
        return base_weights

def apply_low_rank_injection_to_model(merged_model, base_model, other_models: List, 
                                    rank: int = 4, strength: float = 1.0, 
                                    param_names: Optional[List[str]] = None):
    """
    Apply low-rank residual injection to all specified parameters in a model.
    
    Args:
        merged_model: Model to modify
        base_model: Base reference model
        other_models: List of other models to inject from
        rank: SVD rank to keep
        strength: Injection strength (divided among models)
        param_names: Parameters to process (all if None)
    """
    diagnostic_logger.info(f"Applying low-rank residual injection (rank={rank}, strength={strength})")
    
    merged_state = merged_model.state_dict() if hasattr(merged_model, 'state_dict') else merged_model
    base_state = base_model.state_dict() if hasattr(base_model, 'state_dict') else base_model
    
    if param_names is None:
        param_names = list(merged_state.keys())
    
    per_model_strength = strength / len(other_models)
    injection_stats = {'processed_params': 0, 'total_injection_norm': 0.0}
    
    with torch.no_grad():
        for name in param_names:
            if name not in merged_state or name not in base_state:
                continue
                
            base_param = base_state[name]
            if not isinstance(base_param, torch.Tensor):
                continue
            
            # Accumulate low-rank injections from all other models
            accumulated_injection = torch.zeros_like(base_param)
            
            for other_model in other_models:
                other_state = other_model.state_dict() if hasattr(other_model, 'state_dict') else other_model
                
                if name not in other_state:
                    continue
                    
                other_param = other_state[name]
                if not isinstance(other_param, torch.Tensor) or other_param.shape != base_param.shape:
                    continue
                
                # Compute low-rank injection for this model
                injection = low_rank_residual_injection(base_param, other_param, rank, per_model_strength)
                accumulated_injection += (injection - base_param)
            
            # Apply accumulated injection
            if torch.norm(accumulated_injection) > 1e-8:
                merged_state[name] = base_param + accumulated_injection
                injection_stats['processed_params'] += 1
                injection_stats['total_injection_norm'] += torch.norm(accumulated_injection).item()
    
    diagnostic_logger.info(f"Low-rank injection complete: {injection_stats['processed_params']} parameters processed")
    return injection_stats

def mini_finetune_calibration(model, calibration_data, steps: int = 3, lr: float = 1e-5, 
                            loss_fn: Optional[Callable] = None):
    """
    Mini-finetune on calibration set to restore subtle contrasts.
    
    Runs 1-5 steps of gradient descent with tiny learning rate on representative
    samples using reconstruction loss.
    
    Args:
        model: Model to finetune
        calibration_data: Small calibration dataset
        steps: Number of gradient steps
        lr: Learning rate
        loss_fn: Loss function (defaults to MSE reconstruction)
    """
    diagnostic_logger.info(f"Mini-finetuning on calibration data: {steps} steps, lr={lr}")
    
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    
    # Set up optimizer for just a few steps
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    
    model.train()
    total_loss = 0.0
    
    for step in range(steps):
        step_loss = 0.0
        batch_count = 0
        
        for batch in calibration_data:
            if isinstance(batch, (list, tuple)):
                x, target = batch[0], batch[1] if len(batch) > 1 else batch[0]
            else:
                x = target = batch
            
            if not isinstance(x, torch.Tensor):
                continue
            
            optimizer.zero_grad()
            
            try:
                # Forward pass
                output = model(x)
                
                # Reconstruction loss (or use provided target)
                if hasattr(output, 'sample'):  # Handle VAE-like outputs
                    output = output.sample
                
                loss = loss_fn(output, target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                step_loss += loss.item()
                batch_count += 1
                
                # Only use a few batches per step
                if batch_count >= 4:
                    break
                    
            except Exception as e:
                diagnostic_logger.warning(f"Mini-finetune step failed: {e}")
                continue
        
        if batch_count > 0:
            avg_step_loss = step_loss / batch_count
            total_loss += avg_step_loss
            diagnostic_logger.debug(f"Mini-finetune step {step+1}/{steps}: loss={avg_step_loss:.6f}")
    
    model.eval()
    avg_total_loss = total_loss / steps if steps > 0 else 0.0
    diagnostic_logger.info(f"Mini-finetuning complete: avg loss={avg_total_loss:.6f}")
    
    return {'avg_loss': avg_total_loss, 'steps_completed': steps}

def activation_space_sharpening(model, sharpening_lambda: float = 0.1, target_layers: Optional[List[str]] = None):
    """
    Apply sharpening in feature/activation space using unsharp mask.
    
    Treats middle activations as "images" and applies unsharp mask filter:
    act = act + λ * (act - avg_pool(act))
    
    Args:
        model: Model to modify
        sharpening_lambda: Sharpening strength (typically 0.05-0.2)
        target_layers: Layer names to apply sharpening to (auto-detect if None)
    """
    diagnostic_logger.info(f"Setting up activation-space sharpening (λ={sharpening_lambda})")
    
    if target_layers is None:
        # Auto-detect good layers for sharpening (middle conv layers)
        target_layers = []
        layer_names = [name for name, _ in model.named_modules()]
        conv_layers = [name for name in layer_names if any(conv_type in name.lower() 
                      for conv_type in ['conv', 'convolution'])]
        
        # Use middle layers
        if len(conv_layers) > 4:
            start_idx = len(conv_layers) // 3
            end_idx = 2 * len(conv_layers) // 3
            target_layers = conv_layers[start_idx:end_idx]
    
    # Register hooks for sharpening
    sharpening_hooks = []
    
    def make_sharpening_hook(layer_name: str, lambda_val: float):
        def sharpening_hook(module, input, output):
            if not model.training:  # Only apply during inference
                return output
                
            # Apply unsharp mask in activation space
            if output.ndim >= 3:  # Has spatial dimensions
                # Compute "blurred" version using average pooling
                kernel_size = 3
                padding = kernel_size // 2
                
                if output.ndim == 4:  # 2D conv
                    blurred = F.avg_pool2d(output, kernel_size=kernel_size, stride=1, padding=padding)
                elif output.ndim == 5:  # 3D conv
                    blurred = F.avg_pool3d(output, kernel_size=kernel_size, stride=1, padding=padding)
                else:  # 1D conv
                    blurred = F.avg_pool1d(output, kernel_size=kernel_size, stride=1, padding=padding)
                
                # Unsharp mask: original + λ * (original - blurred)
                laplacian = output - blurred
                sharpened = output + lambda_val * laplacian
                
                diagnostic_logger.debug(f"Applied activation sharpening to {layer_name}")
                return sharpened
            
            return output
        
        return sharpening_hook
    
    # Register hooks on target layers
    for layer_name in target_layers:
        try:
            layer = dict(model.named_modules())[layer_name]
            hook = layer.register_forward_hook(make_sharpening_hook(layer_name, sharpening_lambda))
            sharpening_hooks.append((layer_name, hook))
            diagnostic_logger.debug(f"Registered sharpening hook on {layer_name}")
        except KeyError:
            diagnostic_logger.warning(f"Layer {layer_name} not found for sharpening")
    
    diagnostic_logger.info(f"Activation sharpening enabled on {len(sharpening_hooks)} layers")
    
    # Return hook handles for cleanup if needed
    return sharpening_hooks

def dynamic_skip_threshold_scheduling(compatibility_scores: Dict[str, torch.Tensor], 
                                    base_threshold: float = 1e-4) -> Dict[str, float]:
    """
    Dynamic skip threshold scheduling based on coefficient of variation.
    
    Blocks with tight score distributions get lower thresholds (preserve variation),
    blocks with wide spreads get higher thresholds (more aggressive skipping).
    
    Args:
        compatibility_scores: Dict mapping block names to compatibility score tensors
        base_threshold: Base threshold value
        
    Returns:
        Dict mapping block names to adaptive thresholds
    """
    diagnostic_logger.info("Computing dynamic skip thresholds based on score distributions")
    
    adaptive_thresholds = {}
    
    for block_name, scores in compatibility_scores.items():
        if not isinstance(scores, torch.Tensor) or scores.numel() == 0:
            adaptive_thresholds[block_name] = base_threshold
            continue
        
        # Compute coefficient of variation (CV = std / mean)
        mean_score = torch.mean(scores)
        std_score = torch.std(scores)
        
        if mean_score > 1e-8:
            cv = std_score / mean_score
        else:
            cv = 1.0  # Default to high variation
        
        # Adaptive threshold based on CV
        # Low CV (tight distribution) -> lower threshold (preserve more)
        # High CV (wide distribution) -> higher threshold (skip more)
        cv_factor = torch.clamp(cv, 0.1, 2.0)  # Reasonable bounds
        
        adaptive_threshold = base_threshold * cv_factor
        adaptive_thresholds[block_name] = adaptive_threshold.item()
        
        diagnostic_logger.debug(f"Block {block_name}: CV={cv:.3f}, threshold={adaptive_threshold:.2e}")
    
    diagnostic_logger.info(f"Dynamic thresholds computed for {len(adaptive_thresholds)} blocks")
    return adaptive_thresholds

class PostMergeRefinementPipeline:
    """
    Orchestrates the complete post-merge refinement pipeline.
    
    Pipeline: Merge → Frobenius Rescaling → Low-Rank Injection → 
              BN/LayerNorm Recalibration → Mini-Finetune → Activation Sharpening
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.stats = {}
        
    def apply_full_pipeline(self, merged_model, base_model, other_models: List,
                          calibration_data=None, monitor_stats: bool = True):
        """
        Apply the complete refinement pipeline.
        
        Args:
            merged_model: The merged model to refine
            base_model: Original base model for reference
            other_models: List of other models that were merged
            calibration_data: Data for calibration steps
            monitor_stats: Whether to monitor layer statistics
            
        Returns:
            Dictionary of refinement statistics
        """
        pipeline_stats = {'steps_completed': [], 'step_stats': {}}
        
        diagnostic_logger.info("🚀 Starting post-merge refinement pipeline")
        
        with MemoryEfficientContext("post_merge_pipeline"):
            # Step 1: Frobenius norm rescaling
            if self.config.get('frobenius_rescaling', True):
                diagnostic_logger.info("📐 Step 1: Frobenius norm rescaling")
                fro_stats = frobenius_rescale_layers(merged_model, base_model)
                pipeline_stats['steps_completed'].append('frobenius_rescaling')
                pipeline_stats['step_stats']['frobenius_rescaling'] = fro_stats
                
                if monitor_stats:
                    monitor_memory_usage("POST-FROBENIUS")
            
            # Step 2: Low-rank residual injection
            if self.config.get('low_rank_injection', True) and other_models:
                diagnostic_logger.info("🔍 Step 2: Low-rank residual injection")
                rank = self.config.get('svd_rank', 4)
                strength = self.config.get('injection_strength', 0.3)
                
                lr_stats = apply_low_rank_injection_to_model(
                    merged_model, base_model, other_models, rank=rank, strength=strength
                )
                pipeline_stats['steps_completed'].append('low_rank_injection')
                pipeline_stats['step_stats']['low_rank_injection'] = lr_stats
                
                if monitor_stats:
                    monitor_memory_usage("POST-LOW-RANK")
            
            # Step 3: Normalization recalibration (if available)
            if self.config.get('norm_recalibration', True) and calibration_data is not None:
                diagnostic_logger.info("⚖️ Step 3: Normalization recalibration")
                try:
                    from .norm_recalibration import recalibrate_norm_stats
                    recalibrate_norm_stats(merged_model, calibration_data)
                    pipeline_stats['steps_completed'].append('norm_recalibration')
                except Exception as e:
                    diagnostic_logger.warning(f"Norm recalibration failed: {e}")
                
                if monitor_stats:
                    monitor_memory_usage("POST-NORM-RECALIB")
            
            # Step 4: Mini-finetune (if calibration data available)
            if self.config.get('mini_finetune', False) and calibration_data is not None:
                diagnostic_logger.info("🎯 Step 4: Mini-finetuning")
                steps = self.config.get('finetune_steps', 3)
                lr = self.config.get('finetune_lr', 1e-5)
                
                try:
                    ft_stats = mini_finetune_calibration(merged_model, calibration_data, steps=steps, lr=lr)
                    pipeline_stats['steps_completed'].append('mini_finetune')
                    pipeline_stats['step_stats']['mini_finetune'] = ft_stats
                except Exception as e:
                    diagnostic_logger.warning(f"Mini-finetune failed: {e}")
                
                if monitor_stats:
                    monitor_memory_usage("POST-MINI-FINETUNE")
            
            # Step 5: Activation sharpening
            if self.config.get('activation_sharpening', False):
                diagnostic_logger.info("✨ Step 5: Activation space sharpening")
                lambda_val = self.config.get('sharpening_lambda', 0.1)
                
                try:
                    sharpening_hooks = activation_space_sharpening(merged_model, lambda_val)
                    pipeline_stats['steps_completed'].append('activation_sharpening')
                    pipeline_stats['step_stats']['activation_sharpening'] = {
                        'hooks_registered': len(sharpening_hooks),
                        'lambda': lambda_val
                    }
                except Exception as e:
                    diagnostic_logger.warning(f"Activation sharpening failed: {e}")
        
        total_steps = len(pipeline_stats['steps_completed'])
        diagnostic_logger.info(f"✅ Post-merge refinement pipeline complete: {total_steps} steps applied")
        
        return pipeline_stats

def create_default_refinement_config() -> Dict:
    """Create a default configuration for post-merge refinement."""
    return {
        'frobenius_rescaling': True,
        'low_rank_injection': True,
        'svd_rank': 4,
        'injection_strength': 0.3,
        'norm_recalibration': True,
        'mini_finetune': False,  # Disabled by default (requires calibration data)
        'finetune_steps': 3,
        'finetune_lr': 1e-5,
        'activation_sharpening': False,  # Disabled by default (experimental)
        'sharpening_lambda': 0.1
    }