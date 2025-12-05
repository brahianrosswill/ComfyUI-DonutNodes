"""
Normalization layer recalibration utilities for WIDEN merge operations
"""

import torch
try:
    from .logging_config import diagnostic_logger
except ImportError:
    from logging_config import diagnostic_logger

@torch.no_grad()
def recalibrate_norm_stats(model, calib_data=None, device="cuda", num_batches=5):
    """
    Re-calibrate BatchNorm/LayerNorm running statistics after model merging.
    This helps fix broken normalization stats that can cause merged model instability.
    
    Args:
        model: The merged model to recalibrate
        calib_data: Calibration data loader or tensor. If None, uses random data.
        device: Device to run calibration on
        num_batches: Number of calibration batches to use
    """
    if not hasattr(model, 'modules'):
        return
        
    # Find all BatchNorm-like layers
    bn_layers = []
    for module in model.modules():
        if hasattr(module, 'running_mean') and hasattr(module, 'running_var'):
            bn_layers.append(module)
    
    if not bn_layers:
        return
        
    diagnostic_logger.info(f"Recalibrating {len(bn_layers)} normalization layers")
    
    # Store original training state
    was_training = model.training
    model.train()
    
    # Reset running statistics and set high momentum for quick adaptation
    for layer in bn_layers:
        if hasattr(layer, 'reset_running_stats'):
            layer.reset_running_stats()
        if hasattr(layer, 'momentum'):
            layer.momentum = 1.0  # Full replacement each batch
    
    try:
        if calib_data is None:
            # Generate random calibration data based on model's expected input
            # This is a fallback - real calibration data is always better
            input_shape = getattr(model, 'input_shape', None)
            if input_shape is None:
                # Try to infer from first layer
                first_layer = next(model.modules(), None)
                if hasattr(first_layer, 'in_features'):
                    input_shape = (1, first_layer.in_features)
                elif hasattr(first_layer, 'in_channels'):
                    input_shape = (1, first_layer.in_channels, 32, 32)  # Assume image-like
                else:
                    diagnostic_logger.warning("Cannot determine input shape for norm recalibration")
                    return
                    
            for batch_idx in range(num_batches):
                fake_input = torch.randn(input_shape, device=device)
                try:
                    model(fake_input)
                except:
                    break  # Stop if model can't handle fake input
        else:
            # Use provided calibration data
            batch_count = 0
            for batch in calib_data:
                if batch_count >= num_batches:
                    break
                    
                if isinstance(batch, (list, tuple)):
                    x = batch[0]  # Assume first element is input
                else:
                    x = batch
                    
                if not isinstance(x, torch.Tensor):
                    continue
                    
                try:
                    model(x.to(device))
                    batch_count += 1
                except:
                    continue  # Skip problematic batches
                    
    except Exception as e:
        diagnostic_logger.warning(f"Norm recalibration failed: {e}")
    finally:
        # Restore original training state
        model.train(was_training)

def calibrate_renormalize(merged_param, base_param, mode="calibrate", t=1.0, s=1.5):
    """
    Calibrate and renormalize merged parameters relative to base parameters.
    Helps maintain numerical stability and parameter scale consistency.
    
    Args:
        merged_param: The merged parameter tensor
        base_param: The original base parameter tensor
        mode: "calibrate" (rescale to match base norm), "renormalize" (unit norm), or "adaptive"
        t: Temperature parameter for soft normalization (only for adaptive mode)
        s: Scale factor for calibration
    
    Returns:
        Calibrated/renormalized parameter tensor
    """
    if merged_param.shape != base_param.shape:
        return merged_param  # Can't calibrate if shapes don't match
    
    base_norm = base_param.norm()
    merged_norm = merged_param.norm()
    
    if merged_norm == 0 or base_norm == 0:
        return merged_param  # Avoid division by zero
    
    if mode == "calibrate":
        # Scale merged param to match base param norm, with optional scaling factor
        scale_factor = (base_norm / merged_norm) * s
        return merged_param * scale_factor
        
    elif mode == "renormalize":
        # Normalize to unit norm, then scale by base norm
        unit_normalized = merged_param / merged_norm
        return unit_normalized * base_norm * s
        
    elif mode == "adaptive":
        # Soft normalization based on temperature parameter
        # t=0: no change, t=1: full renormalization
        target_norm = base_norm * s
        current_scale = merged_norm / target_norm
        
        # Soft scaling: blend between current and target scale
        soft_scale = (1 - t) + t * (1 / current_scale)
        return merged_param * soft_scale
        
    else:
        return merged_param  # Unknown mode, return unchanged


def calibrate_renormalize(merged_param, base_param, mode="calibrate", t=1.0, s=1.5):
    """Renormalize using calibration algorithm or simple methods - ENHANCED with conservative options"""
    if mode == "none":
        return merged_param

    elif mode == "magnitude":
        # Simple magnitude preservation (original method)
        base_norm = torch.norm(base_param)
        merged_norm = torch.norm(merged_param)
        if merged_norm > 1e-8:  # Avoid division by zero
            return merged_param * (base_norm / merged_norm)
        return merged_param

    elif mode == "calibrate":
        # More conservative calibration-style renormalization with adjustable parameters
        import torch.nn.functional as F

        # Get the difference from base parameter (the "delta")
        param_delta = merged_param - base_param

        # Only calibrate if there's a significant change
        delta_magnitude = torch.norm(param_delta).item()
        if delta_magnitude < 1e-6:
            return merged_param

        # Work with absolute values for calibration
        param_abs = torch.abs(param_delta)
        param_sign = torch.sign(param_delta)

        # Apply softmax normalization to absolute delta values
        if param_abs.numel() > 1:
            # Flatten for softmax, then reshape back
            original_shape = param_abs.shape
            param_flat = param_abs.flatten()

            if param_flat.sum() > 1e-8:  # Avoid division by zero
                # ENHANCED: Adjustable softmax temperature for more/less smoothing
                # Lower t = more conservative (sharper), higher t = more smoothing
                temperature = max(0.1, min(2.0, t))  # Clamp between 0.1-2.0
                sm_m = F.softmax(param_flat * temperature, dim=0)

                # ENHANCED: Adjustable calibration thresholding
                # Lower t = higher threshold (more selective), higher t = lower threshold
                K = param_flat.numel()
                threshold_factor = max(0.1, min(1.0, t))  # Clamp between 0.1-1.0
                thr_m = (threshold_factor / K) * sm_m.sum() * 0.5

                # ENHANCED: Adjustable scaling intensity
                # Lower s = less aggressive scaling, higher s = more aggressive
                scaling_factor = max(1.0, min(3.0, s))  # Clamp between 1.0-3.0
                conservative_intensity = max(0.1, min(1.0, (scaling_factor - 1.0) * 0.5))  # More conservative
                cal_m = torch.where(sm_m > thr_m, 1.0 + conservative_intensity * sm_m, sm_m)

                # Renormalize to preserve relative magnitudes
                if cal_m.sum() > 1e-8:
                    cal_m = cal_m * (param_flat.sum() / cal_m.sum())

                # Reshape back and restore signs
                cal_m = cal_m.reshape(original_shape)
                calibrated_delta = cal_m * param_sign

                # Apply calibrated delta back to base parameter
                calibrated_param = base_param + calibrated_delta

                return calibrated_param
            else:
                return merged_param
        else:
            return merged_param

    else:
        raise ValueError(f"Unknown renormalization mode: {mode}. Use 'none', 'magnitude', or 'calibrate'")