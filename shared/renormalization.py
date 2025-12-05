"""
Renormalization functions for DonutWidenMerge

This module contains the calibrate_renormalize function and related utilities
extracted from bd89e6c for parameter magnitude preservation during merging.
"""

import torch
import torch.nn.functional as F


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
                
                # ENHANCED: Adjustable scaling factor for calibration strength
                # Lower s = more conservative (closer to base), higher s = more aggressive
                scaling_factor = max(0.5, min(3.0, s))  # Clamp between 0.5-3.0
                calibrated_abs = sm_m * param_flat.sum() * scaling_factor
                calibrated_abs = calibrated_abs.reshape(original_shape)
            else:
                calibrated_abs = param_abs
        else:
            calibrated_abs = param_abs

        # Reconstruct with signs and add back to base
        calibrated_delta = calibrated_abs * param_sign
        calibrated_param = base_param + calibrated_delta

        return calibrated_param

    else:
        # Unknown mode - return as-is
        return merged_param