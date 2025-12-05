"""
Merge strength calculation functions for DonutWidenMerge.

This module contains the core functions for computing merge strength values
based on compatibility scores and various parameters. These functions were
extracted from DonutWidenMerge.py to promote code reuse and modularity.

Functions:
- get_adaptive_skip_threshold: Get adaptive skip threshold based on parameter name patterns
- MergeFailureTracker: Track and report merge failures with detailed diagnostics  
- _compatibility_to_merge_strength: Convert compatibility score to merge strength
- _fast_sigmoid_strength: JIT-compiled version for batch sigmoid strength computation
"""

import torch
import numpy as np
import warnings
from .constants import DEFAULT_SKIP_THRESHOLDS


def get_adaptive_skip_threshold(param_name: str, default_threshold: float = 0.0) -> float:
    """
    Get adaptive skip threshold based on parameter name patterns.
    
    Args:
        param_name: Name of the parameter
        default_threshold: Fallback threshold
    """
    param_lower = param_name.lower()
    
    # Check each pattern and return first match
    for pattern, threshold in DEFAULT_SKIP_THRESHOLDS.items():
        if pattern in param_lower:
            return threshold
            
    return default_threshold


class MergeFailureTracker:
    """Track and report merge failures with detailed diagnostics"""
    
    def __init__(self):
        self.failed_params = []
        self.failure_reasons = {}
        self.fallback_count = 0
        
    def record_failure(self, param_name: str, exception: Exception, fallback_used: bool = True):
        """Record a parameter merge failure"""
        self.failed_params.append(param_name)
        self.failure_reasons[param_name] = str(exception)
        if fallback_used:
            self.fallback_count += 1
            
    def check_critical_failures(self, raise_on_critical: bool = True):
        """Check if failures are critical enough to abort"""
        if not self.failed_params:
            return
            
        critical_patterns = ['attention', 'embed', 'head', 'output']
        critical_failures = [p for p in self.failed_params 
                           if any(pattern in p.lower() for pattern in critical_patterns)]
        
        failure_rate = len(self.failed_params) / max(1, len(self.failed_params) + self.fallback_count)
        
        if critical_failures and raise_on_critical:
            raise RuntimeError(f"Critical merge failures in {len(critical_failures)} parameters: {critical_failures[:3]}...")
            
        if failure_rate > 0.1:  # More than 10% failure rate
            warnings.warn(f"High merge failure rate: {failure_rate:.1%} ({len(self.failed_params)} failed)")
            
    def get_summary(self) -> str:
        """Get a summary of merge failures"""
        if not self.failed_params:
            return "✓ All parameters merged successfully"
            
        return f"⚠ {len(self.failed_params)} merge failures, {self.fallback_count} fallbacks used"


def _compatibility_to_merge_strength(compatibility_score, merge_strength, min_strength, max_strength, sensitivity):
    """
    Convert compatibility score to merge strength using merge_strength as base multiplier.
    
    CRITICAL BEHAVIOR NOTES FOR FUTURE DEBUGGING:
    
    1. MERGE STRENGTH SCALING:
       final_strength = merge_strength * (min_strength + range * sigmoid_factor)
       
       This means with default min_strength=0.5, max_strength=1.5:
       - merge_strength=0.01 → actual range: 0.005 to 0.015 (0.5% to 1.5% change)
       - merge_strength=1.0  → actual range: 0.5 to 1.5 (50% to 150% change)
    
    2. FOR NEAR-BASE RESULTS:
       To get merge_strength=0.01 behaving like 1% max change:
       - Set min_strength=0.0, max_strength=1.0
       - This gives actual range: 0.0 to 0.01 (0% to 1% change)
    
    3. WHY DEFAULT VALUES ARE HIGH:
       min_strength=0.5 is designed for normal merging (merge_strength ≥ 0.5)
       where you want meaningful parameter changes, not micro-adjustments.
       
    4. IMPORTANCE_BOOST AMPLIFICATION:
       ComfyUI default importance_boost=2.5 further amplifies the effect!
       merge_strength=0.01 × importance_boost=2.5 → 0.025 (2.5% effective change)
       
    This is correct behavior by design, but users need to adjust ALL parameters
    (min/max_strength AND importance_boost) for very low merge_strength values.
    """
    
    # Special case: merge_strength=0 should return 0 (base model unchanged)
    if merge_strength == 0.0:
        return 0.0
    
    # If sensitivity is 0, disable dynamic strength (use midpoint between min/max multiplied by merge_strength)
    if sensitivity == 0.0:
        return float(merge_strength * (min_strength + max_strength) / 2.0)
    
    # Normalize compatibility scores to a reasonable range for sigmoid
    # Most compatibility scores are very small (0.0001-0.01), so we need to scale them
    # Use the compatibility score directly without centering on 0.5, which is too high
    
    # Apply sigmoid transformation directly to raw compatibility score (preserve all variation)
    # This matches the working bd89e6c version exactly
    sigmoid_input = (compatibility_score - 0.5) * sensitivity
    
    # Apply sigmoid with numerical stability
    if sigmoid_input > 20:
        sigmoid_output = 1.0
    elif sigmoid_input < -20:
        sigmoid_output = 0.0
    else:
        sigmoid_output = 1.0 / (1.0 + np.exp(-sigmoid_input))
    
    # Clamp sigmoid output to valid range (0,1) instead of input
    sigmoid_output = max(0.0, min(sigmoid_output, 1.0))
    
    # Map to strength range using min/max as multipliers
    strength_multiplier = min_strength + (max_strength - min_strength) * sigmoid_output
    
    # Apply merge_strength as base multiplier
    final_strength = merge_strength * strength_multiplier
    
    return float(final_strength)


# Non-JIT version for tensor operations
def _fast_sigmoid_strength(compatibility_tensor: torch.Tensor, merge_strength: float, 
                          min_strength: float, max_strength: float, sensitivity: float) -> torch.Tensor:
    """
    Non-JIT version for batch sigmoid strength computation.
    
    IMPORTANT: This must behave identically to _compatibility_to_merge_strength!
    See that function for detailed behavior documentation.
    """
    if sensitivity == 0.0:
        midpoint = (min_strength + max_strength) / 2.0
        return torch.full_like(compatibility_tensor, merge_strength * midpoint)
    
    # Vectorized sigmoid computation - matches bd89e6c working version exactly
    # Apply sigmoid transformation directly to raw compatibility score (preserve all variation)
    sigmoid_input = (compatibility_tensor - 0.5) * sensitivity
    sigmoid_output = torch.sigmoid(sigmoid_input)  # More numerically stable than manual exp
    
    # Clamp sigmoid output to valid range (0,1) - vectorized clamping
    sigmoid_output = torch.clamp(sigmoid_output, 0.0, 1.0)
    
    # Map to strength range using min/max as multipliers (matches bd89e6c exactly)
    strength_multiplier = min_strength + (max_strength - min_strength) * sigmoid_output
    
    # Apply merge_strength as base multiplier
    final_strength = merge_strength * strength_multiplier
    
    return final_strength