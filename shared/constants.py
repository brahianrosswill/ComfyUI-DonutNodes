"""
Shared constants and configuration variables for ComfyUI-DonutNodes.

This module contains all the shared constants, feature flags, cache variables,
and configuration values used across the codebase.
"""

# Library availability checks
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Per-layer skip threshold configuration
# Lines 221-229 from DonutWidenMerge.py
DEFAULT_SKIP_THRESHOLDS = {
    "conv": 1e-3,
    "linear": 1e-4, 
    "norm": 0.0,  # Never skip normalization layers
    "attention": 1e-5,
    "embed": 0.0,  # Never skip embedding layers
    "bias": 1e-4,
    "weight": 1e-4
}

# Cache variables and configuration
# Global tensor buffer cache for memory reuse
_TENSOR_BUFFER_CACHE = {}

# Global cache for preventing redundant processing
_MERGE_CACHE = {}
_CACHE_MAX_SIZE = 3  # Reduced from 10 to prevent memory bloat

# Shared pooling kernels to reduce allocations
_POOLING_KERNELS = {}

# Global memory profiler for WIDEN merge operations
_WIDEN_MEMORY_PROFILER = None

# Node class mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "DonutWidenMergeUNet": None,  # Will be populated by actual node classes
    "DonutWidenMergeCLIP": None,
    "DonutFillerClip": None,
    "DonutFillerModel": None,
}