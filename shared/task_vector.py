"""
TaskVector Class for WIDEN Merge

This module contains the TaskVector class that extracts task vectors (deltas) between two models
with SDXL awareness and proper memory management.

The TaskVector class computes parameter differences between a base model and finetuned model,
storing metadata about SDXL layer types and parameter magnitudes for use in WIDEN merging.
"""

import torch
import gc
import re

# Import shared modules with fallback
try:
    from .logging_config import diagnostic_logger
except ImportError:
    try:
        from shared.logging_config import diagnostic_logger
    except ImportError:
        import logging
        diagnostic_logger = logging.getLogger(__name__)


class TaskVector:
    """Extract task vector (delta) between two models with SDXL awareness"""
    
    def __init__(self, base_model, finetuned_model, exclude_param_names_regex=None):
        self.task_vector_param_dict = {}
        self.param_metadata = {}  # Store SDXL-specific metadata

        # FIXED: Proper memory management with explicit cleanup
        try:
            base_params = {n: p.detach().cpu().float()  # FIXED: Removed redundant .clone()
                          for n, p in base_model.named_parameters()}
            finetuned_params = {n: p.detach().cpu().float()  # FIXED: Removed redundant .clone()
                               for n, p in finetuned_model.named_parameters()}

            # Extract deltas with SDXL layer classification
            for name in base_params:
                if name in finetuned_params:
                    if exclude_param_names_regex:
                        # Compile regex patterns once for performance
                        if not hasattr(exclude_param_names_regex, '_compiled'):
                            exclude_param_names_regex._compiled = [
                                re.compile(pattern) if isinstance(pattern, str) else pattern 
                                for pattern in exclude_param_names_regex
                            ]
                        skip = any(pattern.search(name) for pattern in exclude_param_names_regex._compiled)
                        if skip:
                            continue

                    delta = finetuned_params[name] - base_params[name]
                    self.task_vector_param_dict[name] = delta

                    # Classify SDXL layer type and store metadata
                    base_magnitude = torch.norm(base_params[name]).item()
                    delta_magnitude = torch.norm(delta).item()
                    self.param_metadata[name] = {
                        'layer_type': self._classify_sdxl_layer(name),
                        'base_magnitude': base_magnitude,
                        'delta_magnitude': delta_magnitude,
                        'change_ratio': delta_magnitude / (base_magnitude + 1e-8)  # FIXED: Avoid division by zero
                    }

        finally:
            # FIXED: Explicit cleanup to prevent memory leaks
            if 'base_params' in locals():
                del base_params
            if 'finetuned_params' in locals():
                del finetuned_params
            gc.collect()

    def _classify_sdxl_layer(self, param_name):
        """Classify SDXL layer types for specialized handling - ENHANCED"""
        name_lower = param_name.lower()

        # UNet structure classification - more comprehensive
        if 'time_embed' in name_lower:
            return 'time_embedding'
        elif 'label_emb' in name_lower:
            return 'class_embedding'
        elif any(x in name_lower for x in ['attn', 'attention']):
            if 'cross' in name_lower:
                return 'cross_attention'  # Text conditioning
            else:
                return 'self_attention'   # Spatial attention
        elif any(x in name_lower for x in ['conv', 'convolution']):
            if 'in_layers' in name_lower or 'input' in name_lower:
                return 'input_conv'
            elif 'out_layers' in name_lower or 'output' in name_lower:
                return 'output_conv'
            elif 'skip' in name_lower or 'residual' in name_lower:
                return 'skip_conv'
            else:
                return 'feature_conv'
        elif any(x in name_lower for x in ['norm', 'group_norm', 'layer_norm']):
            return 'normalization'
        elif 'bias' in name_lower:
            return 'bias'
        elif any(x in name_lower for x in ['down', 'upsample']):
            return 'resolution_change'
        # Enhanced classification for previously 'other' parameters
        elif any(x in name_lower for x in ['proj', 'projection']):
            return 'self_attention'  # Projections are usually part of attention
        elif any(x in name_lower for x in ['to_q', 'to_k', 'to_v', 'to_out']):
            return 'self_attention'  # Attention components
        elif any(x in name_lower for x in ['ff', 'feedforward', 'mlp']):
            return 'self_attention'  # Feed-forward networks in transformers
        elif 'weight' in name_lower and 'emb' in name_lower:
            return 'time_embedding'  # Embedding weights
        else:
            return 'other'