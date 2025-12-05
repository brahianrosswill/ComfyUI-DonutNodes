"""
Tensor alignment utilities for structure-preserving operations.

This module contains functions for aligning tensors with different shapes,
permuting layers for optimal matching, and handling embedding orientations.
"""

import torch
import torch.nn.functional as F
import numpy as np

# Check for scipy availability for Hungarian algorithm
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import logging from the shared module
try:
    from .logging_config import diagnostic_logger
except ImportError:
    # Fallback if running standalone
    import logging
    diagnostic_logger = logging.getLogger(__name__)


def align_linear_layer(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Permute the rows of B to best match A, based on cosine similarity.
    Uses Hungarian algorithm for optimal neuron matching.
    
    Args:
        A, B: (out_features, in_features) weight tensors
    Returns:
        Permuted copy of B that best aligns with A
    """
    if not SCIPY_AVAILABLE or A.shape != B.shape or A.ndim != 2:
        return B
    
    try:
        # Compute normalized rows for cosine similarity
        a_norm = A / (A.norm(dim=1, keepdim=True) + 1e-8)
        b_norm = B / (B.norm(dim=1, keepdim=True) + 1e-8)
        
        # Similarity matrix (out × out)
        sim = torch.mm(a_norm, b_norm.T)  # cosine similarities
        
        # Convert to cost matrix (minimize negative similarity)
        cost = -sim.cpu().numpy()
        
        # Solve assignment problem
        row_idx, col_idx = linear_sum_assignment(cost)
        
        # Permute B according to optimal assignment
        B_permuted = B[col_idx]
        
        diagnostic_logger.debug(f"Aligned linear layer: similarity improved by {sim[row_idx, col_idx].mean().item():.4f}")
        return B_permuted
        
    except Exception as e:
        diagnostic_logger.warning(f"Linear layer alignment failed: {e}")
        return B


def transpose_embeddings_if_needed(param_name: str, param_tensor: torch.Tensor, hidden_size_hint: int = None) -> torch.Tensor:
    """
    Automatically detect and transpose embedding layers if dimensionality suggests it's needed.
    
    Args:
        param_name: Parameter name to check for embedding patterns
        param_tensor: The parameter tensor
        hidden_size_hint: Expected hidden dimension size for validation
    """
    if param_tensor.ndim != 2:
        return param_tensor
        
    # Check if this looks like an embedding layer
    embedding_patterns = ['embed', 'emb', 'token', 'pos', 'position']
    if not any(pattern in param_name.lower() for pattern in embedding_patterns):
        return param_tensor
        
    # Heuristic: if second dim looks like hidden_size, probably correct orientation
    if hidden_size_hint and param_tensor.shape[1] == hidden_size_hint:
        return param_tensor
        
    # If first dimension is much larger than second, might need transpose
    if param_tensor.shape[0] > param_tensor.shape[1] * 2:
        diagnostic_logger.info(f"Auto-transposing potential embedding {param_name}: {param_tensor.shape} -> {param_tensor.shape[::-1]}")
        return param_tensor.T
        
    return param_tensor


def align_and_stack(t1, t2, spatial_dim=None):
    """
    Align two tensors using adaptive pooling instead of flattening.
    Preserves per-feature structure while handling dimension mismatches.
    """
    if t1.shape == t2.shape:
        return torch.stack([t1, t2], dim=0)
    
    # Handle different tensor types with appropriate pooling
    if t1.ndim == 4 and t2.ndim == 4:  # Conv weights (N, C, H, W)
        # Pool spatial dimensions to the smaller size
        target_h = min(t1.size(2), t2.size(2))
        target_w = min(t1.size(3), t2.size(3))
        t1_pooled = F.adaptive_avg_pool2d(t1, (target_h, target_w))
        t2_pooled = F.adaptive_avg_pool2d(t2, (target_h, target_w))
        return torch.stack([t1_pooled, t2_pooled], dim=0)
    
    elif t1.ndim == 2 and t2.ndim == 2:  # Linear weights (in, out)
        # Pool to the smaller dimension in each axis
        target_shape = (min(t1.size(0), t2.size(0)), min(t1.size(1), t2.size(1)))
        t1_pooled = F.adaptive_avg_pool1d(t1.unsqueeze(0), target_shape[1]).squeeze(0)
        t2_pooled = F.adaptive_avg_pool1d(t2.unsqueeze(0), target_shape[1]).squeeze(0)
        
        # Handle first dimension
        if t1_pooled.size(0) > target_shape[0]:
            t1_pooled = t1_pooled[:target_shape[0]]
        if t2_pooled.size(0) > target_shape[0]:
            t2_pooled = t2_pooled[:target_shape[0]]
            
        return torch.stack([t1_pooled, t2_pooled], dim=0)
    
    # Fallback: use broadcasting if possible
    try:
        bcast = torch.broadcast_tensors(t1, t2)
        return torch.stack(bcast, dim=0)
    except RuntimeError:
        # Last resort: pool to scalars but preserve semantic meaning
        return torch.stack([t1.mean(), t2.mean()], dim=0)


def safe_stack(tensors, dim=0):
    """
    Stack tensors safely using broadcast-aware operations.
    """
    if not tensors:
        return None
    
    if len(tensors) == 1:
        return tensors[0].unsqueeze(dim)
    
    # Check if all tensors have the same shape
    shapes = [t.shape for t in tensors]
    if all(s == shapes[0] for s in shapes):
        return torch.stack(tensors, dim)
    
    # Try broadcasting first
    try:
        bcast = torch.broadcast_tensors(*tensors)
        return torch.stack(bcast, dim)
    except RuntimeError:
        # Use adaptive alignment
        return align_tensors(tensors, dim)


def align_tensors(tensors, stack_dim=0):
    """
    Align tensors with different shapes using optimized einsum-style patterns.
    Uses efficient tensor operations and vectorized pooling for better performance.
    """
    if not tensors:
        return None
    
    # Find target shape by taking minimum in each dimension (conservative pooling)
    shapes = [t.shape for t in tensors]
    ndims = [len(s) for s in shapes]
    
    # Handle tensors with different number of dimensions
    if not all(nd == ndims[0] for nd in ndims):
        # Pad smaller tensors with singleton dimensions
        max_ndim = max(ndims)
        padded_tensors = []
        for t in tensors:
            while t.ndim < max_ndim:
                t = t.unsqueeze(-1)
            padded_tensors.append(t)
        tensors = padded_tensors
        shapes = [t.shape for t in tensors]
    
    # Compute target shape (minimum size in each dimension)
    target_shape = tuple(min(s[i] for s in shapes) for i in range(len(shapes[0])))
    
    aligned = []
    for tensor in tensors:
        aligned_tensor = tensor
        
        # Handle 4D tensors (conv weights) with adaptive pooling
        if tensor.ndim == 4 and (tensor.shape[2] > target_shape[2] or tensor.shape[3] > target_shape[3]):
            aligned_tensor = F.adaptive_avg_pool2d(aligned_tensor, target_shape[2:4])
        
        # Handle 2D tensors (linear weights) with optimized einsum-style pooling
        elif tensor.ndim == 2:
            h, w = aligned_tensor.shape
            target_h, target_w = target_shape[0], target_shape[1]
            
            # Optimized pooling using einsum patterns for better memory efficiency
            if h > target_h:
                # Pool along first dimension using vectorized operations
                pool_factor = h // target_h
                remainder = h % target_h
                
                if remainder == 0:
                    # Perfect division - use einsum for efficient reshape + mean
                    # einsum 'hpw->hw' where h=target_h, p=pool_factor, w=width
                    aligned_tensor = torch.einsum('hpw->hw', 
                        aligned_tensor[:target_h * pool_factor].view(target_h, pool_factor, w))
                else:
                    # Handle remainder with efficient concatenation
                    main_part = torch.einsum('hpw->hw',
                        aligned_tensor[:target_h * pool_factor].view(target_h, pool_factor, w))
                    remainder_part = aligned_tensor[target_h * pool_factor:].mean(dim=0, keepdim=True)
                    aligned_tensor = torch.cat([main_part, remainder_part], dim=0)[:target_h]
            
            if aligned_tensor.size(1) > target_w:
                # Pool along second dimension using vectorized operations
                h_curr, w_curr = aligned_tensor.shape
                pool_factor = w_curr // target_w
                remainder = w_curr % target_w
                
                if remainder == 0:
                    # Perfect division - use einsum for efficient reshape + mean
                    # einsum 'hwp->hw' where h=height, w=target_w, p=pool_factor
                    aligned_tensor = torch.einsum('hwp->hw',
                        aligned_tensor[:, :target_w * pool_factor].view(h_curr, target_w, pool_factor))
                else:
                    # Handle remainder with efficient concatenation
                    main_part = torch.einsum('hwp->hw',
                        aligned_tensor[:, :target_w * pool_factor].view(h_curr, target_w, pool_factor))
                    remainder_part = aligned_tensor[:, target_w * pool_factor:].mean(dim=1, keepdim=True)
                    aligned_tensor = torch.cat([main_part, remainder_part], dim=1)[:, :target_w]
        
        # Handle other dimensions with optimized slicing and padding
        else:
            # Efficient slicing using advanced indexing
            if aligned_tensor.shape != target_shape:
                # Create slice indices once and reuse
                slices = tuple(slice(0, min(aligned_tensor.size(i), target_shape[i])) 
                             for i in range(aligned_tensor.ndim))
                aligned_tensor = aligned_tensor[slices]
                
                # Vectorized padding calculation
                current_shape = aligned_tensor.shape
                pad_needed = any(current_shape[i] < target_shape[i] for i in range(len(current_shape)))
                
                if pad_needed:
                    # Compute padding in one pass using vectorized calculation
                    pad_widths = []
                    for i in range(aligned_tensor.ndim - 1, -1, -1):
                        pad_widths.extend([0, max(0, target_shape[i] - current_shape[i])])
                    
                    aligned_tensor = F.pad(aligned_tensor, pad_widths, mode='constant', value=0)
        
        aligned.append(aligned_tensor)
    
    # Use efficient stacking
    return torch.stack(aligned, dim=stack_dim)


def create_condensed_delta(delta: torch.Tensor, param_name: str) -> torch.Tensor:
    """
    Create a condensed version of delta that preserves per-channel structure
    while averaging out spatial dimensions for more informative metadata.
    
    Args:
        delta: The parameter delta tensor
        param_name: Name of the parameter for debugging
    
    Returns:
        Condensed delta preserving channel structure
    """
    if delta.ndim <= 2:
        # Already condensed enough
        return delta
    
    # For conv layers and higher-D tensors: preserve [out_ch, in_ch], average spatial
    condensed = delta.mean(dim=tuple(range(2, delta.ndim)))
    
    diagnostic_logger.debug(f"Condensed {param_name}: {delta.shape} -> {condensed.shape}")
    return condensed


def safe_align_and_stack(t1: torch.Tensor, t2: torch.Tensor, keep_dims: int = 2) -> torch.Tensor:
    """
    Safely align and stack two tensors, preserving important dimensions while
    condensing others using mean instead of aggressive flattening.
    
    Args:
        t1, t2: Tensors to align and stack
        keep_dims: Number of leading dimensions to preserve
    
    Returns:
        Stacked tensor with aligned shapes
    """
    # Collapse dims beyond `keep_dims` so shapes match more naturally
    if t1.dim() > keep_dims:
        t1 = t1.mean(dim=tuple(range(keep_dims, t1.dim())))
    if t2.dim() > keep_dims:
        t2 = t2.mean(dim=tuple(range(keep_dims, t2.dim())))
    
    try:
        # Try broadcasting first
        a, b = torch.broadcast_tensors(t1, t2)
        return torch.stack([a, b], dim=0)
    except Exception as e:
        diagnostic_logger.debug(f"Broadcasting failed, using fallback: {e}")
        # Fallback: ensure same shape by padding or truncating
        if t1.shape != t2.shape:
            # Use the smaller shape
            min_shape = [min(s1, s2) for s1, s2 in zip(t1.shape, t2.shape)]
            slices = tuple(slice(0, s) for s in min_shape)
            t1 = t1[slices]
            t2 = t2[slices]
        return torch.stack([t1, t2], dim=0)