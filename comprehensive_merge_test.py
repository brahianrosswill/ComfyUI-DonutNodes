#!/usr/bin/env python3
"""
Comprehensive test to identify all factors affecting merge_strength=0.01 behavior.
"""

import torch
import numpy as np

def test_with_dynamic_strength():
    """Test with sensitivity > 0 (dynamic strength enabled)"""
    print("=== DYNAMIC STRENGTH ENABLED (sensitivity=5.0) ===")
    
    merge_strength = 0.01
    min_strength = 0.0
    max_strength = 1.0
    sensitivity = 5.0
    
    # Test different compatibility scores
    compat_scores = [0.001, 0.01, 0.1, 0.5, 1.0]
    
    for compat in compat_scores:
        # Manual calculation
        compat_clamped = torch.clamp(torch.tensor(compat), -10.0, 10.0)
        sigmoid_factor = torch.sigmoid(sensitivity * compat_clamped)
        strength_range = max_strength - min_strength
        scaled_strength = min_strength + strength_range * sigmoid_factor
        final_strength = scaled_strength * merge_strength
        
        print(f"  compat={compat:.3f} → sigmoid={sigmoid_factor:.6f} → final={final_strength:.6f}")
    
    print(f"  Range: {merge_strength * min_strength:.6f} to {merge_strength * max_strength:.6f}")

def test_with_static_strength():
    """Test with sensitivity = 0 (static strength)"""
    print("\n=== STATIC STRENGTH (sensitivity=0.0) ===")
    
    merge_strength = 0.01
    min_strength = 0.0
    max_strength = 1.0
    sensitivity = 0.0
    
    # When sensitivity=0, uses midpoint
    midpoint = (min_strength + max_strength) / 2.0
    final_strength = merge_strength * midpoint
    
    print(f"  midpoint = ({min_strength} + {max_strength}) / 2 = {midpoint}")
    print(f"  final_strength = {merge_strength} * {midpoint} = {final_strength}")

def test_normalization_modes():
    """Test how normalization modes affect the final result"""
    print("\n=== NORMALIZATION MODE EFFECTS ===")
    
    # Simulate parameter vectors
    base_param = torch.tensor([1.0, 2.0, 3.0, 4.0])
    other_param = torch.tensor([1.1, 2.2, 3.3, 4.4])  # 10% difference
    
    merge_strength = 0.01
    
    # Test different normalization approaches
    modes = ["none", "magnitude", "direction"]
    
    for mode in modes:
        print(f"\n  Mode: {mode}")
        
        if mode == "none":
            # Simple linear interpolation
            delta = other_param - base_param
            merged = base_param + merge_strength * delta
            
        elif mode == "magnitude":
            # Magnitude-based normalization
            base_magnitude = torch.norm(base_param)
            other_magnitude = torch.norm(other_param)
            
            # Direction from base
            base_direction = base_param / (base_magnitude + 1e-8)
            other_direction = other_param / (other_magnitude + 1e-8)
            
            # Interpolate direction and magnitude separately
            merged_direction = base_direction + merge_strength * (other_direction - base_direction)
            merged_magnitude = base_magnitude + merge_strength * (other_magnitude - base_magnitude)
            
            merged = merged_direction * merged_magnitude
            
        elif mode == "direction":
            # Direction-only interpolation (preserve base magnitude)
            base_magnitude = torch.norm(base_param)
            
            base_direction = base_param / (base_magnitude + 1e-8)
            other_direction = other_param / (torch.norm(other_param) + 1e-8)
            
            merged_direction = base_direction + merge_strength * (other_direction - base_direction)
            merged = merged_direction * base_magnitude
        
        change_percent = torch.norm(merged - base_param) / torch.norm(base_param) * 100
        print(f"    Base:   {base_param.tolist()}")
        print(f"    Other:  {other_param.tolist()}")
        print(f"    Merged: {merged.tolist()}")
        print(f"    Change: {change_percent:.3f}%")

def test_importance_threshold_effects():
    """Test how importance_threshold affects parameter inclusion"""
    print("\n=== IMPORTANCE THRESHOLD EFFECTS ===")
    
    # Simulate different parameter importance scores
    import math
    
    merge_strength = 0.01
    importance_threshold = 1.0  # Default
    importance_boost = 2.0     # Default
    
    # Different parameter scenarios
    scenarios = [
        ("Low importance", 0.5),
        ("Threshold importance", 1.0),
        ("High importance", 2.0),
        ("Very high importance", 5.0),
    ]
    
    for name, importance in scenarios:
        # Simulate the importance calculation effect
        if importance >= importance_threshold:
            boosted_strength = merge_strength * importance_boost
            included = True
        else:
            boosted_strength = merge_strength
            included = importance >= importance_threshold
        
        print(f"  {name} (score={importance}):")
        print(f"    Included: {included}")
        print(f"    Effective strength: {boosted_strength:.6f}")

def test_skip_threshold_effects():
    """Test how skip_threshold affects parameter processing"""
    print("\n=== SKIP THRESHOLD EFFECTS ===")
    
    # Simulate compatibility scores and skip thresholds
    skip_threshold = 1e-4  # Default percentile threshold
    
    compat_scores = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    
    for compat in compat_scores:
        skipped = compat < skip_threshold
        print(f"  Compatibility {compat:.1e}: {'SKIPPED' if skipped else 'PROCESSED'}")

def test_actual_merge_formula():
    """Test the complete merge formula"""
    print("\n=== COMPLETE MERGE FORMULA TEST ===")
    
    # Simulate the full merge process
    base_param = torch.tensor([1.0, 2.0, 3.0])
    other_param = torch.tensor([1.05, 2.1, 3.15])  # 5% increase
    
    merge_strength = 0.01
    min_strength = 0.0
    max_strength = 1.0
    sensitivity = 0.0  # Static strength
    compatibility_score = 0.01
    importance_threshold = 1.0
    importance_score = 2.0  # Above threshold
    importance_boost = 2.0
    
    print(f"Base parameter: {base_param.tolist()}")
    print(f"Other parameter: {other_param.tolist()}")
    print(f"Expected 5% difference per element")
    
    # Step 1: Check if parameter passes importance threshold
    if importance_score >= importance_threshold:
        effective_merge_strength = merge_strength * importance_boost
        print(f"Importance boost applied: {merge_strength} * {importance_boost} = {effective_merge_strength}")
    else:
        effective_merge_strength = merge_strength
        print(f"No importance boost: {effective_merge_strength}")
    
    # Step 2: Calculate dynamic strength (static mode)
    if sensitivity == 0.0:
        midpoint = (min_strength + max_strength) / 2.0
        final_strength = effective_merge_strength * midpoint
        print(f"Static strength: {effective_merge_strength} * {midpoint} = {final_strength}")
    
    # Step 3: Apply the merge
    delta = other_param - base_param
    merged_param = base_param + final_strength * delta
    
    actual_change = torch.norm(merged_param - base_param) / torch.norm(base_param) * 100
    
    print(f"Delta: {delta.tolist()}")
    print(f"Final strength applied: {final_strength:.6f}")
    print(f"Merged result: {merged_param.tolist()}")
    print(f"Actual change: {actual_change:.6f}%")
    print(f"Expected with 0.01 strength: ~0.1% (0.01 * 5% * 2 boost * 0.5 midpoint)")

if __name__ == "__main__":
    print("COMPREHENSIVE MERGE STRENGTH ANALYSIS")
    print("=" * 60)
    
    test_with_dynamic_strength()
    test_with_static_strength()
    test_normalization_modes()
    test_importance_threshold_effects()
    test_skip_threshold_effects()
    test_actual_merge_formula()
    
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY:")
    print("=" * 60)
    print("Factors that can amplify merge_strength=0.01:")
    print("1. importance_boost=2.0 → doubles effective strength to 0.02")
    print("2. Static mode midpoint with max_strength=1.0 → 0.5x multiplier")
    print("3. Combined: 0.01 * 2.0 * 0.5 = 0.01 (1% effective change)")
    print("4. Normalization modes can alter the final parameter change")
    print("\nFor truly minimal changes:")
    print("- Use min_strength=0.0, max_strength=0.2 (reduces midpoint to 0.1)")
    print("- Consider importance_boost=1.0 (no boost)")
    print("- Use sensitivity=5.0 with low compatibility (enables fine control)")