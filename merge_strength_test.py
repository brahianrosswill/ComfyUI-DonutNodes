#!/usr/bin/env python3
"""
Test script to demonstrate merge strength behavior and verify the fix.

This script shows how min_strength and max_strength affect the final merge strength
and explains why merge_strength=0.01 wasn't producing near-base results.
"""

import torch
import numpy as np

def _compatibility_to_merge_strength_OLD(compatibility_score, merge_strength, min_strength, max_strength, sensitivity):
    """OLD (BUGGY) version with incorrect clamping"""
    compat = torch.clamp(compatibility_score, -10.0, 10.0)
    sigmoid_factor = torch.sigmoid(sensitivity * compat)
    strength_range = max_strength - min_strength
    scaled_strength = min_strength + strength_range * sigmoid_factor
    final_strength = scaled_strength * merge_strength
    # BUG: Clamps to original range instead of scaled range
    return torch.clamp(final_strength, min_strength, max_strength)

def _compatibility_to_merge_strength_NEW(compatibility_score, merge_strength, min_strength, max_strength, sensitivity):
    """NEW (FIXED) version with correct clamping"""
    compat = torch.clamp(compatibility_score, -10.0, 10.0)
    sigmoid_factor = torch.sigmoid(sensitivity * compat)
    strength_range = max_strength - min_strength
    scaled_strength = min_strength + strength_range * sigmoid_factor
    final_strength = scaled_strength * merge_strength
    # FIX: Clamps to scaled range
    min_clamped = min_strength * merge_strength
    max_clamped = max_strength * merge_strength
    return torch.clamp(final_strength, min_clamped, max_clamped)

def test_merge_strength_behavior():
    """Test merge strength calculation with different scenarios"""
    
    print("=" * 60)
    print("MERGE STRENGTH BEHAVIOR TEST")
    print("=" * 60)
    
    # Test scenarios
    scenarios = [
        ("Very Low Merge", 0.01, 0.5, 1.5, 0.0),  # sensitivity=0 disables dynamic strength
        ("Low Merge", 0.1, 0.5, 1.5, 0.0),
        ("Medium Merge", 0.5, 0.5, 1.5, 0.0),
        ("Full Merge", 1.0, 0.5, 1.5, 0.0),
        ("Corrected Low", 0.01, 0.0, 1.0, 0.0),  # Better range for low merge
    ]
    
    # Test compatibility scores
    compat_scores = torch.tensor([0.001, 0.01, 0.1, 0.5, 1.0])
    
    for scenario_name, merge_str, min_str, max_str, sensitivity in scenarios:
        print(f"\n{scenario_name}:")
        print(f"  merge_strength={merge_str}, min_strength={min_str}, max_strength={max_str}")
        
        # Test with different compatibility scores
        for compat in compat_scores:
            old_result = _compatibility_to_merge_strength_OLD(compat, merge_str, min_str, max_str, sensitivity)
            new_result = _compatibility_to_merge_strength_NEW(compat, merge_str, min_str, max_str, sensitivity)
            
            print(f"    compat={compat:.3f}: OLD={old_result:.6f}, NEW={new_result:.6f}")
    
    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print("=" * 60)
    print("1. OLD version had clamping bug that prevented very low merge strengths")
    print("2. Default min_strength=0.5 means merge_strength=0.01 still applies 0.5% minimum change")
    print("3. For near-base results with merge_strength=0.01, use min_strength=0.0")
    print("4. The 'Corrected Low' scenario shows proper near-zero behavior")
    
    print("\n" + "=" * 60)
    print("RECOMMENDED FIX:")
    print("=" * 60)
    print("For merge_strength < 0.1:")
    print("- Set min_strength = 0.0")
    print("- Set max_strength = 1.0")
    print("- This gives actual range: 0.0 to merge_strength")
    print("- At merge_strength=0.01: range is 0.0 to 0.01 (1% max change)")

if __name__ == "__main__":
    test_merge_strength_behavior()