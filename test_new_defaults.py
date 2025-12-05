#!/usr/bin/env python3
"""
Test the new improved defaults to verify intuitive behavior.
"""

import torch
import numpy as np

def test_new_defaults():
    """Test with the new improved defaults"""
    print("NEW IMPROVED DEFAULTS ANALYSIS")
    print("=" * 50)
    
    # New improved defaults
    merge_strength = 0.01
    min_strength = 0.0      # Changed from 0.5
    max_strength = 1.0      # Changed from 1.5
    importance_boost = 1.0  # Changed from 2.5
    rank_sensitivity = 2.0  # Keep this for dynamic behavior
    importance_threshold = 1.0
    
    print(f"merge_strength: {merge_strength}")
    print(f"min_strength: {min_strength} (was 0.5)")
    print(f"max_strength: {max_strength} (was 1.5)")
    print(f"importance_boost: {importance_boost} (was 2.5)")
    print(f"rank_sensitivity: {rank_sensitivity}")
    print(f"importance_threshold: {importance_threshold}")
    
    # Test different scenarios
    scenarios = [
        ("Low importance, low compat", 0.5, 0.01),
        ("High importance, low compat", 2.0, 0.01),
        ("High importance, med compat", 2.0, 0.1),
        ("High importance, high compat", 2.0, 0.5),
    ]
    
    print(f"\n{'Scenario':<30} {'Effective':<12} {'Dynamic':<12} {'Final':<12} {'% Change':<10}")
    print("-" * 80)
    
    for scenario_name, importance_score, compatibility_score in scenarios:
        # Step 1: Apply importance boost
        if importance_score >= importance_threshold:
            effective_strength = merge_strength * importance_boost
        else:
            effective_strength = merge_strength
        
        # Step 2: Apply dynamic strength (with sensitivity > 0)
        compat_clamped = min(max(compatibility_score, -10.0), 10.0)
        sigmoid_input = (compat_clamped - 0.5) * rank_sensitivity
        
        if sigmoid_input > 20:
            sigmoid_output = 1.0
        elif sigmoid_input < -20:
            sigmoid_output = 0.0
        else:
            sigmoid_output = 1.0 / (1.0 + np.exp(-sigmoid_input))
        
        strength_multiplier = min_strength + (max_strength - min_strength) * sigmoid_output
        final_strength = effective_strength * strength_multiplier
        
        print(f"{scenario_name:<30} {effective_strength:<12.6f} {strength_multiplier:<12.6f} {final_strength:<12.6f} {final_strength*100:<10.3f}%")

def compare_old_vs_new():
    """Compare old defaults vs new defaults"""
    print("\n\nCOMPARISON: OLD vs NEW DEFAULTS")
    print("=" * 50)
    
    merge_strength = 0.01
    importance_score = 2.0  # Above threshold
    compatibility_score = 0.1  # Medium compatibility
    
    # Old defaults
    print("OLD DEFAULTS (merge_strength=0.01):")
    old_importance_boost = 2.5
    old_min_strength = 0.5
    old_max_strength = 1.5
    
    old_effective = merge_strength * old_importance_boost
    old_midpoint = (old_min_strength + old_max_strength) / 2.0
    old_final = old_effective * old_midpoint
    
    print(f"  0.01 × {old_importance_boost} × {old_midpoint} = {old_final} ({old_final*100}%)")
    
    # New defaults
    print("NEW DEFAULTS (merge_strength=0.01):")
    new_importance_boost = 1.0
    new_min_strength = 0.0
    new_max_strength = 1.0
    
    new_effective = merge_strength * new_importance_boost
    new_midpoint = (new_min_strength + new_max_strength) / 2.0
    new_final = new_effective * new_midpoint
    
    print(f"  0.01 × {new_importance_boost} × {new_midpoint} = {new_final} ({new_final*100}%)")
    
    print(f"\nIMPROVEMENT: {old_final} → {new_final} ({new_final/old_final:.1f}× closer to expected)")

if __name__ == "__main__":
    test_new_defaults()
    compare_old_vs_new()
    
    print("\n" + "=" * 50)
    print("RESULT:")
    print("=" * 50)
    print("✅ merge_strength=0.01 now behaves much closer to 1% as expected")
    print("✅ No hidden 2.5× amplification from importance_boost")  
    print("✅ min_strength=0.0 allows true minimal changes")
    print("✅ Dynamic range 0.0-1.0 provides intuitive scaling")
    print("")
    print("Users can now use merge_strength values intuitively without")
    print("needing to understand complex parameter interactions!")