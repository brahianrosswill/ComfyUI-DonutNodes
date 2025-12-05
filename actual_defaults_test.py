#!/usr/bin/env python3
"""
Test the actual behavior with ComfyUI default values to understand why
merge_strength=0.01 doesn't produce near-base results.
"""

import torch
import numpy as np

def test_actual_comfyui_defaults():
    """Test with the actual ComfyUI default values"""
    print("ACTUAL COMFYUI DEFAULTS ANALYSIS")
    print("=" * 50)
    
    # ComfyUI defaults
    merge_strength = 0.01
    min_strength = 0.5
    max_strength = 1.5
    importance_boost = 2.5
    rank_sensitivity = 2.0
    importance_threshold = 1.0
    
    print(f"merge_strength: {merge_strength}")
    print(f"min_strength: {min_strength}")
    print(f"max_strength: {max_strength}")
    print(f"importance_boost: {importance_boost}")
    print(f"rank_sensitivity: {rank_sensitivity}")
    print(f"importance_threshold: {importance_threshold}")
    
    # Simulate parameter with different importance and compatibility
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

def test_corrected_settings():
    """Test with corrected settings for near-base results"""
    print("\n\nCORRECTED SETTINGS FOR NEAR-BASE RESULTS")
    print("=" * 50)
    
    # Corrected settings
    merge_strength = 0.01
    min_strength = 0.0      # Changed from 0.5
    max_strength = 0.2      # Changed from 1.5
    importance_boost = 1.0  # Changed from 2.5
    rank_sensitivity = 0.0  # Changed from 2.0 (disable dynamic)
    importance_threshold = 1.0
    
    print(f"merge_strength: {merge_strength}")
    print(f"min_strength: {min_strength}")
    print(f"max_strength: {max_strength}")
    print(f"importance_boost: {importance_boost}")
    print(f"rank_sensitivity: {rank_sensitivity} (disabled)")
    print(f"importance_threshold: {importance_threshold}")
    
    # Same scenarios
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
        
        # Step 2: Static strength (sensitivity = 0)
        midpoint = (min_strength + max_strength) / 2.0
        final_strength = effective_strength * midpoint
        
        print(f"{scenario_name:<30} {effective_strength:<12.6f} {midpoint:<12.6f} {final_strength:<12.6f} {final_strength*100:<10.3f}%")

def calculate_total_amplification():
    """Calculate the total amplification from all factors"""
    print("\n\nTOTAL AMPLIFICATION ANALYSIS")
    print("=" * 50)
    
    base_merge_strength = 0.01
    
    # ComfyUI defaults amplification
    print("ComfyUI Defaults:")
    importance_factor = 2.5  # importance_boost
    midpoint_factor = (0.5 + 1.5) / 2.0  # (min_strength + max_strength) / 2
    total_amplification = importance_factor * midpoint_factor
    final_effective = base_merge_strength * total_amplification
    
    print(f"  Base merge_strength: {base_merge_strength} (1%)")
    print(f"  Importance boost: ×{importance_factor}")
    print(f"  Midpoint factor: ×{midpoint_factor}")
    print(f"  Total amplification: ×{total_amplification}")
    print(f"  Final effective strength: {final_effective} ({final_effective*100}%)")
    
    # Corrected settings
    print("\nCorrected Settings:")
    importance_factor = 1.0  # no boost
    midpoint_factor = (0.0 + 0.2) / 2.0  # smaller range
    total_amplification = importance_factor * midpoint_factor
    final_effective = base_merge_strength * total_amplification
    
    print(f"  Base merge_strength: {base_merge_strength} (1%)")
    print(f"  Importance boost: ×{importance_factor}")
    print(f"  Midpoint factor: ×{midpoint_factor}")
    print(f"  Total amplification: ×{total_amplification}")
    print(f"  Final effective strength: {final_effective} ({final_effective*100}%)")

if __name__ == "__main__":
    test_actual_comfyui_defaults()
    test_corrected_settings()
    calculate_total_amplification()
    
    print("\n" + "=" * 50)
    print("CONCLUSION:")
    print("=" * 50)
    print("With ComfyUI defaults, merge_strength=0.01 becomes:")
    print("  0.01 × 2.5 (importance_boost) × 1.0 (midpoint) = 0.025 (2.5%)")
    print("")
    print("For truly minimal changes with merge_strength=0.01:")
    print("  1. Set importance_boost = 1.0 (no boost)")
    print("  2. Set min_strength = 0.0, max_strength = 0.2") 
    print("  3. Set rank_sensitivity = 0.0 (disable dynamic)")
    print("  4. Result: 0.01 × 1.0 × 0.1 = 0.001 (0.1%)")
    print("")
    print("The 'bug' is actually by design - the defaults are tuned")
    print("for meaningful merging, not micro-adjustments.")