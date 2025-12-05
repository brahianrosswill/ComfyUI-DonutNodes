#!/usr/bin/env python3
"""
Analyze the strength distribution calculation to understand why 0.383-1.000 instead of 0.0-1.0
"""

import numpy as np

def analyze_strength_distribution():
    """
    Reproduce the strength calculation to understand the 0.383-1.000 range issue
    """
    print("=== STRENGTH DISTRIBUTION ANALYSIS ===\n")
    
    # User's diagnostic data
    compatibility_min = 0.0002
    compatibility_max = 0.6250
    compatibility_avg = 0.0029
    
    # User's parameters
    merge_strength = 1.0  # User specified range 0.0-1.0, so this should be 1.0
    min_strength = 0.0    # From node defaults
    max_strength = 1.0    # From node defaults  
    rank_sensitivity = 1.0
    importance_boost = 2.5
    
    print(f"INPUT PARAMETERS:")
    print(f"  merge_strength: {merge_strength}")
    print(f"  min_strength: {min_strength}")
    print(f"  max_strength: {max_strength}")
    print(f"  rank_sensitivity: {rank_sensitivity}")
    print(f"  importance_boost: {importance_boost}")
    print(f"")
    
    print(f"COMPATIBILITY SCORES:")
    print(f"  Range: {compatibility_min} - {compatibility_max}")
    print(f"  Average: {compatibility_avg}")
    print(f"")
    
    # Test the strength calculation function
    def _compatibility_to_merge_strength(compatibility_score, merge_strength, min_strength, max_strength, sensitivity):
        """Reproduce the exact calculation from shared/merge_strength.py"""
        
        # Special case: merge_strength=0 should return 0 (base model unchanged)
        if merge_strength == 0.0:
            return 0.0
        
        # If sensitivity is 0, disable dynamic strength (use merge_strength directly)
        if sensitivity == 0.0:
            return float(merge_strength)
        
        # Scale the raw compatibility score for better sigmoid behavior
        scaled_compatibility = compatibility_score * 100.0  # Scale up small values
        sigmoid_input = (scaled_compatibility - 0.5) * sensitivity
        
        # Apply sigmoid with numerical stability
        if sigmoid_input > 20:
            sigmoid_output = 1.0
        elif sigmoid_input < -20:
            sigmoid_output = 0.0
        else:
            sigmoid_output = 1.0 / (1.0 + np.exp(-sigmoid_input))
        
        # Map sigmoid to strength multiplier range: min_strength to max_strength  
        strength_multiplier = min_strength + (max_strength - min_strength) * sigmoid_output
        
        # Apply merge_strength as the final scaling factor
        final_strength = merge_strength * strength_multiplier
        
        return float(final_strength)
    
    # Calculate strengths for the compatibility range
    test_compatibilities = [compatibility_min, compatibility_avg, compatibility_max]
    print("STRENGTH CALCULATION BREAKDOWN:")
    print("=" * 70)
    
    calculated_strengths = []
    
    for comp in test_compatibilities:
        print(f"\nCompatibility Score: {comp}")
        
        # Step 1: Scale compatibility
        scaled = comp * 100.0
        print(f"  1. Scaled compatibility: {comp} × 100.0 = {scaled}")
        
        # Step 2: Sigmoid input  
        sigmoid_input = (scaled - 0.5) * rank_sensitivity
        print(f"  2. Sigmoid input: ({scaled} - 0.5) × {rank_sensitivity} = {sigmoid_input}")
        
        # Step 3: Sigmoid output
        if sigmoid_input > 20:
            sigmoid_output = 1.0
        elif sigmoid_input < -20:
            sigmoid_output = 0.0
        else:
            sigmoid_output = 1.0 / (1.0 + np.exp(-sigmoid_input))
        print(f"  3. Sigmoid output: 1/(1+exp(-{sigmoid_input})) = {sigmoid_output}")
        
        # Step 4: Strength multiplier
        strength_multiplier = min_strength + (max_strength - min_strength) * sigmoid_output
        print(f"  4. Strength multiplier: {min_strength} + ({max_strength} - {min_strength}) × {sigmoid_output} = {strength_multiplier}")
        
        # Step 5: Final strength (before importance_boost)
        final_strength = merge_strength * strength_multiplier
        print(f"  5. Final strength: {merge_strength} × {strength_multiplier} = {final_strength}")
        
        # Step 6: With importance_boost applied (this happens elsewhere in the code)
        final_with_boost = final_strength * importance_boost
        print(f"  6. With importance_boost: {final_strength} × {importance_boost} = {final_with_boost}")
        
        calculated_strengths.append(final_with_boost)
    
    print(f"\n" + "=" * 70)
    print(f"CALCULATED STRENGTH RANGE:")
    print(f"  Min: {min(calculated_strengths):.3f}")
    print(f"  Max: {max(calculated_strengths):.3f}")
    print(f"  Expected from user: 0.000 - 1.000")
    print(f"  Actual from diagnostic: 0.383 - 1.000")
    
    # Let's also test with a wider range of compatibility scores
    print(f"\n" + "=" * 70)
    print("FULL RANGE ANALYSIS:")
    
    # Test with 1000 points across the compatibility range  
    test_range = np.linspace(compatibility_min, compatibility_max, 1000)
    all_strengths = []
    
    for comp in test_range:
        strength = _compatibility_to_merge_strength(comp, merge_strength, min_strength, max_strength, rank_sensitivity)
        strength_with_boost = strength * importance_boost
        all_strengths.append(strength_with_boost)
    
    print(f"  Computed range: {min(all_strengths):.3f} - {max(all_strengths):.3f}")
    
    # Find where the minimum occurs
    min_idx = np.argmin(all_strengths)
    min_comp = test_range[min_idx]
    print(f"  Minimum occurs at compatibility: {min_comp}")
    
    # Analyze the sigmoid behavior at the minimum
    scaled_min = min_comp * 100.0
    sigmoid_input_min = (scaled_min - 0.5) * rank_sensitivity
    sigmoid_output_min = 1.0 / (1.0 + np.exp(-sigmoid_input_min)) if abs(sigmoid_input_min) < 20 else (1.0 if sigmoid_input_min > 20 else 0.0)
    
    print(f"\nAT MINIMUM COMPATIBILITY ({min_comp}):")
    print(f"  Scaled: {scaled_min}")
    print(f"  Sigmoid input: {sigmoid_input_min}")
    print(f"  Sigmoid output: {sigmoid_output_min}")
    
    # This explains why we don't reach 0.0!
    print(f"\nROOT CAUSE ANALYSIS:")
    print(f"  The minimum compatibility score ({compatibility_min}) is still > 0")
    print(f"  When scaled by 100: {compatibility_min * 100.0}")
    print(f"  Sigmoid input: ({compatibility_min * 100.0} - 0.5) × {rank_sensitivity} = {(compatibility_min * 100.0 - 0.5) * rank_sensitivity}")
    print(f"  This gives a sigmoid output > 0, not 0")
    print(f"  So the minimum strength is min_strength + (max_strength - min_strength) × sigmoid_output")
    print(f"  Which equals: {min_strength} + ({max_strength} - {min_strength}) × {sigmoid_output_min} = {min_strength + (max_strength - min_strength) * sigmoid_output_min}")
    print(f"  Final: {(min_strength + (max_strength - min_strength) * sigmoid_output_min) * merge_strength * importance_boost}")

if __name__ == "__main__":
    analyze_strength_distribution()