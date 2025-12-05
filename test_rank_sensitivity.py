#!/usr/bin/env python3
"""
Quick test to verify rank_sensitivity behavior with actual compatibility values
"""

import numpy as np

def test_rank_sensitivity_effect():
    """Test how rank_sensitivity affects strength distribution with realistic compatibility values"""
    
    # Your actual compatibility range from the diagnostic
    compatibility_scores = [0.0002, 0.001, 0.01, 0.1, 0.6250]
    
    # Test parameters
    merge_strength = 1.0
    min_strength = 0.0  
    max_strength = 1.0
    
    # Test different sensitivity values
    sensitivity_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    print("Compatibility → Strength mapping with different rank_sensitivity values:")
    print("=" * 80)
    
    for sensitivity in sensitivity_values:
        print(f"\nrank_sensitivity = {sensitivity}")
        strengths = []
        
        for comp in compatibility_scores:
            # Reproduce the exact calculation from the merge function
            scaled_compatibility = comp * 100.0
            sigmoid_input = (scaled_compatibility - 0.5) * sensitivity
            
            # Apply sigmoid with numerical stability (same as the code)
            if sigmoid_input > 20:
                sigmoid_output = 1.0
            elif sigmoid_input < -20:
                sigmoid_output = 0.0
            else:
                sigmoid_output = 1.0 / (1.0 + np.exp(-sigmoid_input))
            
            # Map to strength range
            strength_multiplier = min_strength + (max_strength - min_strength) * sigmoid_output
            final_strength = merge_strength * strength_multiplier
            
            strengths.append(final_strength)
        
        # Show the mapping
        for i, (comp, strength) in enumerate(zip(compatibility_scores, strengths)):
            print(f"  {comp:8.4f} → {strength:.3f}")
        
        min_str, max_str, avg_str = min(strengths), max(strengths), np.mean(strengths)
        print(f"  Range: {min_str:.3f} - {max_str:.3f} (avg: {avg_str:.3f})")

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("If the ranges look similar across different rank_sensitivity values,")
    print("then the sigmoid function is saturating and sensitivity has minimal effect.")
    print("If they vary significantly, then rank_sensitivity is working correctly.")

if __name__ == "__main__":
    test_rank_sensitivity_effect()