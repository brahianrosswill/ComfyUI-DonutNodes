#!/usr/bin/env python3
"""
Detailed analysis of strength distribution calculation showing the exact cause of 0.383-1.000 range
"""

import numpy as np

def detailed_analysis():
    """
    Complete analysis showing why strength distribution shows 0.383-1.000 instead of 0.0-1.0
    """
    print("=" * 80)
    print("DETAILED STRENGTH DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # User's data from diagnostic output
    compatibility_min = 0.0002
    compatibility_max = 0.6250
    compatibility_avg = 0.0029
    
    # User's parameters
    merge_strength = 1.0  # User specified "0.0-1.0" range
    min_strength = 0.0    # Node default
    max_strength = 1.0    # Node default
    rank_sensitivity = 1.0
    importance_boost = 2.5
    
    print(f"USER'S CONFIGURATION:")
    print(f"  merge_strength: {merge_strength}")
    print(f"  min_strength: {min_strength}")
    print(f"  max_strength: {max_strength}")
    print(f"  rank_sensitivity: {rank_sensitivity}")
    print(f"  importance_boost: {importance_boost}")
    print(f"")
    
    print(f"ACTUAL COMPATIBILITY SCORES (from diagnostic):")
    print(f"  Range: {compatibility_min:.4f} - {compatibility_max:.4f}")
    print(f"  Average: {compatibility_avg:.4f}")
    print(f"")
    
    # Key issue analysis
    print("=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)
    
    def calculate_strength_step_by_step(compatibility_score, label):
        print(f"\n{label} (compatibility = {compatibility_score}):")
        print("-" * 50)
        
        # Step 1: Scale compatibility  
        scaled = compatibility_score * 100.0
        print(f"  1. Scale compatibility: {compatibility_score} × 100.0 = {scaled}")
        
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
        print(f"  3. Sigmoid output: 1/(1+exp(-{sigmoid_input})) = {sigmoid_output:.6f}")
        
        # Step 4: Map to strength multiplier range
        strength_multiplier = min_strength + (max_strength - min_strength) * sigmoid_output
        print(f"  4. Strength multiplier: {min_strength} + ({max_strength} - {min_strength}) × {sigmoid_output:.6f}")
        print(f"     = {min_strength} + {max_strength - min_strength} × {sigmoid_output:.6f} = {strength_multiplier:.6f}")
        
        # Step 5: Apply merge_strength 
        final_strength = merge_strength * strength_multiplier
        print(f"  5. Final strength: {merge_strength} × {strength_multiplier:.6f} = {final_strength:.6f}")
        
        # Step 6: Apply importance_boost (happens elsewhere)
        with_boost = final_strength * importance_boost
        print(f"  6. With importance_boost: {final_strength:.6f} × {importance_boost} = {with_boost:.6f}")
        
        return final_strength, with_boost
    
    # Analyze the key data points
    min_base, min_boosted = calculate_strength_step_by_step(compatibility_min, "MINIMUM COMPATIBILITY")
    avg_base, avg_boosted = calculate_strength_step_by_step(compatibility_avg, "AVERAGE COMPATIBILITY") 
    max_base, max_boosted = calculate_strength_step_by_step(compatibility_max, "MAXIMUM COMPATIBILITY")
    
    print("\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)
    
    print(f"\nBASE STRENGTH RANGE (before importance_boost):")
    print(f"  Min: {min_base:.6f}")
    print(f"  Max: {max_base:.6f}")
    print(f"  Expected by user: 0.000000 - 1.000000")
    
    print(f"\nWITH IMPORTANCE_BOOST APPLIED:")
    print(f"  Min: {min_boosted:.6f}")  
    print(f"  Max: {max_boosted:.6f}")
    print(f"  User's diagnostic shows: 0.383 - 1.000")
    
    # The key insight: why we don't get 0.0 minimum
    print(f"\n" + "=" * 80)
    print("WHY THE MINIMUM IS 0.383 NOT 0.0")
    print("=" * 80)
    
    print(f"\n1. COMPATIBILITY SCORE SCALING:")
    print(f"   Even the minimum compatibility ({compatibility_min}) gets scaled up:")
    print(f"   {compatibility_min} × 100.0 = {compatibility_min * 100.0}")
    
    print(f"\n2. SIGMOID CENTERING PROBLEM:")
    print(f"   The sigmoid is centered at 0.5, but scaled compatibility is {compatibility_min * 100.0}")
    print(f"   Sigmoid input: ({compatibility_min * 100.0} - 0.5) × {rank_sensitivity} = {(compatibility_min * 100.0 - 0.5) * rank_sensitivity}")
    print(f"   This gives sigmoid output = {1.0 / (1.0 + np.exp(-((compatibility_min * 100.0 - 0.5) * rank_sensitivity))):.6f}")
    
    print(f"\n3. STRENGTH MULTIPLIER NEVER REACHES 0:")
    sigmoid_out_min = 1.0 / (1.0 + np.exp(-((compatibility_min * 100.0 - 0.5) * rank_sensitivity)))
    print(f"   With min_strength=0.0, max_strength=1.0:")
    print(f"   strength_multiplier = 0.0 + (1.0 - 0.0) × {sigmoid_out_min:.6f} = {sigmoid_out_min:.6f}")
    print(f"   This is why the minimum is {sigmoid_out_min:.3f}, not 0.0!")
    
    print(f"\n4. DIAGNOSTIC RANGE EXPLANATION:")
    print(f"   Base range: {min_base:.3f} - {max_base:.3f}")
    print(f"   But importance_boost={importance_boost} is applied elsewhere, giving:")
    print(f"   Boosted range: {min_boosted:.3f} - {max_boosted:.3f}")
    print(f"   However, diagnostic shows 0.383 - 1.000, suggesting:")
    print(f"   - The 0.383 matches our {min_base:.3f} (no importance_boost on minimum)")
    print(f"   - The 1.000 suggests clamping or different calculation for maximum")
    
    # Find where the threshold might be
    print(f"\n" + "=" * 80)
    print("JIT VERSION DISCREPANCY ANALYSIS")  
    print("=" * 80)
    
    print(f"\nThe JIT version in _fast_sigmoid_strength has been FIXED!")
    print(f"Line 160: scaled_compatibility = compatibility_tensor * 100.0")
    print(f"Line 161: sigmoid_input = (scaled_compatibility - 0.5) * sensitivity")
    print(f"Now matches the non-JIT version:")
    print(f"Line 124: sigmoid_input = (scaled_compatibility - 0.5) * sensitivity")
    print(f"")
    print(f"Both functions should now behave identically!")
    print(f"This means for compatibility_tensor values like 0.0002:")
    print(f"  JIT: sigmoid_input = (0.0002 - 0.5) × 1.0 = -0.4998")
    print(f"  Non-JIT: sigmoid_input = (0.02 - 0.5) × 1.0 = -0.48")
    
    # Test JIT behavior
    jit_sigmoid_input = (compatibility_min - 0.5) * rank_sensitivity
    jit_sigmoid_output = 1.0 / (1.0 + np.exp(-jit_sigmoid_input))
    jit_strength_multiplier = min_strength + (max_strength - min_strength) * jit_sigmoid_output
    jit_final = merge_strength * jit_strength_multiplier
    
    print(f"\nJIT VERSION CALCULATION (with bug):")
    print(f"  Sigmoid input: {jit_sigmoid_input:.6f}")
    print(f"  Sigmoid output: {jit_sigmoid_output:.6f}")
    print(f"  Final strength: {jit_final:.6f}")
    print(f"  This gives minimum ~{jit_final:.3f} which is close to the diagnostic 0.383!")
    
if __name__ == "__main__":
    detailed_analysis()