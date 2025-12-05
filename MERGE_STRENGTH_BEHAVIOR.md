# Merge Strength Behavior Documentation

## Critical Understanding for Future Debugging

### The Problem
Users expect `merge_strength=0.01` to produce a model nearly identical to the base model (1% maximum change). However, with default settings, this doesn't happen.

### Why This Occurs

The final merge strength is calculated as:
```
final_strength = merge_strength * (min_strength + range * sigmoid_factor)
```

Where `range = max_strength - min_strength`

### Default Values Impact

With ComfyUI defaults:
- `min_strength = 0.5`
- `max_strength = 1.5`
- `merge_strength = 0.01`

The actual strength range becomes:
- **Minimum**: `0.01 * 0.5 = 0.005` (0.5% change)
- **Maximum**: `0.01 * 1.5 = 0.015` (1.5% change)

Even the "minimum" is still 0.5% change, not near-zero!

### The Solution

For near-base results with very low merge strengths (< 0.1):

#### Option 1: Adjust min_strength
- Set `min_strength = 0.0`
- Keep `max_strength = 1.0`
- Result: actual range 0.0 to 0.01 (0% to 1% change)

#### Option 2: Use sensitivity=0.0
- When `sensitivity = 0.0`, dynamic strength is disabled
- Uses fixed: `merge_strength * (min_strength + max_strength) / 2`
- With defaults: `0.01 * (0.5 + 1.5) / 2 = 0.01` (1% change)

### Recurring Bug History

**The Clamping Bug** (FIXED):
```python
# WRONG (old code):
final_strength = scaled_strength * merge_strength
return torch.clamp(final_strength, min_strength, max_strength)

# CORRECT (fixed):
final_strength = scaled_strength * merge_strength
min_clamped = min_strength * merge_strength
max_clamped = max_strength * merge_strength
return torch.clamp(final_strength, min_clamped, max_clamped)
```

The old version clamped to the unscaled range, which prevented very low merge strengths from working properly.

### Test Cases

Run `merge_strength_test.py` to verify behavior:

```bash
python merge_strength_test.py
```

Expected results:
- `merge_strength=0.01, min_strength=0.5` → minimum 0.5% change (not near-base)
- `merge_strength=0.01, min_strength=0.0` → minimum 0.0% change (near-base)

### For ComfyUI Users

**CRITICAL DISCOVERY**: With ComfyUI defaults, `merge_strength=0.01` actually produces **2.5% effective change**, not 1%!

#### Why This Happens (Total Amplification)

The actual strength calculation is:
```
final_strength = merge_strength × importance_boost × dynamic_strength_factor
```

With ComfyUI defaults:
- `merge_strength = 0.01` (1%)
- `importance_boost = 2.5` (2.5× amplification for important parameters)
- `dynamic_strength_factor ≈ 1.0` (midpoint of min_strength=0.5, max_strength=1.5)
- **Result**: `0.01 × 2.5 × 1.0 = 0.025` (2.5% effective change)

#### To Get Truly Minimal Changes

For `merge_strength=0.01` to behave like 0.1% change:

1. **Set `importance_boost = 1.0`** (disable importance amplification)
2. **Set `min_strength = 0.0`** and `max_strength = 0.2` (reduce midpoint to 0.1)
3. **Set `rank_sensitivity = 0.0`** (disable dynamic strength for consistency)
4. **Result**: `0.01 × 1.0 × 0.1 = 0.001` (0.1% effective change)

#### Quick Reference

| Setting | Default | For Near-Base | Purpose |
|---------|---------|---------------|---------|
| `merge_strength` | 1.0 | 0.01 | Base merge intensity |
| `importance_boost` | 2.5 | 1.0 | Amplification for important params |
| `min_strength` | 0.5 | 0.0 | Minimum dynamic strength |
| `max_strength` | 1.5 | 0.2 | Maximum dynamic strength |
| `rank_sensitivity` | 2.0 | 0.0 | Dynamic strength sensitivity |

### Implementation Notes

1. **Two Functions Must Stay in Sync**:
   - `_compatibility_to_merge_strength()` (regular)
   - `_fast_sigmoid_strength()` (JIT-compiled)

2. **Both Locations**:
   - `merge_strength.py` (modular)
   - `DonutWidenMerge.py` (main file)

3. **Critical Code Sections**:
   - Clamping logic must use scaled ranges
   - Documentation must explain the behavior
   - Test cases must verify low merge_strength behavior

### Design Intent

The default `min_strength=0.5` is intentional for normal merging scenarios where users want meaningful parameter changes. It prevents "do nothing" merges.

For fine-tuning and micro-adjustments, users should manually set `min_strength=0.0`.