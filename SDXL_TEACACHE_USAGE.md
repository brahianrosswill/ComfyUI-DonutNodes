# SDXL TeaCache Usage Guide

## Overview
DonutSDXLTeaCache provides timestep-aware caching specifically designed for SDXL models, accelerating inference by intelligently reusing computation from similar timesteps.

## Nodes Available

### 1. Donut SDXL TeaCache
Main acceleration node that applies caching to SDXL models.

**Inputs:**
- `model`: SDXL model to accelerate
- `rel_l1_thresh`: Cache threshold (0.0-2.0, default: 0.35)
  - Lower = more aggressive caching (faster, potential quality loss)
  - Higher = more conservative caching (slower, better quality)
- `start_percent`: Start caching at this denoising percentage (0.0-1.0)
- `end_percent`: Stop caching at this denoising percentage (0.0-1.0)
- `cache_device`: "cuda" or "cpu" - where to store cache data
- `enable`: Enable/disable caching
- `cache_mode`: Preset strategies:
  - `conservative`: Safest, minimal speedup (~1.2x)
  - `balanced`: Good balance of speed/quality (~1.5x)
  - `aggressive`: Maximum speedup (~2x+, quality loss possible)

**Output:**
- `model`: Accelerated SDXL model

### 2. Donut SDXL TeaCache Stats
Displays cache performance statistics and hit rates.

## Recommended Settings

### For Quality-Focused Work:
```
rel_l1_thresh: 0.5
cache_mode: conservative
start_percent: 0.2
end_percent: 0.9
```

### For Balanced Performance:
```
rel_l1_thresh: 0.35
cache_mode: balanced  
start_percent: 0.0
end_percent: 1.0
```

### For Maximum Speed:
```
rel_l1_thresh: 0.2
cache_mode: aggressive
start_percent: 0.0
end_percent: 1.0
```

## Workflow Example

```
Load Checkpoint -> Donut SDXL TeaCache -> KSampler -> VAE Decode
                            |
                            v
                   Donut SDXL TeaCache Stats (optional)
```

## Performance Expectations

- **Conservative**: 1.2-1.4x speedup, minimal quality impact
- **Balanced**: 1.4-1.8x speedup, slight quality impact
- **Aggressive**: 1.8-2.5x speedup, noticeable quality impact

## Technical Details

### How It Works:
1. **Timestep Analysis**: Monitors timestep changes during denoising
2. **Smart Caching**: Caches UNet outputs when timestep changes are small
3. **Residual Application**: Applies cached residuals to new inputs
4. **Adaptive Thresholding**: Adjusts caching based on accumulated timestep distance

### Memory Usage:
- **CUDA caching**: Higher VRAM usage (~10-20%), faster access
- **CPU caching**: Lower VRAM usage, slightly slower access

### When TeaCache Helps Most:
- ✅ Iterative prompt refinement
- ✅ Batch generation with similar settings
- ✅ Parameter exploration workflows
- ✅ A/B testing different prompts
- ✅ Multiple generations with same model/settings

### When TeaCache Helps Less:
- ❌ Single unique generations
- ❌ Completely different prompts each time
- ❌ Constantly changing model settings
- ❌ Very short sampling (< 20 steps)

## Integration with Optuna

TeaCache works excellently with the Donut Optuna SDXL Optimizer:

```
Load Checkpoint -> Donut SDXL TeaCache -> Donut Optuna SDXL Optimizer
```

This combination provides:
- **Accelerated optimization trials** (2-3x faster per trial)
- **More trials in same time budget**
- **Better exploration of parameter space**

## Troubleshooting

### Poor Quality Results:
- Increase `rel_l1_thresh` (try 0.5-1.0)
- Use "conservative" cache mode
- Reduce caching range (e.g., start_percent: 0.3, end_percent: 0.8)

### Minimal Speedup:
- Decrease `rel_l1_thresh` (try 0.2-0.3)
- Use "aggressive" cache mode
- Use CUDA cache device
- Ensure you're doing iterative work (not single generations)

### Memory Issues:
- Switch cache_device to "cpu"
- Increase `rel_l1_thresh` to cache less
- Reduce batch sizes

## Advanced Usage

### Custom Timestep Ranges:
For specific optimization, you can cache only certain denoising phases:

- **Early denoising only**: start_percent: 0.0, end_percent: 0.5
- **Mid denoising only**: start_percent: 0.3, end_percent: 0.7  
- **Late denoising only**: start_percent: 0.6, end_percent: 1.0

### Combining with Other Optimizations:
TeaCache stacks well with:
- Model compilation (`torch.compile`)
- SDXL Lightning/Turbo models
- Reduced precision inference
- Other DonutNodes optimizations

## Future Enhancements

Planned improvements:
- [ ] Automatic threshold tuning based on model behavior
- [ ] Per-layer caching granularity
- [ ] Quality-aware cache eviction
- [ ] Integration with WIDEN merge optimization
- [ ] Prompt-aware caching strategies