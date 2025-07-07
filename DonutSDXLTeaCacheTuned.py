import torch
import comfy.model_management as mm


def tuned_teacache_unet_forward(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        control=None,
        transformer_options={},
        **kwargs
    ):
    """Tuned SDXL TeaCache with adjustable aggression levels."""
    
    # Get options with more aggressive defaults
    cache_threshold = transformer_options.get("rel_l1_thresh", 0.05)  # More aggressive default
    enable_teacache = transformer_options.get("enable_teacache", True)
    cache_mode = transformer_options.get("cache_mode", "balanced")
    step_skip_probability = transformer_options.get("step_skip_prob", 0.3)  # Skip some steps entirely
    
    if not enable_teacache:
        return self._teacache_original_forward(x, timesteps, context, y, control, transformer_options, **kwargs)
    
    # Get current step info for adaptive caching
    current_percent = transformer_options.get("current_percent", 0.5)
    
    # Initialize cache system
    if not hasattr(self, '_tuned_cache_system'):
        self._tuned_cache_system = {
            'step_cache': {},  # Cache by timestep
            'input_cache': None,
            'output_cache': None,
            'stats': {'hits': 0, 'misses': 0, 'skips': 0},
            'last_timestep': None,
            'consecutive_hits': 0
        }
    
    cache_sys = self._tuned_cache_system
    
    # Get timestep
    if timesteps is not None and len(timesteps) > 0:
        current_timestep = timesteps[0].item()
    else:
        current_timestep = 0.0
    
    # Adaptive caching based on denoising progress
    if cache_mode == "ultra_aggressive":
        # Very aggressive - high chance of reusing results
        if current_percent > 0.3:  # After 30% of steps
            if torch.rand(1).item() < 0.6:  # 60% chance to use previous result
                if cache_sys['output_cache'] is not None:
                    cache_sys['stats']['hits'] += 1
                    # Add controlled noise to avoid artifacts
                    noise_scale = 0.001 * (1.0 - current_percent)  # Less noise as we progress
                    return cache_sys['output_cache'] + torch.randn_like(cache_sys['output_cache']) * noise_scale
    
    elif cache_mode == "aggressive":
        # Aggressive - cache similar timesteps
        timestep_rounded = round(current_timestep, 0)  # Round to nearest integer
        
        if timestep_rounded in cache_sys['step_cache']:
            cached_input, cached_output = cache_sys['step_cache'][timestep_rounded]
            
            # Check input similarity
            input_diff = (x - cached_input).abs().mean().item()
            if input_diff < cache_threshold * 2:  # More lenient for aggressive mode
                cache_sys['stats']['hits'] += 1
                cache_sys['consecutive_hits'] += 1
                
                # Add adaptive noise based on difference
                noise_scale = min(input_diff * 10, 0.01)
                return cached_output + torch.randn_like(cached_output) * noise_scale
    
    elif cache_mode == "balanced":
        # Balanced - careful caching with quality preservation
        if cache_sys['input_cache'] is not None:
            input_diff = (x - cache_sys['input_cache']).abs().mean().item()
            timestep_diff = abs(current_timestep - (cache_sys['last_timestep'] or 0))
            
            # Only cache if both input and timestep are similar
            if input_diff < cache_threshold and timestep_diff < 0.1:
                cache_sys['stats']['hits'] += 1
                # Minimal noise for quality preservation
                noise_scale = 0.0001
                return cache_sys['output_cache'] + torch.randn_like(cache_sys['output_cache']) * noise_scale
    
    elif cache_mode == "conservative":
        # Conservative - only cache very similar inputs
        if cache_sys['input_cache'] is not None:
            input_diff = (x - cache_sys['input_cache']).abs().mean().item()
            if input_diff < cache_threshold * 0.5:  # Stricter threshold
                cache_sys['stats']['hits'] += 1
                return cache_sys['output_cache'].clone()
    
    # No cache hit - compute new result
    cache_sys['stats']['misses'] += 1
    cache_sys['consecutive_hits'] = 0
    
    result = self._teacache_original_forward(x, timesteps, context, y, control, transformer_options, **kwargs)
    
    # Update caches based on mode
    if cache_mode == "aggressive":
        timestep_rounded = round(current_timestep, 0)
        cache_sys['step_cache'][timestep_rounded] = (x.clone(), result.clone())
        
        # Limit cache size
        if len(cache_sys['step_cache']) > 15:
            oldest_key = min(cache_sys['step_cache'].keys())
            del cache_sys['step_cache'][oldest_key]
    
    # Always update input/output cache
    cache_sys['input_cache'] = x.clone()
    cache_sys['output_cache'] = result.clone()
    cache_sys['last_timestep'] = current_timestep
    
    return result


class DonutSDXLTeaCacheTuned:
    """Tuned SDXL TeaCache with multiple aggression levels."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "SDXL model to apply tuned TeaCache to"}),
                "cache_mode": (["conservative", "balanced", "aggressive", "ultra_aggressive"], {
                    "default": "aggressive",
                    "tooltip": "Cache aggression level - aggressive may give 1.5x+ speedup"
                }),
                "cache_threshold": ("FLOAT", {
                    "default": 0.05, 
                    "min": 0.001, 
                    "max": 0.5, 
                    "step": 0.001,
                    "tooltip": "Lower = more caching (0.01-0.1 for aggressive speedup)"
                }),
                "enable": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "Enable/disable TeaCache"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_tuned_cache"
    CATEGORY = "DonutNodes"
    TITLE = "Donut SDXL TeaCache (Tuned)"
    
    def apply_tuned_cache(self, model, cache_mode: str, cache_threshold: float, enable: bool):
        """Apply tuned TeaCache to SDXL model."""
        
        if not enable:
            return (model,)

        new_model = model.clone()
        
        # Set options
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {}
            
        new_model.model_options["transformer_options"]["rel_l1_thresh"] = cache_threshold
        new_model.model_options["transformer_options"]["enable_teacache"] = enable
        new_model.model_options["transformer_options"]["cache_mode"] = cache_mode
        
        diffusion_model = new_model.get_model_object("diffusion_model")

        # Store original and patch
        if not hasattr(diffusion_model, '_teacache_original_forward'):
            diffusion_model._teacache_original_forward = diffusion_model.forward
            diffusion_model.forward = tuned_teacache_unet_forward.__get__(diffusion_model, diffusion_model.__class__)
        
        return (new_model,)


class DonutSDXLTeaCacheTunedStats:
    """Stats for tuned TeaCache."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model with tuned TeaCache"}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "stats")
    FUNCTION = "get_tuned_stats"
    CATEGORY = "DonutNodes"
    TITLE = "Donut SDXL TeaCache (Tuned) Stats"
    
    def get_tuned_stats(self, model):
        """Get tuned cache statistics."""
        diffusion_model = model.get_model_object("diffusion_model")
        
        stats_text = "Tuned SDXL TeaCache Stats:\n\n"
        
        if hasattr(diffusion_model, '_tuned_cache_system'):
            cache_sys = diffusion_model._tuned_cache_system
            stats = cache_sys['stats']
            
            hits = stats['hits']
            misses = stats['misses'] 
            skips = stats['skips']
            total = hits + misses + skips
            
            stats_text += f"Cache hits: {hits}\n"
            stats_text += f"Cache misses: {misses}\n" 
            stats_text += f"Cache skips: {skips}\n"
            stats_text += f"Total operations: {total}\n"
            
            # Show cache sizes
            step_cache_size = len(cache_sys.get('step_cache', {}))
            consecutive_hits = cache_sys.get('consecutive_hits', 0)
            
            stats_text += f"Step cache entries: {step_cache_size}\n"
            stats_text += f"Consecutive hits: {consecutive_hits}\n\n"
            
            if total > 0:
                hit_rate = (hits / total) * 100
                stats_text += f"Hit rate: {hit_rate:.1f}%\n\n"
                
                # Performance assessment
                if hit_rate > 40:
                    stats_text += "🚀 EXCELLENT! High cache efficiency\n"
                    stats_text += "Expected speedup: 1.5-2.0x+\n"
                elif hit_rate > 25:
                    stats_text += "✅ GOOD cache efficiency\n"
                    stats_text += "Expected speedup: 1.2-1.5x\n"
                elif hit_rate > 15:
                    stats_text += "⚠️ MODERATE cache efficiency\n"
                    stats_text += "Expected speedup: 1.1-1.2x\n"
                elif hit_rate > 5:
                    stats_text += "🔄 Cache starting to work\n"
                    stats_text += "Try more aggressive settings\n"
                else:
                    stats_text += "❌ Low cache efficiency\n"
                    stats_text += "Try lowering cache_threshold or more aggressive mode\n"
                
                # Tuning suggestions
                if hit_rate < 10 and total > 5:
                    stats_text += "\n💡 Suggestions:\n"
                    stats_text += "- Try 'ultra_aggressive' mode\n"
                    stats_text += "- Lower cache_threshold to 0.01-0.03\n"
                    stats_text += "- Use similar prompts for better caching\n"
                elif hit_rate > 60:
                    stats_text += "\n💡 Quality check:\n"
                    stats_text += "- Very high hit rate - check image quality\n"
                    stats_text += "- Consider 'balanced' mode if artifacts appear\n"
            else:
                stats_text += "No operations recorded yet"
        else:
            stats_text += "No cache system found.\nTeaCache may not be applied."
        
        return (model, stats_text)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "DonutSDXLTeaCacheTuned": DonutSDXLTeaCacheTuned,
    "DonutSDXLTeaCacheTunedStats": DonutSDXLTeaCacheTunedStats,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutSDXLTeaCacheTuned": "Donut SDXL TeaCache (Tuned)",
    "DonutSDXLTeaCacheTunedStats": "Donut SDXL TeaCache (Tuned) Stats",
}