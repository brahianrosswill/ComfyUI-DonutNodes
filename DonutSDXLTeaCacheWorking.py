import torch
import comfy.model_management as mm


def working_teacache_unet_forward(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        control=None,
        transformer_options={},
        **kwargs
    ):
    """Simple working TeaCache that focuses on step-level caching."""
    
    # Get options
    cache_threshold = transformer_options.get("rel_l1_thresh", 0.1)
    enable_teacache = transformer_options.get("enable_teacache", True)
    
    if not enable_teacache:
        return self._teacache_original_forward(x, timesteps, context, y, control, transformer_options, **kwargs)
    
    # Simple step-based caching
    if timesteps is not None and len(timesteps) > 0:
        timestep_val = timesteps[0].item()
    else:
        timestep_val = 0.0
    
    # Initialize cache
    if not hasattr(self, '_working_cache'):
        self._working_cache = {}
        self._working_stats = {'hits': 0, 'misses': 0, 'skips': 0}
    
    # Create a simple cache key based on timestep and input characteristics
    input_shape = x.shape
    input_mean = x.mean().item()
    input_std = x.std().item()
    
    # Use rounded timestep for cache key (group similar timesteps)
    timestep_rounded = round(timestep_val, 1)
    cache_key = (input_shape, timestep_rounded, round(input_mean, 3), round(input_std, 3))
    
    # Check cache
    if cache_key in self._working_cache:
        cached_result, cached_input = self._working_cache[cache_key]
        
        # Simple similarity check
        input_diff = (x - cached_input).abs().mean().item()
        similarity_threshold = cache_threshold
        
        if input_diff < similarity_threshold:
            # Cache hit!
            self._working_stats['hits'] += 1
            # Add small noise to avoid exact duplication
            noise_scale = 1e-5
            result = cached_result + torch.randn_like(cached_result) * noise_scale
            return result.to(x.device)
        else:
            # Similar timestep but different input, update cache
            self._working_stats['skips'] += 1
    
    # Cache miss - compute result
    self._working_stats['misses'] += 1
    result = self._teacache_original_forward(x, timesteps, context, y, control, transformer_options, **kwargs)
    
    # Store in cache (with size limit)
    if len(self._working_cache) < 10:  # Limit cache size
        self._working_cache[cache_key] = (result.clone(), x.clone())
    elif len(self._working_cache) >= 20:  # Clean cache if too large
        # Remove oldest entries
        keys_to_remove = list(self._working_cache.keys())[:5]
        for key in keys_to_remove:
            del self._working_cache[key]
        self._working_cache[cache_key] = (result.clone(), x.clone())
    
    return result


class DonutSDXLTeaCacheWorking:
    """Simple working SDXL TeaCache implementation."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "SDXL model to apply working TeaCache to"}),
                "cache_strength": ("FLOAT", {
                    "default": 0.1, 
                    "min": 0.001, 
                    "max": 1.0, 
                    "step": 0.001,
                    "tooltip": "Lower = more aggressive caching (0.05-0.2 recommended)"
                }),
                "enable": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "Enable/disable TeaCache"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_working_cache"
    CATEGORY = "DonutNodes"
    TITLE = "Donut SDXL TeaCache (Working)"
    
    def apply_working_cache(self, model, cache_strength: float, enable: bool):
        """Apply working TeaCache to SDXL model."""
        
        if not enable:
            return (model,)

        new_model = model.clone()
        
        # Set options
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {}
            
        new_model.model_options["transformer_options"]["rel_l1_thresh"] = cache_strength
        new_model.model_options["transformer_options"]["enable_teacache"] = enable
        
        diffusion_model = new_model.get_model_object("diffusion_model")

        # Store original and patch
        if not hasattr(diffusion_model, '_teacache_original_forward'):
            diffusion_model._teacache_original_forward = diffusion_model.forward
            diffusion_model.forward = working_teacache_unet_forward.__get__(diffusion_model, diffusion_model.__class__)
        
        return (new_model,)


class DonutSDXLTeaCacheWorkingStats:
    """Stats for working TeaCache."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model with working TeaCache"}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "stats")
    FUNCTION = "get_working_stats"
    CATEGORY = "DonutNodes"
    TITLE = "Donut SDXL TeaCache (Working) Stats"
    
    def get_working_stats(self, model):
        """Get working cache statistics."""
        diffusion_model = model.get_model_object("diffusion_model")
        
        stats_text = "Working SDXL TeaCache Stats:\n\n"
        
        if hasattr(diffusion_model, '_working_stats'):
            stats = diffusion_model._working_stats
            hits = stats['hits']
            misses = stats['misses']
            skips = stats['skips']
            total = hits + misses + skips
            
            stats_text += f"Cache hits: {hits}\n"
            stats_text += f"Cache misses: {misses}\n"
            stats_text += f"Cache skips: {skips}\n"
            stats_text += f"Total operations: {total}\n"
            
            if hasattr(diffusion_model, '_working_cache'):
                cache_size = len(diffusion_model._working_cache)
                stats_text += f"Cache entries: {cache_size}\n\n"
            
            if total > 0:
                hit_rate = (hits / total) * 100
                effective_rate = (hits / (hits + misses)) * 100 if (hits + misses) > 0 else 0
                
                stats_text += f"Overall hit rate: {hit_rate:.1f}%\n"
                stats_text += f"Effective hit rate: {effective_rate:.1f}%\n\n"
                
                if hit_rate > 20:
                    stats_text += "🎉 EXCELLENT cache performance!"
                elif hit_rate > 10:
                    stats_text += "✅ GOOD cache performance"
                elif hit_rate > 5:
                    stats_text += "⚠️ MODERATE cache performance"
                elif hit_rate > 0:
                    stats_text += "🔄 Cache starting to work"
                else:
                    stats_text += "❌ No cache benefits yet"
                    
                # Suggestions
                if hit_rate < 5 and total > 10:
                    stats_text += "\n\n💡 Try lowering cache_strength for more aggressive caching"
                elif hit_rate > 50:
                    stats_text += "\n\n💡 Consider raising cache_strength to maintain quality"
            else:
                stats_text += "No cache activity yet"
        else:
            stats_text += "No cache statistics found.\nTeaCache may not be applied or used yet."
        
        return (model, stats_text)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "DonutSDXLTeaCacheWorking": DonutSDXLTeaCacheWorking,
    "DonutSDXLTeaCacheWorkingStats": DonutSDXLTeaCacheWorkingStats,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutSDXLTeaCacheWorking": "Donut SDXL TeaCache (Working)",
    "DonutSDXLTeaCacheWorkingStats": "Donut SDXL TeaCache (Working) Stats",
}