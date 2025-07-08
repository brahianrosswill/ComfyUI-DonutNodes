import torch
import comfy.model_management as mm
from unittest.mock import patch


def poly1d(coefficients, x):
    """Polynomial evaluation for cache distance calculation."""
    result = torch.zeros_like(x)
    for i, coeff in enumerate(coefficients):
        result += coeff * (x ** (len(coefficients) - 1 - i))
    return result


def teacache_sdxl_unet_forward(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        control=None,
        transformer_options={},
        **kwargs
    ):
    """SDXL UNet forward pass with TeaCache acceleration."""
    
    # Get TeaCache options
    rel_l1_thresh = transformer_options.get("rel_l1_thresh", 0.1)
    coefficients = transformer_options.get("coefficients", [1.0, -0.5, 0.1])
    enable_teacache = transformer_options.get("enable_teacache", True)
    cache_device = transformer_options.get("cache_device", x.device)
    
    # For SDXL, we use timestep and input as cache key indicators
    cache_input = torch.cat([x.flatten()[:100], timesteps.flatten()[:10]]).to(cache_device)
    
    if not enable_teacache:
        # TeaCache disabled, run original forward
        return self._teacache_original_forward(x, timesteps, context, y, control, transformer_options, **kwargs)
    
    # Initialize cache state
    if not hasattr(self, 'teacache_accumulated_distance'):
        self.teacache_accumulated_distance = 0.0
        self.teacache_previous_input = None
        self.teacache_previous_output = None
        self.teacache_should_calc = True
    
    # Calculate distance from previous input
    if self.teacache_previous_input is not None:
        try:
            input_diff = (cache_input - self.teacache_previous_input).abs().mean()
            input_norm = self.teacache_previous_input.abs().mean()
            
            if input_norm > 1e-8:
                rel_l1 = input_diff / input_norm
                distance_increment = poly1d(coefficients, rel_l1)
                self.teacache_accumulated_distance += distance_increment.item()
                
                if self.teacache_accumulated_distance < rel_l1_thresh:
                    self.teacache_should_calc = False
                else:
                    self.teacache_should_calc = True
                    self.teacache_accumulated_distance = 0.0
            else:
                self.teacache_should_calc = True
        except Exception:
            self.teacache_should_calc = True
            self.teacache_accumulated_distance = 0.0
    
    # Store current input for next comparison
    self.teacache_previous_input = cache_input.clone()
    
    if not self.teacache_should_calc and self.teacache_previous_output is not None:
        # Use cached output
        return self.teacache_previous_output.to(x.device)
    else:
        # Calculate new output
        result = self._teacache_original_forward(x, timesteps, context, y, control, transformer_options, **kwargs)
        
        # Cache the result
        self.teacache_previous_output = result.to(cache_device).clone()
        
        return result


class DonutSDXLTeaCache:
    """SDXL TeaCache node for ComfyUI that accelerates SDXL diffusion models."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The SDXL diffusion model TeaCache will be applied to."}),
                "rel_l1_thresh": ("FLOAT", {
                    "default": 0.1, 
                    "min": 0.0, 
                    "max": 10.0, 
                    "step": 0.01, 
                    "tooltip": "Cache threshold - lower values = more aggressive caching. Must be non-negative."
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01, 
                    "tooltip": "Start percentage of denoising steps to apply TeaCache."
                }),
                "end_percent": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01, 
                    "tooltip": "End percentage of denoising steps to apply TeaCache."
                }),
                "cache_device": (["cuda", "cpu"], {
                    "default": "cuda", 
                    "tooltip": "Device where cache will reside."
                }),
                "enable": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "Enable/disable TeaCache."
                }),
                "cache_mode": (["conservative", "balanced", "aggressive"], {
                    "default": "balanced", 
                    "tooltip": "Cache mode preset that adjusts rel_l1_thresh."
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_teacache"
    CATEGORY = "DonutNodes"
    TITLE = "Donut SDXL TeaCache"
    
    def apply_teacache(self, model, rel_l1_thresh: float, start_percent: float, end_percent: float, 
                      cache_device: str, enable: bool, cache_mode: str):
        """Apply TeaCache to SDXL model."""
        
        if not enable or rel_l1_thresh == 0:
            return (model,)

        # Adjust threshold based on cache mode
        mode_adjustments = {
            "conservative": 1.5,  # Higher threshold = less caching
            "balanced": 1.0,      # Use threshold as-is
            "aggressive": 0.5     # Lower threshold = more caching
        }
        adjusted_thresh = rel_l1_thresh * mode_adjustments[cache_mode]

        # SDXL coefficients (conservative values for UNet architecture)
        sdxl_coefficients = [1.0, -0.3, 0.05]

        new_model = model.clone()
        
        # Set transformer options
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {}
            
        new_model.model_options["transformer_options"]["rel_l1_thresh"] = adjusted_thresh
        new_model.model_options["transformer_options"]["coefficients"] = sdxl_coefficients
        new_model.model_options["transformer_options"]["model_type"] = "sdxl"
        new_model.model_options["transformer_options"]["cache_device"] = (
            mm.get_torch_device() if cache_device == "cuda" else torch.device("cpu")
        )
        
        diffusion_model = new_model.get_model_object("diffusion_model")

        # Store original forward method and patch with teacache version
        if not hasattr(diffusion_model, '_teacache_original_forward'):
            diffusion_model._teacache_original_forward = diffusion_model.forward
        
        # Use patch.multiple like the original teacache
        context = patch.multiple(
            diffusion_model,
            forward=teacache_sdxl_unet_forward.__get__(diffusion_model, diffusion_model.__class__)
        )
        
        def unet_wrapper_function(model_function, kwargs):
            """Wrapper to manage TeaCache during sampling."""
            input_tensor = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            
            # Get sampling information
            sigmas = c.get("transformer_options", {}).get("sample_sigmas")
            if sigmas is not None:
                # Find current step
                matched_step_index = (sigmas == timestep[0]).nonzero()
                if len(matched_step_index) > 0:
                    current_step_index = matched_step_index.item()
                else:
                    current_step_index = 0
                    for i in range(len(sigmas) - 1):
                        if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                            current_step_index = i
                            break
                
                # Reset cache at start of sampling
                if current_step_index == 0:
                    # Clear SDXL cache attributes
                    for attr in ['teacache_accumulated_distance', 'teacache_previous_input', 
                                'teacache_previous_output', 'teacache_should_calc']:
                        if hasattr(diffusion_model, attr):
                            delattr(diffusion_model, attr)
                
                # Check if TeaCache should be active for this step
                current_percent = current_step_index / (len(sigmas) - 1)
                c["transformer_options"]["current_percent"] = current_percent
                
                if start_percent <= current_percent <= end_percent:
                    c["transformer_options"]["enable_teacache"] = True
                else:
                    c["transformer_options"]["enable_teacache"] = False
            else:
                c["transformer_options"]["enable_teacache"] = enable

            with context:
                return model_function(input_tensor, timestep, **c)

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)
        
        return (new_model,)


class DonutSDXLTeaCacheStats:
    """Statistics node for monitoring SDXL TeaCache performance."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "TeaCache-enabled SDXL model to get stats from."}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "stats")
    FUNCTION = "get_stats"
    CATEGORY = "DonutNodes"
    TITLE = "Donut SDXL TeaCache Stats"
    
    def get_stats(self, model):
        """Get TeaCache statistics."""
        diffusion_model = model.get_model_object("diffusion_model")
        
        stats_text = "SDXL TeaCache Statistics:\n"
        
        if hasattr(diffusion_model, 'teacache_accumulated_distance'):
            stats_text += f"\n- Accumulated Distance: {diffusion_model.teacache_accumulated_distance:.4f}\n"
            stats_text += f"- Should Calculate: {getattr(diffusion_model, 'teacache_should_calc', 'Unknown')}\n"
            stats_text += f"- Has Previous Input: {hasattr(diffusion_model, 'teacache_previous_input') and diffusion_model.teacache_previous_input is not None}\n"
            stats_text += f"- Has Cached Output: {hasattr(diffusion_model, 'teacache_previous_output') and diffusion_model.teacache_previous_output is not None}\n"
            
            # Show cache effectiveness
            if hasattr(diffusion_model, 'teacache_should_calc'):
                cache_status = "❌ CALCULATING" if diffusion_model.teacache_should_calc else "✅ USING CACHE"
                stats_text += f"- Cache Status: {cache_status}\n"
        else:
            stats_text += "\nNo TeaCache state found. Model may not have TeaCache applied or hasn't been run yet."
        
        return (model, stats_text)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "DonutSDXLTeaCacheBase": DonutSDXLTeaCache,
    "DonutSDXLTeaCacheBaseStats": DonutSDXLTeaCacheStats,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutSDXLTeaCacheBase": "Donut SDXL TeaCache (Base)",
    "DonutSDXLTeaCacheBaseStats": "Donut SDXL TeaCache (Base) Stats",
}