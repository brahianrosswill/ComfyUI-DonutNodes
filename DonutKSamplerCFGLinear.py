#!/usr/bin/env python3
"""
DonutSampler - Custom KSampler with Linear CFG Progression

A custom KSampler that allows defining start and end CFG values with linear
interpolation between them throughout the sampling steps.

This implementation creates a custom sampling loop with dynamic CFG values per step.
"""

import torch
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.k_diffusion.sampling as k_diffusion_sampling
import latent_preview


class DonutSampler:
    """
    Custom KSampler with linear CFG progression from start to end values.
    
    Implements dynamic CFG by creating a custom sampling function that interpolates
    CFG values linearly from start to end across the sampling steps.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg_start": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_halfway": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "halfway_step": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "cfg_progression_info")
    FUNCTION = "sample_with_linear_cfg"
    CATEGORY = "donut/sampling"

    def __init__(self):
        self.cfg_history = []

    def calculate_cfg_for_step(self, step, total_steps, cfg_start, cfg_halfway, cfg_end, halfway_step):
        """Calculate the CFG value for a specific step with optional halfway point."""
        if total_steps <= 1:
            return cfg_start
        
        # Check if halfway CFG is disabled (same as start or end)
        halfway_disabled = (cfg_halfway == cfg_start or cfg_halfway == cfg_end)
        
        if halfway_disabled:
            # Simple linear interpolation between start and end
            progress = step / (total_steps - 1)
            current_cfg = cfg_start + (cfg_end - cfg_start) * progress
        else:
            # Clamp halfway_step to valid range
            halfway_step = max(1, min(halfway_step, total_steps - 1))
            
            if step <= halfway_step:
                # First segment: start -> halfway
                if halfway_step == 0:
                    current_cfg = cfg_start
                else:
                    progress = step / halfway_step
                    current_cfg = cfg_start + (cfg_halfway - cfg_start) * progress
            else:
                # Second segment: halfway -> end
                remaining_steps = total_steps - 1 - halfway_step
                if remaining_steps == 0:
                    current_cfg = cfg_halfway
                else:
                    progress = (step - halfway_step) / remaining_steps
                    current_cfg = cfg_halfway + (cfg_end - cfg_halfway) * progress
        
        return current_cfg

    def sample_with_linear_cfg(self, model, seed, steps, cfg_start, cfg_halfway, cfg_end, halfway_step, sampler_name, 
                              scheduler, positive, negative, latent_image, denoise):
        """
        Basic sampling using ComfyUI's common_ksampler with dynamic CFG injection.
        """
        try:
            # Pre-calculate CFG values for each step
            self.cfg_history = []
            cfg_values = []
            for i in range(steps):
                cfg_val = self.calculate_cfg_for_step(i, steps, cfg_start, cfg_halfway, cfg_end, halfway_step)
                cfg_values.append(cfg_val)
                self.cfg_history.append((i, cfg_val))
            
            # Check if halfway is enabled
            halfway_disabled = (cfg_halfway == cfg_start or cfg_halfway == cfg_end)
            if halfway_disabled:
                print(f"DonutSampler: CFG progression {cfg_start:.2f} -> {cfg_end:.2f} over {steps} steps (halfway disabled)")
            else:
                clamped_halfway_step = max(1, min(halfway_step, steps - 1))
                print(f"DonutSampler: CFG progression {cfg_start:.2f} -> {cfg_halfway:.2f} (step {clamped_halfway_step}) -> {cfg_end:.2f} over {steps} steps")
            
            # Show CFG values for debugging
            if len(cfg_values) >= 10:
                print(f"CFG linear progression preview:")
                indices = [0, len(cfg_values)//4, len(cfg_values)//2, 3*len(cfg_values)//4, len(cfg_values)-1]
                for i in indices:
                    if i < len(cfg_values):
                        progress = i / (len(cfg_values) - 1) if len(cfg_values) > 1 else 0
                        print(f"  Step {i}: progress={progress:.2f} -> CFG={cfg_values[i]:.2f}")
            else:
                print(f"CFG values: {[f'{v:.2f}' for v in cfg_values]}")
            
            # Create a custom CFGGuider that handles dynamic CFG
            import comfy.samplers
            
            class DynamicCFGGuider(comfy.samplers.CFGGuider):
                def __init__(self, model_patcher, cfg_values):
                    super().__init__(model_patcher)
                    self.cfg_values = cfg_values
                    self.step_count = 0
                
                def predict_noise(self, x, timestep, model_options={}, seed=None):
                    # Get dynamic CFG for current step
                    current_cfg = self.cfg_values[min(self.step_count, len(self.cfg_values) - 1)]
                    print(f"DonutSampler: Step {self.step_count}, CFG {current_cfg:.2f}")
                    
                    # Temporarily set CFG and call parent
                    old_cfg = self.cfg
                    self.cfg = current_cfg
                    result = super().predict_noise(x, timestep, model_options, seed)
                    self.cfg = old_cfg
                    
                    self.step_count += 1
                    return result
            
            # Monkey patch CFGGuider creation
            original_cfg_guider_class = comfy.samplers.CFGGuider
            comfy.samplers.CFGGuider = lambda model_patcher: DynamicCFGGuider(model_patcher, cfg_values)
            
            try:
                # Use ComfyUI's standard common_ksampler
                from nodes import common_ksampler
                result = common_ksampler(model, seed, steps, cfg_start, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
                
            finally:
                # Restore original CFGGuider
                comfy.samplers.CFGGuider = original_cfg_guider_class
            
            # Create CFG progression info
            cfg_info = self.format_cfg_info(cfg_start, cfg_halfway, cfg_end, halfway_step, steps, sampler_name, scheduler)
            
            return (result[0], cfg_info)
            
        except Exception as e:
            import traceback
            error_msg = f"Sampling failed: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return (latent_image, f"ERROR: {error_msg}")
    
    def create_ascii_chart(self, cfg_values, cfg_start, cfg_end, width=50, height=12):
        """Create an ASCII art chart of the CFG progression."""
        if not cfg_values:
            return "No CFG data available"
        
        # Create the chart grid
        chart = []
        
        # Calculate scaling
        cfg_min = min(cfg_values)
        cfg_max = max(cfg_values)
        cfg_range = cfg_max - cfg_min if cfg_max != cfg_min else 1
        
        # Create chart lines from top to bottom
        for row in range(height):
            line = []
            # Y-axis labels (CFG values)
            y_value = cfg_max - (row / (height - 1)) * cfg_range
            line.append(f"{y_value:4.1f}│")
            
            # Plot the curve
            for col in range(width):
                step_index = int((col / (width - 1)) * (len(cfg_values) - 1))
                cfg_val = cfg_values[step_index]
                
                # Check if this point should be plotted
                expected_y = cfg_max - (row / (height - 1)) * cfg_range
                tolerance = cfg_range / (height * 2)
                
                if abs(cfg_val - expected_y) <= tolerance:
                    line.append("●")
                elif row == height - 1:  # Bottom line
                    line.append("─")
                elif col == 0:  # Left edge
                    line.append("│")
                else:
                    line.append(" ")
            
            chart.append("".join(line))
        
        # Add bottom axis
        bottom_line = "    └" + "─" * width
        chart.append(bottom_line)
        
        # Add step labels
        step_labels = "     "
        for i in range(0, width, width//5):
            step_num = int((i / (width - 1)) * (len(cfg_values) - 1))
            step_labels += f"{step_num:2d}" + " " * (width//5 - 2)
        chart.append(step_labels[:len(bottom_line)])
        chart.append("     Steps →")
        
        return "\n".join(chart)

    def format_cfg_info(self, cfg_start, cfg_halfway, cfg_end, halfway_step, steps, sampler_name, scheduler):
        """Format the CFG progression information."""
        halfway_disabled = (cfg_halfway == cfg_start or cfg_halfway == cfg_end)
        
        if halfway_disabled:
            cfg_info = f"DonutSampler - Linear CFG Progression\n"
        else:
            cfg_info = f"DonutSampler - Three-Point CFG Progression\n"
        
        cfg_info += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        cfg_info += f"Start CFG: {cfg_start:.2f}\n"
        
        if not halfway_disabled:
            clamped_halfway_step = max(1, min(halfway_step, steps - 1))
            cfg_info += f"Halfway CFG: {cfg_halfway:.2f} (at step {clamped_halfway_step})\n"
        
        cfg_info += f"End CFG: {cfg_end:.2f}\n"
        cfg_info += f"Steps: {steps}\n"
        cfg_info += f"Sampler: {sampler_name}\n"
        cfg_info += f"Scheduler: {scheduler}\n"
        
        # Add ASCII chart
        if self.cfg_history:
            cfg_values = [cfg_val for _, cfg_val in self.cfg_history]
            cfg_info += f"\nCFG Progression Chart:\n"
            cfg_info += self.create_ascii_chart(cfg_values, cfg_start, cfg_end)
            cfg_info += f"\n"
            
            # Show key numerical values
            cfg_info += f"\nKey Steps:\n"
            if len(self.cfg_history) > 8:
                shown = self.cfg_history[:4] + [(-1, "...")] + self.cfg_history[-4:]
            else:
                shown = self.cfg_history
            
            for step, cfg_val in shown:
                if step == -1:
                    cfg_info += f"  ...\n"
                else:
                    cfg_info += f"  Step {step+1}: CFG={cfg_val:.2f}\n"
        
        return cfg_info




class DonutKSamplerAdvanced:
    """
    Advanced DonutSampler based on KSamplerAdvanced with linear CFG progression.
    
    Includes all KSamplerAdvanced features:
    - add_noise control
    - start_at_step/end_at_step for partial sampling
    - return_with_leftover_noise option
    - Linear CFG progression from cfg_start to cfg_end
    - Easing curves for CFG progression
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"],),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg_start": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_halfway": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "halfway_step": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
                "cfg_curve": (["linear", "exponential", "logarithmic", "ease_in", "ease_out", "ease_in_out", "sine_wave", "cosine_wave", "smooth_step", "smoother_step", "circular_in", "circular_out", "back_in", "back_out", "elastic_in", "elastic_out", "bounce_in", "bounce_out", "dramatic_exponential", "dramatic_logarithmic"], {"default": "linear"}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "cfg_progression_info")
    FUNCTION = "sample_advanced"
    CATEGORY = "donut/sampling"

    def __init__(self):
        self.cfg_history = []

    def apply_curve(self, progress, curve_type):
        """Apply different mathematical curves to the progression."""
        import math
        
        # Clamp progress to [0, 1]
        progress = max(0.0, min(1.0, progress))
        
        if curve_type == "linear":
            return progress
        
        # Basic easing curves
        elif curve_type == "ease_in":
            return progress * progress
        elif curve_type == "ease_out":
            return 1 - (1 - progress) * (1 - progress)
        elif curve_type == "ease_in_out":
            if progress < 0.5:
                return 2 * progress * progress
            else:
                return 1 - 2 * (1 - progress) * (1 - progress)
        
        # Mathematical function curves
        elif curve_type == "exponential":
            # Exponential curve: starts slow, accelerates rapidly
            if progress == 0:
                return 0
            elif progress == 1:
                return 1
            else:
                # Use a gentler exponential curve
                return (math.exp(3 * progress) - 1) / (math.exp(3) - 1)
        
        elif curve_type == "logarithmic":
            # Logarithmic curve: starts fast, decelerates
            if progress == 0:
                return 0
            elif progress == 1:
                return 1
            else:
                # Use a gentler logarithmic curve
                return math.log(1 + 2 * progress) / math.log(3)
        
        elif curve_type == "sine_wave":
            # Sine wave: smooth S-curve
            return 0.5 * (1 - math.cos(progress * math.pi))
        
        elif curve_type == "cosine_wave":
            # Cosine wave: inverted sine
            return 0.5 * (1 + math.cos((1 - progress) * math.pi))
        
        elif curve_type == "smooth_step":
            # Smooth step function: 3t²-2t³
            return progress * progress * (3 - 2 * progress)
        
        elif curve_type == "smoother_step":
            # Even smoother step: 6t⁵-15t⁴+10t³
            return progress * progress * progress * (progress * (progress * 6 - 15) + 10)
        
        # Circular curves
        elif curve_type == "circular_in":
            return 1 - math.sqrt(1 - progress * progress)
        
        elif curve_type == "circular_out":
            return math.sqrt(1 - (progress - 1) * (progress - 1))
        
        # Back curves (overshoot)
        elif curve_type == "back_in":
            c1 = 1.70158
            c3 = c1 + 1
            return c3 * progress * progress * progress - c1 * progress * progress
        
        elif curve_type == "back_out":
            c1 = 1.70158
            c3 = c1 + 1
            return 1 + c3 * math.pow(progress - 1, 3) + c1 * math.pow(progress - 1, 2)
        
        # Elastic curves (spring-like)
        elif curve_type == "elastic_in":
            c4 = (2 * math.pi) / 3
            if progress == 0:
                return 0
            elif progress == 1:
                return 1
            else:
                return -math.pow(2, 10 * progress - 10) * math.sin((progress * 10 - 10.75) * c4)
        
        elif curve_type == "elastic_out":
            c4 = (2 * math.pi) / 3
            if progress == 0:
                return 0
            elif progress == 1:
                return 1
            else:
                return math.pow(2, -10 * progress) * math.sin((progress * 10 - 0.75) * c4) + 1
        
        # Bounce curves
        elif curve_type == "bounce_in":
            return 1 - self.bounce_out_helper(1 - progress)
        
        elif curve_type == "bounce_out":
            return self.bounce_out_helper(progress)
        
        # Dramatic curves for more visible differences
        elif curve_type == "dramatic_exponential":
            # Very steep exponential - stays high for most of the time
            if progress == 0:
                return 0
            elif progress == 1:
                return 1
            else:
                return math.pow(progress, 4)  # x^4 curve
        
        elif curve_type == "dramatic_logarithmic":
            # Very steep logarithmic - drops quickly then levels off
            if progress == 0:
                return 0
            elif progress == 1:
                return 1
            else:
                return math.pow(progress, 0.25)  # x^0.25 curve (4th root)
        
        # Default fallback
        return progress
    
    def bounce_out_helper(self, x):
        """Helper function for bounce curves."""
        n1 = 7.5625
        d1 = 2.75
        
        if x < 1 / d1:
            return n1 * x * x
        elif x < 2 / d1:
            return n1 * (x - 1.5 / d1) * (x - 1.5 / d1) + 0.75
        elif x < 2.5 / d1:
            return n1 * (x - 2.25 / d1) * (x - 2.25 / d1) + 0.9375
        else:
            return n1 * (x - 2.625 / d1) * (x - 2.625 / d1) + 0.984375

    def calculate_cfg_for_step(self, step, total_steps, cfg_start, cfg_halfway, cfg_end, halfway_step, curve_type="linear"):
        """Calculate the CFG value for a specific step with optional halfway point and curve."""
        if total_steps <= 1:
            return cfg_start
        
        # Check if halfway CFG is disabled (same as start or end)
        halfway_disabled = (cfg_halfway == cfg_start or cfg_halfway == cfg_end)
        
        if halfway_disabled:
            # Simple curved interpolation between start and end
            progress = step / (total_steps - 1)
            curved_progress = self.apply_curve(progress, curve_type)
            current_cfg = cfg_start + (cfg_end - cfg_start) * curved_progress
        else:
            # Clamp halfway_step to valid range
            halfway_step = max(1, min(halfway_step, total_steps - 1))
            
            if step <= halfway_step:
                # First segment: start -> halfway with curve
                if halfway_step == 0:
                    current_cfg = cfg_start
                else:
                    progress = step / halfway_step
                    curved_progress = self.apply_curve(progress, curve_type)
                    current_cfg = cfg_start + (cfg_halfway - cfg_start) * curved_progress
            else:
                # Second segment: halfway -> end with curve
                remaining_steps = total_steps - 1 - halfway_step
                if remaining_steps == 0:
                    current_cfg = cfg_halfway
                else:
                    progress = (step - halfway_step) / remaining_steps
                    curved_progress = self.apply_curve(progress, curve_type)
                    current_cfg = cfg_halfway + (cfg_end - cfg_halfway) * curved_progress
        
        # Debug: Print curve calculations for first few steps
        if step < 3 or step >= total_steps - 2:
            if halfway_disabled:
                progress = step / (total_steps - 1)
                curved_progress = self.apply_curve(progress, curve_type)
                print(f"Step {step}: progress={progress:.3f} -> curved={curved_progress:.3f} -> CFG={current_cfg:.3f} (curve: {curve_type})")
            else:
                segment = "first" if step <= halfway_step else "second"
                print(f"Step {step}: {segment} segment -> CFG={current_cfg:.3f} (curve: {curve_type})")
        
        return current_cfg

    def common_ksampler_with_dynamic_cfg(self, model, seed, steps, cfg_start, cfg_halfway, cfg_end, halfway_step, sampler_name, scheduler, 
                                        positive, negative, latent, denoise=1.0, disable_noise=False, 
                                        start_step=None, last_step=None, force_full_denoise=False, curve_type="linear"):
        """
        Advanced sampling with dynamic CFG using CFGGuider approach.
        """
        # Pre-calculate CFG values for each step
        self.cfg_history = []
        cfg_values = []
        for i in range(steps):
            cfg_val = self.calculate_cfg_for_step(i, steps, cfg_start, cfg_halfway, cfg_end, halfway_step, curve_type)
            cfg_values.append(cfg_val)
            self.cfg_history.append((i, cfg_val))
        
        # Check if halfway is enabled
        halfway_disabled = (cfg_halfway == cfg_start or cfg_halfway == cfg_end)
        if halfway_disabled:
            print(f"DonutSampler Advanced: CFG {curve_type} progression {cfg_start:.2f} -> {cfg_end:.2f} over {steps} steps (halfway disabled)")
        else:
            clamped_halfway_step = max(1, min(halfway_step, steps - 1))
            print(f"DonutSampler Advanced: CFG {curve_type} progression {cfg_start:.2f} -> {cfg_halfway:.2f} (step {clamped_halfway_step}) -> {cfg_end:.2f} over {steps} steps")
        
        # Show curve comparison for debugging
        if len(cfg_values) >= 10:
            print(f"CFG progression preview ({curve_type}):")
            indices = [0, len(cfg_values)//4, len(cfg_values)//2, 3*len(cfg_values)//4, len(cfg_values)-1]
            for i in indices:
                if i < len(cfg_values):
                    progress = i / (len(cfg_values) - 1) if len(cfg_values) > 1 else 0
                    curved = self.apply_curve(progress, curve_type)
                    print(f"  Step {i}: progress={progress:.2f} -> curved={curved:.2f} -> CFG={cfg_values[i]:.2f}")
        else:
            print(f"CFG values: {[f'{v:.2f}' for v in cfg_values]}")
        
        # Create a custom CFGGuider that handles dynamic CFG
        import comfy.samplers
        
        class DynamicCFGGuiderAdvanced(comfy.samplers.CFGGuider):
            def __init__(self, model_patcher, cfg_values, curve_type):
                super().__init__(model_patcher)
                self.cfg_values = cfg_values
                self.curve_type = curve_type
                self.step_count = 0
            
            def predict_noise(self, x, timestep, model_options={}, seed=None):
                # Get dynamic CFG for current step
                current_cfg = self.cfg_values[min(self.step_count, len(self.cfg_values) - 1)]
                print(f"DonutSampler Advanced: Step {self.step_count}, CFG {current_cfg:.2f} ({self.curve_type})")
                
                # Temporarily set CFG and call parent
                old_cfg = self.cfg
                self.cfg = current_cfg
                result = super().predict_noise(x, timestep, model_options, seed)
                self.cfg = old_cfg
                
                self.step_count += 1
                return result
        
        # Monkey patch CFGGuider creation
        original_cfg_guider_class = comfy.samplers.CFGGuider
        comfy.samplers.CFGGuider = lambda model_patcher: DynamicCFGGuiderAdvanced(model_patcher, cfg_values, curve_type)
        
        try:
            # Use ComfyUI's standard common_ksampler
            from nodes import common_ksampler
            result = common_ksampler(model, seed, steps, cfg_start, sampler_name, scheduler, positive, negative, latent, 
                                   denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step, 
                                   force_full_denoise=force_full_denoise)
            
        finally:
            # Restore original CFGGuider
            comfy.samplers.CFGGuider = original_cfg_guider_class
        
        return result

    def sample_advanced(self, model, add_noise, noise_seed, steps, cfg_start, cfg_halfway, cfg_end, halfway_step, sampler_name, 
                       scheduler, positive, negative, latent_image, start_at_step, end_at_step, 
                       return_with_leftover_noise, cfg_curve="linear", denoise=1.0):
        """
        Main advanced sampling function with linear CFG progression.
        Based on KSamplerAdvanced.sample() but with dynamic CFG.
        """
        try:
            # Process advanced parameters (from KSamplerAdvanced.sample)
            force_full_denoise = True
            if return_with_leftover_noise == "enable":
                force_full_denoise = False
            
            disable_noise = False
            if add_noise == "disable":
                disable_noise = True

            # Use our modified common_ksampler with dynamic CFG
            result = self.common_ksampler_with_dynamic_cfg(
                model, noise_seed, steps, cfg_start, cfg_halfway, cfg_end, halfway_step, sampler_name, scheduler, 
                positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, 
                start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise,
                curve_type=cfg_curve
            )

            # Create CFG progression info
            cfg_info = self.format_advanced_cfg_info(cfg_start, cfg_halfway, cfg_end, halfway_step, steps, sampler_name, scheduler, 
                                                   start_at_step, end_at_step, add_noise, return_with_leftover_noise, cfg_curve)

            return (result[0], cfg_info)
            
        except Exception as e:
            import traceback
            error_msg = f"Advanced sampling failed: {str(e)}\n{traceback.format_exc()}"
            return (latent_image, f"ERROR: {error_msg}")

    def create_ascii_chart(self, cfg_values, cfg_start, cfg_end, width=50, height=12):
        """Create an ASCII art chart of the CFG progression."""
        if not cfg_values:
            return "No CFG data available"
        
        # Create the chart grid
        chart = []
        
        # Calculate scaling
        cfg_min = min(cfg_values)
        cfg_max = max(cfg_values)
        cfg_range = cfg_max - cfg_min if cfg_max != cfg_min else 1
        
        # Create chart lines from top to bottom
        for row in range(height):
            line = []
            # Y-axis labels (CFG values)
            y_value = cfg_max - (row / (height - 1)) * cfg_range
            line.append(f"{y_value:4.1f}│")
            
            # Plot the curve
            for col in range(width):
                step_index = int((col / (width - 1)) * (len(cfg_values) - 1))
                cfg_val = cfg_values[step_index]
                
                # Check if this point should be plotted
                expected_y = cfg_max - (row / (height - 1)) * cfg_range
                tolerance = cfg_range / (height * 2)
                
                if abs(cfg_val - expected_y) <= tolerance:
                    line.append("●")
                elif row == height - 1:  # Bottom line
                    line.append("─")
                elif col == 0:  # Left edge
                    line.append("│")
                else:
                    line.append(" ")
            
            chart.append("".join(line))
        
        # Add bottom axis
        bottom_line = "    └" + "─" * width
        chart.append(bottom_line)
        
        # Add step labels
        step_labels = "     "
        for i in range(0, width, width//5):
            step_num = int((i / (width - 1)) * (len(cfg_values) - 1))
            step_labels += f"{step_num:2d}" + " " * (width//5 - 2)
        chart.append(step_labels[:len(bottom_line)])
        chart.append("     Steps →")
        
        return "\n".join(chart)

    def format_advanced_cfg_info(self, cfg_start, cfg_halfway, cfg_end, halfway_step, steps, sampler_name, scheduler, 
                                start_at_step, end_at_step, add_noise, return_with_leftover_noise, cfg_curve="linear"):
        """Format the advanced CFG progression information."""
        halfway_disabled = (cfg_halfway == cfg_start or cfg_halfway == cfg_end)
        
        if halfway_disabled:
            cfg_info = f"DonutSampler Advanced - {cfg_curve.title()} CFG Progression\n"
        else:
            cfg_info = f"DonutSampler Advanced - Three-Point {cfg_curve.title()} CFG Progression\n"
        
        cfg_info += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        if halfway_disabled:
            cfg_info += f"CFG Range: {cfg_start:.2f} → {cfg_end:.2f} | Curve: {cfg_curve}\n"
        else:
            clamped_halfway_step = max(1, min(halfway_step, steps - 1))
            cfg_info += f"CFG Range: {cfg_start:.2f} → {cfg_halfway:.2f} (step {clamped_halfway_step}) → {cfg_end:.2f} | Curve: {cfg_curve}\n"
        
        cfg_info += f"Steps: {steps} | Sampler: {sampler_name} | Scheduler: {scheduler}\n"
        cfg_info += f"Step Range: {start_at_step} to {end_at_step}\n"
        cfg_info += f"Add Noise: {add_noise} | Return w/ Leftover Noise: {return_with_leftover_noise}\n"
        
        # Add ASCII chart
        if self.cfg_history:
            cfg_values = [cfg_val for _, cfg_val in self.cfg_history]
            cfg_info += f"\nCFG Progression Chart ({cfg_curve}):\n"
            cfg_info += self.create_ascii_chart(cfg_values, cfg_start, cfg_end)
            cfg_info += f"\n"
            
            # Show key numerical values
            cfg_info += f"\nKey Steps:\n"
            if len(self.cfg_history) > 8:
                shown = self.cfg_history[:4] + [(-1, "...")] + self.cfg_history[-4:]
            else:
                shown = self.cfg_history
            
            for step, cfg_val in shown:
                if step == -1:
                    cfg_info += f"  ...\n"
                else:
                    actual_step = start_at_step + step
                    cfg_info += f"  Step {actual_step+1}: CFG={cfg_val:.2f}\n"
        
        return cfg_info


NODE_CLASS_MAPPINGS = {
    "DonutSampler": DonutSampler,
    "DonutSampler (Advanced)": DonutKSamplerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutSampler": "DonutSampler",
    "DonutSampler (Advanced)": "DonutSampler (Advanced)",
}