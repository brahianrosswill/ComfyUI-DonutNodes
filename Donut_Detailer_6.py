import torch
import copy

class DonutDetailerLoRA6:
    # This node processes LoRA objects.
    class_type = "LORA"
    aux_id = "donutsdelivery/comfyui-donutdetailer"  # Must be in the format 'github-user/repo-name'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora": ("LORA",),
                # Sliders for the "down_blocks" (input) group:
                "Weight_down": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.001}),
                "Bias_down":   ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.001}),
                # Sliders for the "mid_block" group:
                "Weight_mid":  ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.001}),
                "Bias_mid":    ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.001}),
                # Sliders for the "up_blocks" (output) group:
                "Weight_up":   ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.001}),
                "Bias_up":     ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("LORA",)
    FUNCTION = "apply_patch"
    CATEGORY = "LoRA Patches"

    def apply_patch(self, lora, Weight_down, Bias_down, Weight_mid, Bias_mid, Weight_up, Bias_up):
        """
        Donut Detailer LoRA 6:
        Applies direct multipliers to a LoRA's parameters according to group-specific rules.

        Groups are determined by the following substrings in parameter names:
          - "down_blocks": treated as the input group.
          - "mid_block":    treated as the middle group.
          - "up_blocks":    treated as the output group.

        For each group, the parameter is multiplied by the corresponding slider:
          - For weights: multiply by the corresponding Weight_*.
          - For biases:  multiply by the corresponding Bias_*.

        With all sliders set to 1.0, the node produces no change (bypass).
        """
        # Clone the LoRA so that only this branch is modified.
        new_lora = copy.deepcopy(lora)

        # Get the named parameters from the LoRA.
        # LoRAs for SDXL typically follow Diffusers-style naming: e.g. "lora_unet_down_blocks_0_..."
        with torch.no_grad():
            for name, param in new_lora.named_parameters():
                lower_name = name.lower()
                if "down_blocks" in lower_name:
                    if "weight" in lower_name:
                        param.data.mul_(Weight_down)
                        print(f"Patching {name}: weight × {Weight_down:.4f}")
                    elif "bias" in lower_name:
                        param.data.mul_(Bias_down)
                        print(f"Patching {name}: bias × {Bias_down:.4f}")
                elif "mid_block" in lower_name:
                    if "weight" in lower_name:
                        param.data.mul_(Weight_mid)
                        print(f"Patching {name}: weight × {Weight_mid:.4f}")
                    elif "bias" in lower_name:
                        param.data.mul_(Bias_mid)
                        print(f"Patching {name}: bias × {Bias_mid:.4f}")
                elif "up_blocks" in lower_name:
                    if "weight" in lower_name:
                        param.data.mul_(Weight_up)
                        print(f"Patching {name}: weight × {Weight_up:.4f}")
                    elif "bias" in lower_name:
                        param.data.mul_(Bias_up)
                        print(f"Patching {name}: bias × {Bias_up:.4f}")

        return (new_lora,)

NODE_CLASS_MAPPINGS = {
    "Donut Detailer LoRA 6": DonutDetailerLoRA6,
}
