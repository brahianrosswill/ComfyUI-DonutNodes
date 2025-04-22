import torch
import copy

class DonutDetailerLoRA5:
    # Must be uppercase “LoRA”
    class_type = "LoRA"
    aux_id = "DonutsDelivery/ComfyUI-DonutDetailer"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Must be (“LoRA”,)
                "lora": ("LoRA",),
                "Weight_down": ("FLOAT", {"default": 1.1, "min": -10.0, "max": 10.0, "step": 0.001}),
                "Bias_down":   ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "Weight_mid":  ("FLOAT", {"default": 1.1, "min": -10.0, "max": 10.0, "step": 0.001}),
                "Bias_mid":    ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "Weight_up":   ("FLOAT", {"default": 1.1, "min": -10.0, "max": 10.0, "step": 0.001}),
                "Bias_up":     ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
            }
        }

    # Must be a one‑element tuple
    RETURN_TYPES = ("LoRA",)
    FUNCTION = "apply_patch"
    CATEGORY = "LoRA Patches"

    def apply_patch(self, lora, Weight_down, Bias_down, Weight_mid, Bias_mid, Weight_up, Bias_up):
        # lora is a dict with keys "lora", "strength_model", "strength_clip", etc.
        if not (isinstance(lora, dict) and "lora" in lora):
            raise TypeError(f"DonutDetailerLoRA6 expected a LoRA dict, got {type(lora)}")

        new_lora = copy.deepcopy(lora)

        with torch.no_grad():
            # The actual weights live under new_lora["lora"]
            for name, param in new_lora["lora"].items():
                if "down" in name:
                    if "weight" in name:
                        new_lora["lora"][name] = param * Weight_down
                    elif "bias" in name:
                        new_lora["lora"][name] = param * Bias_down
                elif "mid" in name:
                    if "weight" in name:
                        new_lora["lora"][name] = param * Weight_mid
                    elif "bias" in name:
                        new_lora["lora"][name] = param * Bias_mid
                elif "up" in name:
                    if "weight" in name:
                        new_lora["lora"][name] = param * Weight_up
                    elif "bias" in name:
                        new_lora["lora"][name] = param * Bias_up

        # return a 1‑tuple containing our patched LoRA dict
        return (new_lora,)

# Register only this node in this file:
NODE_CLASS_MAPPINGS = {
    "Donut Detailer LoRA 5": DonutDetailerLoRA5,
}
