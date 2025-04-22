import torch
import copy

class DonutDetailer2:
    # Required property so that ComfyUI recognizes the node type.
    class_type = "MODEL"
    aux_id = "DonutsDelivery/ComfyUI-DonutDetailer"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                # Input block parameters:
                "Multiplier_in":    ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "S1_in":   ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "S2_in":   ("FLOAT", {"default": 2.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                # Output block 0 parameters:
                "Multiplier_out0":  ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "S1_out0": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "S2_out0": ("FLOAT", {"default": 2.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                # Output block 2 parameters:
                "Multiplier_out2":  ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "S1_out2": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "S2_out2": ("FLOAT", {"default": 2.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patch"
    CATEGORY = "Model Patches"

    def apply_patch(self, model, K_in, S1_in, S2_in, K_out0, S1_out0, S2_out0, K_out2, S1_out2, S2_out2):
        """
        Donut Detailer 2:
        Applies group-specific adjustments to key parameter groups in an SDXL model.

        The following formulas are applied per group:

          Weight multiplier = 1 - (K × S1 × 0.01)
          Bias multiplier   = 1 + (K × S2 × 0.02)

        Groups:
          1. Input Block (e.g. "input_blocks.0.0." or "diffusion_model.input_blocks.0.0."):
             - With defaults: K_in = 1, S1_in = 0, S2_in = 0 (no change)

          2. Output Block 0 (e.g. "out.0." or "diffusion_model.out.0."):
             - With defaults: K_out0 = 1, S1_out0 = 0, S2_out0 = 1

          3. Output Block 2 (e.g. "out.2." or "diffusion_model.out.2."):
             - With defaults: K_out2 = 1, S1_out2 = 0, S2_out2 = 1

        With these defaults, the node acts as a bypass.
        """
        # Clone the model so that only this branch is modified.
        new_model = copy.deepcopy(model)

        # Determine the underlying module to target.
        if hasattr(new_model, "named_parameters"):
            target_model = new_model
        elif hasattr(new_model, "unet"):
            target_model = new_model.unet
        elif hasattr(new_model, "model"):
            target_model = new_model.model
        else:
            target_model = new_model

        print("Target model type:", type(target_model))

        # Determine naming prefixes by inspecting the first parameter key.
        param_iter = target_model.named_parameters()
        try:
            first_key = next(param_iter)[0]
        except StopIteration:
            first_key = ""

        if first_key.startswith("diffusion_model."):
            prefix_in   = "diffusion_model.input_blocks.0.0."
            prefix_out0 = "diffusion_model.out.0."
            prefix_out2 = "diffusion_model.out.2."
        else:
            prefix_in   = "input_blocks.0.0."
            prefix_out0 = "out.0."
            prefix_out2 = "out.2."

        print("Using prefixes:")
        print("  Input block:", prefix_in)
        print("  Output block 0:", prefix_out0)
        print("  Output block 2:", prefix_out2)

        # Compute multipliers for each group.
        weight_in_mult   = 1 - (K_in   * S1_in   * 0.01)
        bias_in_mult     = 1 + (K_in   * S2_in   * 0.02)

        weight_out0_mult = 1 - (K_out0 * S1_out0 * 0.01)
        bias_out0_mult   = 1 + (K_out0 * S2_out0 * 0.02)

        weight_out2_mult = 1 - (K_out2 * S1_out2 * 0.01)
        bias_out2_mult   = 1 + (K_out2 * S2_out2 * 0.02)

        with torch.no_grad():
            for name, param in target_model.named_parameters():
                if name.startswith(prefix_in):
                    if "weight" in name:
                        param.data.mul_(weight_in_mult)
                        print(f"Patching {name}: weight × {weight_in_mult:.4f}")
                    elif "bias" in name:
                        param.data.mul_(bias_in_mult)
                        print(f"Patching {name}: bias × {bias_in_mult:.4f}")
                elif name.startswith(prefix_out0):
                    if "weight" in name:
                        param.data.mul_(weight_out0_mult)
                        print(f"Patching {name}: weight × {weight_out0_mult:.4f}")
                    elif "bias" in name:
                        param.data.mul_(bias_out0_mult)
                        print(f"Patching {name}: bias × {bias_out0_mult:.4f}")
                elif name.startswith(prefix_out2):
                    if "weight" in name:
                        param.data.mul_(weight_out2_mult)
                        print(f"Patching {name}: weight × {weight_out2_mult:.4f}")
                    elif "bias" in name:
                        param.data.mul_(bias_out2_mult)
                        print(f"Patching {name}: bias × {bias_out2_mult:.4f}")

        return (new_model,)

# Register the node with ComfyUI under the name "Donut Detailer 2".
NODE_CLASS_MAPPINGS = {
    "Donut Detailer 2": DonutDetailer2,
}
