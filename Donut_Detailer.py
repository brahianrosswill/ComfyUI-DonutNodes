import torch
import copy

class DonutDetailer:
    # Required property so that ComfyUI recognizes the node type.
    class_type = "MODEL"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                # Input block parameters:
                "Scale_in":    ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.001}),
                "Weight_in":   ("FLOAT", {"default": 0.0, "min": -10.0,  "max": 10.0,  "step": 0.001}),
                "Bias_in":     ("FLOAT", {"default": 0.0, "min": -10.0,  "max": 10.0,  "step": 0.001}),
                # Output block 0 parameters:
                "Scale_out0":  ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.001}),
                "Weight_out0": ("FLOAT", {"default": 0.0, "min": -10.0,  "max": 10.0,  "step": 0.001}),
                "Bias_out0":   ("FLOAT", {"default": 1.0, "min": -10.0,  "max": 10.0,  "step": 0.001}),
                # Output block 2 parameters:
                "Scale_out2":  ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.001}),
                "Weight_out2": ("FLOAT", {"default": 0.0, "min": -10.0,  "max": 10.0,  "step": 0.001}),
                "Bias_out2":   ("FLOAT", {"default": 1.0, "min": -10.0,  "max": 10.0,  "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patch"
    CATEGORY = "Model Patches"

    def apply_patch(
        self, model,
        Scale_in, Weight_in, Bias_in,
        Scale_out0, Weight_out0, Bias_out0,
        Scale_out2, Weight_out2, Bias_out2
    ):
        """
        Donut Detailer:
        This node applies adjustments to three groups of parameters in an SDXL model.

        1. Input Block (e.g. "input_blocks.0.0." or "diffusion_model.input_blocks.0.0."):
           - Weight: multiplied by (1 - Scale_in × Weight_in)
           - Bias:   multiplied by (1 + Scale_in × Bias_in)

        2. Output Block 0 (e.g. "out.0." or "diffusion_model.out.0."):
           - Weight: multiplied by (1 - Scale_out0 × Weight_out0)
           - Bias:   multiplied by (Scale_out0 × Bias_out0)

        3. Output Block 2 (e.g. "out.2." or "diffusion_model.out.2."):
           - Weight: multiplied by (1 - Scale_out2 × Weight_out2)
           - Bias:   multiplied by (Scale_out2 × Bias_out2)

        With these default values (Scale_in=1, Weight_in=0, Bias_in=0; Scale_out=1, Weight_out=0, Bias_out=1),
        the node produces a bypass effect.
        """
        # Clone the model so that only this branch is modified.
        new_model = copy.deepcopy(model)

        # Determine the underlying model module.
        if hasattr(new_model, "named_parameters"):
            target_model = new_model
        elif hasattr(new_model, "unet"):
            target_model = new_model.unet
        elif hasattr(new_model, "model"):
            target_model = new_model.model
        else:
            target_model = new_model

        print("Target model type:", type(target_model))

        # Decide naming convention by checking the first parameter key.
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

        with torch.no_grad():
            for name, param in target_model.named_parameters():
                if name.startswith(prefix_in):
                    # Input block adjustments.
                    if "weight" in name:
                        param.data.mul_(1 - Scale_in * Weight_in)
                        print(f"Patching {name}: weight × (1 - {Scale_in}×{Weight_in})")
                    elif "bias" in name:
                        param.data.mul_(1 + Scale_in * Bias_in)
                        print(f"Patching {name}: bias × (1 + {Scale_in}×{Bias_in})")
                elif name.startswith(prefix_out0):
                    # Output block 0 adjustments.
                    if "weight" in name:
                        param.data.mul_(1 - Scale_out0 * Weight_out0)
                        print(f"Patching {name}: weight × (1 - {Scale_out0}×{Weight_out0})")
                    elif "bias" in name:
                        param.data.mul_(Scale_out0 * Bias_out0)
                        print(f"Patching {name}: bias × ({Scale_out0}×{Bias_out0})")
                elif name.startswith(prefix_out2):
                    # Output block 2 adjustments.
                    if "weight" in name:
                        param.data.mul_(1 - Scale_out2 * Weight_out2)
                        print(f"Patching {name}: weight × (1 - {Scale_out2}×{Weight_out2})")
                    elif "bias" in name:
                        param.data.mul_(Scale_out2 * Bias_out2)
                        print(f"Patching {name}: bias × ({Scale_out2}×{Bias_out2})")

        return (new_model,)

NODE_CLASS_MAPPINGS = {
    "Donut Detailer": DonutDetailer,
}
