import torch
import copy

class DonutDetailer4:
    # Required property for ComfyUI.
    class_type = "MODEL"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                # Multipliers for Input Block:
                "Weight_in": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "Bias_in":   ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                # Multipliers for Output Block 0:
                "Weight_out0": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "Bias_out0":   ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                # Multipliers for Output Block 2:
                "Weight_out2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "Bias_out2":   ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patch"
    CATEGORY = "Model Patches"

    def apply_patch(self, model, Weight_in, Bias_in, Weight_out0, Bias_out0, Weight_out2, Bias_out2):
        """
        Donut Detailer 4:
        This node applies direct multipliers to three groups of parameters in an SDXL model.

        For each group, the node multiplies the parameter values by the corresponding slider:

          1. Input Block (e.g. "input_blocks.0.0." or "diffusion_model.input_blocks.0.0."):
             - weight *= Weight_in
             - bias   *= Bias_in

          2. Output Block 0 (e.g. "out.0." or "diffusion_model.out.0."):
             - weight *= Weight_out0
             - bias   *= Bias_out0

          3. Output Block 2 (e.g. "out.2." or "diffusion_model.out.2."):
             - weight *= Weight_out2
             - bias   *= Bias_out2

        With the default values (all 1.0), the node acts as a bypass.
        """
        # Clone the model so that only the branch passing through this node is modified.
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

        with torch.no_grad():
            for name, param in target_model.named_parameters():
                if name.startswith(prefix_in):
                    if "weight" in name:
                        param.data.mul_(Weight_in)
                        print(f"Patching {name}: weight × {Weight_in:.4f}")
                    elif "bias" in name:
                        param.data.mul_(Bias_in)
                        print(f"Patching {name}: bias × {Bias_in:.4f}")
                elif name.startswith(prefix_out0):
                    if "weight" in name:
                        param.data.mul_(Weight_out0)
                        print(f"Patching {name}: weight × {Weight_out0:.4f}")
                    elif "bias" in name:
                        param.data.mul_(Bias_out0)
                        print(f"Patching {name}: bias × {Bias_out0:.4f}")
                elif name.startswith(prefix_out2):
                    if "weight" in name:
                        param.data.mul_(Weight_out2)
                        print(f"Patching {name}: weight × {Weight_out2:.4f}")
                    elif "bias" in name:
                        param.data.mul_(Bias_out2)
                        print(f"Patching {name}: bias × {Bias_out2:.4f}")

        return (new_model,)

# Register the node with ComfyUI under the name "Donut Detailer 4".
NODE_CLASS_MAPPINGS = {
    "Donut Detailer 4": DonutDetailer4,
}
