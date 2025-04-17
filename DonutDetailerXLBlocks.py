import torch
import copy

class DonutDetailerXLBlocks:
    class_type = "MODEL"
    aux_id = "DonutsDelivery/ComfyUI-DonutDetailer"

    @classmethod
    def INPUT_TYPES(cls):
        input_dict = {"model": ("MODEL",)}
        for block_group in [
            ("input_blocks", 9),
            ("middle_block", 3),
            ("output_blocks", 9)
        ]:
            prefix, count = block_group
            for i in range(count):
                input_dict[f"{prefix}.{i}_weight"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001})
                input_dict[f"{prefix}.{i}_bias"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001})

        # Final output layer
        input_dict["out._weight"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001})
        input_dict["out._bias"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001})

        return {"required": input_dict}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patch"
    CATEGORY = "Model Patches"

    def apply_patch(self, model, **kwargs):
        new_model = copy.deepcopy(model)

        if hasattr(new_model, "named_parameters"):
            target_model = new_model
        elif hasattr(new_model, "unet"):
            target_model = new_model.unet
        elif hasattr(new_model, "model"):
            target_model = new_model.model
        else:
            target_model = new_model

        with torch.no_grad():
            for name, param in target_model.named_parameters():
                for prefix in [
                    "input_blocks.",
                    "middle_block.",
                    "output_blocks."
                ]:
                    for i in range(20):  # max range to catch anything extra
                        if name.startswith(f"diffusion_model.{prefix}{i}.") or name.startswith(f"{prefix}{i}."):
                            block_name = f"{prefix}{i}"
                            if "weight" in name:
                                param.data.mul_(kwargs.get(f"{block_name}_weight", 1.0))
                                print(f"Patching {name}: weight × {kwargs.get(f'{block_name}_weight', 1.0):.4f}")
                            elif "bias" in name:
                                param.data.mul_(kwargs.get(f"{block_name}_bias", 1.0))
                                print(f"Patching {name}: bias × {kwargs.get(f'{block_name}_bias', 1.0):.4f}")

                # Special case for final out layer
                if name.startswith("diffusion_model.out.") or name.startswith("out."):
                    if "weight" in name:
                        param.data.mul_(kwargs.get("out._weight", 1.0))
                        print(f"Patching {name}: weight × {kwargs.get('out._weight', 1.0):.4f}")
                    elif "bias" in name:
                        param.data.mul_(kwargs.get("out._bias", 1.0))
                        print(f"Patching {name}: bias × {kwargs.get('out._bias', 1.0):.4f}")

        return (new_model,)

NODE_CLASS_MAPPINGS = {
    "Donut Detailer XL Blocks": DonutDetailerXLBlocks,
}
