import torch
import copy

class DonutDetailerXLBlocks:
    class_type = "MODEL"
    # Make sure aux_id matches your GitHub repo slug
    aux_id = "DonutsDelivery/ComfyUI-DonutDetailer"

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model": ("MODEL",),
        }
        # Define block groups: (prefix, count)
        groups = [
            ("input_blocks", 9),
            ("middle_block", 3),
            ("output_blocks", 9),
            ("out", 1),
        ]
        for prefix, count in groups:
            for i in range(count):
                # for the lone "out" layer, use prefix="out" and i=0 once
                name = prefix if prefix == "out" else f"{prefix}_{i}"
                required[f"{name}_weight"] = ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001
                })
                required[f"{name}_bias"] = ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001
                })
        return {"required": required}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patch"
    CATEGORY = "Model Patches"

    def apply_patch(self, model, **kwargs):
        # clone so only this branch is modified
        new_model = copy.deepcopy(model)

        # unwrap to the nn.Module that has named_parameters()
        if hasattr(new_model, "named_parameters"):
            target = new_model
        elif hasattr(new_model, "unet"):
            target = new_model.unet
        elif hasattr(new_model, "model"):
            target = new_model.model
        else:
            target = new_model

        # detect whether names start with "diffusion_model."
        names = target.named_parameters()
        try:
            first_key = next(names)[0]
        except StopIteration:
            first_key = ""
        dm_pref = "diffusion_model." if first_key.startswith("diffusion_model.") else ""

        with torch.no_grad():
            for name, param in target.named_parameters():
                # input_blocks.0 – input_blocks.8
                for i in range(9):
                    pfx = f"{dm_pref}input_blocks.{i}."
                    key = f"input_blocks_{i}"
                    if name.startswith(pfx):
                        if name.endswith(".weight"):
                            param.data.mul_(kwargs[f"{key}_weight"])
                        elif name.endswith(".bias"):
                            param.data.mul_(kwargs[f"{key}_bias"])

                # middle_block.0 – middle_block.2
                for i in range(3):
                    pfx = f"{dm_pref}middle_block.{i}."
                    key = f"middle_block_{i}"
                    if name.startswith(pfx):
                        if name.endswith(".weight"):
                            param.data.mul_(kwargs[f"{key}_weight"])
                        elif name.endswith(".bias"):
                            param.data.mul_(kwargs[f"{key}_bias"])

                # output_blocks.0 – output_blocks.8
                for i in range(9):
                    pfx = f"{dm_pref}output_blocks.{i}."
                    key = f"output_blocks_{i}"
                    if name.startswith(pfx):
                        if name.endswith(".weight"):
                            param.data.mul_(kwargs[f"{key}_weight"])
                        elif name.endswith(".bias"):
                            param.data.mul_(kwargs[f"{key}_bias"])

                # the final out.* layer
                out_pfx = f"{dm_pref}out."
                if name.startswith(out_pfx):
                    key = "out"
                    if name.endswith(".weight"):
                        param.data.mul_(kwargs[f"{key}_weight"])
                    elif name.endswith(".bias"):
                        param.data.mul_(kwargs[f"{key}_bias"])

        return (new_model,)


NODE_CLASS_MAPPINGS = {
    "Donut Detailer XL Blocks": DonutDetailerXLBlocks,
}
