import torch
from .merging_methods import MergingMethod

class DonutWidenMerge:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "base_model":                ("MODEL",),
            "model_to_merge":            ("MODEL",),
            "exclude_param_names_regex": ("STRING", {"default": ""}),
            "above_average_value_ratio": ("FLOAT",  {"default": 1.0, "min":0.0, "max":10.0, "step":0.01}),
            "score_calibration_value":   ("FLOAT",  {"default": 1.0, "min":0.0, "max":10.0, "step":0.01}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION     = "execute"
    CATEGORY     = "merging"

    def execute(self,
                base_model,
                model_to_merge,
                exclude_param_names_regex,
                above_average_value_ratio,
                score_calibration_value):

        # unwrap ComfyUI model wrappers
        def _unwrap(wrapper):
            return getattr(wrapper, "model", wrapper)

        merged_model  = _unwrap(base_model)
        finetuned_mod = _unwrap(model_to_merge)

        # make sure both modules live on the same device
        device = next(merged_model.parameters()).device
        finetuned_mod.to(device)

        # split comma-list of regexes into a Python list
        regex_list = [r.strip() for r in exclude_param_names_regex.split(",") if r.strip()]

        # instantiate the “widen_merging” merger
        merger = MergingMethod("widen_merging")
        merged_params = merger.widen_merging(
            merged_model=merged_model,
            models_to_merge=[finetuned_mod],
            exclude_param_names_regex=regex_list,
            above_average_value_ratio=above_average_value_ratio,
            score_calibration_value=score_calibration_value,
        )

        # load the new weights back into your base_model
        merged_model.load_state_dict(merged_params, strict=False)

        # return the patched wrapper so you can pipe it straight into your sampler
        return (base_model,)


NODE_CLASS_MAPPINGS = {
    "DonutWidenMerge": DonutWidenMerge
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutWidenMerge": "Donut Widen Merge"
}
