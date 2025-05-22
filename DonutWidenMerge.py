import re
import torch
from .merging_methods import MergingMethods

class DonutWidenMerge:
    class_type = "ANY"
    aux_id     = "DonutsDelivery/ComfyUI-DonutDetailer"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "base_model":            ("ANY",),
            "model_to_merge":        ("ANY",),
            "exclude_regex":         ("STRING", {"default": "^$", "multiline": False}),
            "above_avg_value_ratio": ("FLOAT",  {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "score_calibration":     ("FLOAT",  {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("ANY",)
    FUNCTION     = "execute"
    CATEGORY     = "merging"

    def execute(self,
                base_model,
                model_to_merge,
                exclude_regex,
                above_avg_value_ratio,
                score_calibration):
        # Parse exclusion regex list
        regex_list = [r.strip() for r in exclude_regex.split(',') if r.strip()]

        # Initialize WIDEN merger
        merger = MergingMethods()
        # Clone base to avoid in-place changes
        merged_model = merger.widen_merging(
            merged_model=base_model.clone(),
            models_to_merge=[model_to_merge],
            exclude_param_names_regex=regex_list,
            above_average_value_ratio=above_avg_value_ratio,
            score_calibration_value=score_calibration
        )

        # Return the merged model
        return (merged_model,)

NODE_CLASS_MAPPINGS = {
    "DonutWidenMerge": DonutWidenMerge
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutWidenMerge": "Donut Widen Merge"
}
