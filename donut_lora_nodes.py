import comfy.sd
import comfy.utils
import folder_paths
import hashlib
from random import random, uniform

# ------------------------------------------------------------------------
# CR: DonutLoRAStack
# ------------------------------------------------------------------------
class DonutLoRAStack:
    """Build a stack of up to 3 LoRAs, each with model/clip strength and per-block weights."""
    class_type = "CUSTOM"
    aux_id     = "DonutsDelivery/ComfyUI-DonutNodes"

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "switch_1":       (["Off", "On"],),
                "lora_name_1":    (loras,),
                "model_weight_1": ("FLOAT", {"default":1.0, "min":-10.0, "max":10.0, "step":0.01}),
                "clip_weight_1":  ("FLOAT", {"default":1.0, "min":-10.0, "max":10.0, "step":0.01}),
                "block_weights_1":("STRING",{"default":"1,1,1,1,1,1,1,1,1,1,1,1"}),
                "switch_2":       (["Off", "On"],),
                "lora_name_2":    (loras,),
                "model_weight_2": ("FLOAT", {"default":1.0, "min":-10.0, "max":10.0, "step":0.01}),
                "clip_weight_2":  ("FLOAT", {"default":1.0, "min":-10.0, "max":10.0, "step":0.01}),
                "block_weights_2":("STRING",{"default":"1,1,1,1,1,1,1,1,1,1,1,1"}),
                "switch_3":       (["Off", "On"],),
                "lora_name_3":    (loras,),
                "model_weight_3": ("FLOAT", {"default":1.0, "min":-10.0, "max":10.0, "step":0.01}),
                "clip_weight_3":  ("FLOAT", {"default":1.0, "min":-10.0, "max":10.0, "step":0.01}),
                "block_weights_3":("STRING",{"default":"1,1,1,1,1,1,1,1,1,1,1,1"}),
            },
            "optional": {
                "lora_stack": ("LORA_STACK",)
            }
        }

    RETURN_TYPES    = ("LORA_STACK", "STRING")
    RETURN_NAMES    = ("lora_stack", "show_help")
    FUNCTION        = "build_stack"
    CATEGORY        = "Comfyroll/LoRA"

    def build_stack(
        self,
        switch_1, lora_name_1, model_weight_1, clip_weight_1, block_weights_1,
        switch_2, lora_name_2, model_weight_2, clip_weight_2, block_weights_2,
        switch_3, lora_name_3, model_weight_3, clip_weight_3, block_weights_3,
        lora_stack=None
    ):
        out = list(lora_stack) if lora_stack else []

        def _parse_blocks(s: str):
            vals = [float(x) for x in s.split(",")]
            if len(vals) != 12:
                raise ValueError("You must supply exactly 12 comma-separated block weights")
            return vals

        def _maybe_add(sw, name, m_w, c_w, b_s):
            if sw == "On" and name != "None":
                out.append((name, m_w, c_w, _parse_blocks(b_s)))

        _maybe_add(switch_1, lora_name_1, model_weight_1, clip_weight_1, block_weights_1)
        _maybe_add(switch_2, lora_name_2, model_weight_2, clip_weight_2, block_weights_2)
        _maybe_add(switch_3, lora_name_3, model_weight_3, clip_weight_3, block_weights_3)

        help_url = (
            "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/"
            "wiki/LoRA-Nodes#donut-lora-stack"
        )
        return (out, help_url)


# ------------------------------------------------------------------------
# CR: DonutApplyLoRAStack
# ------------------------------------------------------------------------
class DonutApplyLoRAStack:
    """Apply a LORA_STACK to a MODEL+CLIP."""
    class_type = "CUSTOM"
    aux_id     = "DonutsDelivery/ComfyUI-DonutNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":      ("MODEL",),
                "clip":       ("CLIP",),
                "lora_stack": ("LORA_STACK",),
            }
        }

    RETURN_TYPES    = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES    = ("model", "clip", "show_help")
    FUNCTION        = "apply_stack"
    CATEGORY        = "Comfyroll/LoRA"

    def apply_stack(self, model, clip, lora_stack=None):
        show_url = (
            "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/"
            "wiki/LoRA-Nodes#donut-apply-lora-stack"
        )
        if not lora_stack:
            return (model, clip, show_url)

        m, c = model, clip
        for name, m_str, c_str, _block_ws in lora_stack:
            pth = folder_paths.get_full_path("loras", name)
            lora = comfy.utils.load_torch_file(pth, safe_load=True)
            # simplified call: no block_weights argument
            m, c = comfy.sd.load_lora_for_models(m, c, lora, m_str, c_str)

        return (m, c, show_url)


# finally, export
NODE_CLASS_MAPPINGS = {
    "DonutLoRAStack":      DonutLoRAStack,
    "DonutApplyLoRAStack": DonutApplyLoRAStack,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    key: key.replace("Donut", "Donut ") for key in NODE_CLASS_MAPPINGS
}
