import comfy.sd
import comfy.utils
import folder_paths
from .lora_block_weight import LoraLoaderBlockWeight

# ------------------------------------------------------------------------
# DonutLoRAStack: build up to 3 LoRAs with independent model & clip strengths + optional per-block vectors
# ------------------------------------------------------------------------
class DonutLoRAStack:
    class_type = "CUSTOM"
    aux_id     = "DonutsDelivery/ComfyUI-DonutNodes"

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "switch_1":       (["Off","On"],),
                "lora_name_1":    (loras,),
                "model_weight_1": ("FLOAT", {"default":1.0,"min":-1000,"max":1000,"step":0.01}),
                "clip_weight_1":  ("FLOAT", {"default":1.0,"min":-1000,"max":1000,"step":0.01}),
                "block_vector_1": ("STRING",{"default":"1,1,1,1,1,1,1,1,1,1,1,1","placeholder":"12 comma-sep floats"}),

                "switch_2":       (["Off","On"],),
                "lora_name_2":    (loras,),
                "model_weight_2": ("FLOAT", {"default":1.0,"min":-1000,"max":1000,"step":0.01}),
                "clip_weight_2":  ("FLOAT", {"default":1.0,"min":-1000,"max":1000,"step":0.01}),
                "block_vector_2": ("STRING",{"default":"1,1,1,1,1,1,1,1,1,1,1,1","placeholder":"optional"}),

                "switch_3":       (["Off","On"],),
                "lora_name_3":    (loras,),
                "model_weight_3": ("FLOAT", {"default":1.0,"min":-1000,"max":1000,"step":0.01}),
                "clip_weight_3":  ("FLOAT", {"default":1.0,"min":-1000,"max":1000,"step":0.01}),
                "block_vector_3": ("STRING",{"default":"1,1,1,1,1,1,1,1,1,1,1,1","placeholder":"optional"}),
            },
            "optional": {"lora_stack": ("LORA_STACK",)},
        }

    RETURN_TYPES = ("LORA_STACK","STRING")
    RETURN_NAMES = ("lora_stack","show_help")
    FUNCTION     = "build_stack"
    CATEGORY     = "Comfyanonymous/LoRA"

    def build_stack(
        self,
        switch_1, lora_name_1, model_weight_1, clip_weight_1, block_vector_1,
        switch_2, lora_name_2, model_weight_2, clip_weight_2, block_vector_2,
        switch_3, lora_name_3, model_weight_3, clip_weight_3, block_vector_3,
        lora_stack=None
    ):
        stack = list(lora_stack) if lora_stack else []

        def _maybe_add(sw, name, mw, cw, bv):
            if sw == "On" and name != "None":
                stack.append((name, mw, cw, bv.strip()))

        _maybe_add(switch_1, lora_name_1, model_weight_1, clip_weight_1, block_vector_1)
        _maybe_add(switch_2, lora_name_2, model_weight_2, clip_weight_2, block_vector_2)
        _maybe_add(switch_3, lora_name_3, model_weight_3, clip_weight_3, block_vector_3)

        help_url = (
            "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/"
            "wiki/LoRA-Nodes#cr-lora-stack"
        )
        return (stack, help_url)


# ------------------------------------------------------------------------
# DonutApplyLoRAStack: per-block UNet + uniform CLIP merges, always in that order
# ------------------------------------------------------------------------
class DonutApplyLoRAStack:
    class_type = "CUSTOM"
    aux_id     = "DonutsDelivery/ComfyUI-DonutNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "model":      ("MODEL",),
            "clip":       ("CLIP",),
            "lora_stack": ("LORA_STACK",),
        }}

    RETURN_TYPES = ("MODEL","CLIP","STRING")
    RETURN_NAMES = ("model","clip","show_help")
    FUNCTION     = "apply_stack"
    CATEGORY     = "Comfyanonymous/LoRA"

    def apply_stack(self, model, clip, lora_stack=None):
        help_url = (
            "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/"
            "wiki/LoRA-Nodes#cr-apply-lora-stack"
        )
        if not lora_stack:
            return (model, clip, help_url)

        unet, text_enc = model, clip
        loader         = LoraLoaderBlockWeight()

        for name, mw, cw, bv in lora_stack:
            path = folder_paths.get_full_path("loras", name)
            lora = comfy.utils.load_torch_file(path, safe_load=True)
            vector = bv if bv else ",".join(["1"] * 12)

            # 1) block-weighted UNet merge (clip_strength=0)
            unet, text_enc, _ = loader.load_lora_for_models(
                unet, text_enc, lora,
                strength_model=    mw,
                strength_clip=     0.0,
                inverse=           False,
                seed=              0,
                A=                 1.0,
                B=                 1.0,
                block_vector=      vector
            )

            # 2) uniform CLIP merge (no block control)
            unet, text_enc = comfy.sd.load_lora_for_models(
                unet, text_enc, lora,
                0.0,               # no UNet change
                cw                 # clip strength
            )

        return (unet, text_enc, help_url)


NODE_CLASS_MAPPINGS = {
    "DonutLoRAStack":      DonutLoRAStack,
    "DonutApplyLoRAStack": DonutApplyLoRAStack,
}
NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
