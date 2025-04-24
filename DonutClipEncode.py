import torch
from nodes import MAX_RESOLUTION

class DonutClipEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT",    {"default": 1024.0, "min": 0,    "max": MAX_RESOLUTION}),
            "height":("INT",    {"default": 1024.0, "min": 0,    "max": MAX_RESOLUTION}),
            "clip":  ("CLIP", ),
            "text_g":("STRING",{"multiline": True, "dynamicPrompts": True}),
            "text_l":("STRING",{"multiline": True, "dynamicPrompts": True}),
            "mix":   ("FLOAT",  {"default": 0.5,     "min": 0.0,  "max": 1.0,  "step": 0.01}),
            "size_cond_factor":("INT",   {"default": 4,      "min": 1,    "max": 16}),
            "layer_idx":   ("INT",     {"default": -2,     "min": -33,  "max": 33}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION     = "execute"
    CATEGORY     = "essentials"

    def execute(self, clip, width, height, size_cond_factor,
                text_g, text_l, mix, layer_idx):
        # 1) upscale
        width  *= size_cond_factor
        height *= size_cond_factor

        # 2) clone & set stopping layer
        clip = clip.clone()
        clip.clip_layer(layer_idx)

        # 3) tokenize both prompts (or empty)
        tokens_g = clip.tokenize(text_g) if text_g.strip() else clip.tokenize("")
        tokens_l = clip.tokenize(text_l) if text_l.strip() else clip.tokenize("")
        empty    = clip.tokenize("")
        empty_g, empty_l = empty["g"], empty["l"]

        # 4) default case: exact mid mix uses single-pass encode
        if abs(mix - 0.5) < 1e-6:
            tokens = {"g": tokens_g["g"], "l": tokens_l["l"]}
            cond_combined, pooled_combined = clip.encode_from_tokens(tokens, return_pooled=True)
        else:
            # split encodings for mix != 0.5
            cond_g, pooled_g = clip.encode_from_tokens(
                {"g": tokens_g["g"], "l": empty_l}, return_pooled=True)
            cond_l, pooled_l = clip.encode_from_tokens(
                {"g": empty_g,       "l": tokens_l["l"]}, return_pooled=True)

            # mix weights sum to 2 to preserve CFG magnitude
            w_l = mix * 2.0
            w_g = (1.0 - mix) * 2.0

            cond_combined   = cond_g    * w_g + cond_l    * w_l
            pooled_combined = pooled_g * w_g + pooled_l * w_l

        # 5) wrap for ComfyUI
        return ([[cond_combined, {
                    "pooled_output": pooled_combined,
                    "width": width, "height": height
                }]], )

NODE_CLASS_MAPPINGS = {
    "DonutClipEncode": DonutClipEncode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutClipEncode": "Donut Clip Encode"
}
