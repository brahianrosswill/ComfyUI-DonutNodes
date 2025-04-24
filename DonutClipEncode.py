import torch
from nodes import MAX_RESOLUTION

class DonutClipEncode:
    class_type = "CLIP"
    aux_id = "DonutsDelivery/ComfyUI-DonutDetailer"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip":                   ("CLIP",),
            "width":                  ("INT",    {"default": 1024, "min": 0,    "max": MAX_RESOLUTION}),
            "height":                 ("INT",    {"default": 1024, "min": 0,    "max": MAX_RESOLUTION}),
            "text_g":                 ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "text_l":                 ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "mix":                    ("FLOAT",  {"default": 0.5,  "min": 0.0,  "max": 1.0,   "step": 0.01}),
            "split_vs_pooled_ratio":  ("FLOAT",  {"default": 0.5,  "min": 0.0,  "max": 1.0,   "step": 0.0001}),
            "preset": (["Default", "Split Only", "Continuous", "Split vs Pooled"],),
            "size_cond_factor":       ("INT",    {"default": 4,    "min": 1,    "max": 16}),
            "layer_idx":              ("INT",    {"default": -2,   "min": -33,  "max": 33}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION     = "execute"
    CATEGORY     = "essentials"

    def execute(self,
                clip,
                width,
                height,
                text_g,
                text_l,
                mix,
                split_vs_pooled_ratio,
                preset,
                size_cond_factor,
                layer_idx):
        # 1) Upscale resolution
        width  *= size_cond_factor
        height *= size_cond_factor

        # 2) Prepare CLIP model and stop at layer
        clip = clip.clone()
        clip.clip_layer(layer_idx)

        # 3) Tokenize prompts or empty
        empty = clip.tokenize("")
        eg, el = empty['g'], empty['l']
        tg = clip.tokenize(text_g) if text_g.strip() else empty
        tl = clip.tokenize(text_l) if text_l.strip() else empty

        # 4) Joint encoding
        cond_joint, pooled_joint = clip.encode_from_tokens({'g': tg['g'], 'l': tl['l']}, return_pooled=True)

        # 5) Split-only encoding by mix
        cond_g, pooled_g = clip.encode_from_tokens({'g': tg['g'], 'l': el}, return_pooled=True)
        cond_l, pooled_l = clip.encode_from_tokens({'g': eg,       'l': tl['l']}, return_pooled=True)
        cond_split   = cond_g * (1 - mix) + cond_l * mix
        pooled_split = pooled_g * (1 - mix) + pooled_l * mix

        # 6) Dispatch by preset
        if preset == 'Default':
            cond, pooled = cond_joint, pooled_joint
        elif preset == 'Split Only':
            cond, pooled = cond_split, pooled_split
        elif preset == 'Continuous':
            exp = 3.0
            alpha = mix ** (1.0 / exp)
            cond   = cond_joint * (1 - alpha) + cond_split * alpha
            pooled = pooled_joint * (1 - alpha) + pooled_split * alpha
        else:  # Split vs Pooled
            gamma = 0.3
            alpha = split_vs_pooled_ratio ** gamma
            cond   = cond_split
            pooled = pooled_split * alpha + pooled_joint * (1.0 - alpha)

        # 7) Wrap and return
        return ([[cond, {'pooled_output': pooled, 'width': width, 'height': height}]],)

NODE_CLASS_MAPPINGS = {
    'DonutClipEncode': DonutClipEncode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'DonutClipEncode': 'Donut Clip Encode'
}
