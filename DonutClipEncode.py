import torch
from nodes import MAX_RESOLUTION

class DonutClipEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip":              ("CLIP",),
            "width":             ("INT",    {"default": 1024, "min": 0,    "max": MAX_RESOLUTION}),
            "height":            ("INT",    {"default": 1024, "min": 0,    "max": MAX_RESOLUTION}),
            "text_g":            ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "text_l":            ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "mix":               ("FLOAT",  {"default": 0.5,  "min": 0.0,  "max": 1.0, "step": 0.01}),
            # Combo dropdown of four modes:
            "preset":            (["default","original","split_only","continuous"],),
            "size_cond_factor":  ("INT",    {"default": 4,    "min": 1,    "max": 16}),
            "layer_idx":         ("INT",    {"default": -2,   "min": -33,  "max": 33}),
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
                preset,
                size_cond_factor,
                layer_idx):
        # 1) upscale resolution
        width  *= size_cond_factor
        height *= size_cond_factor

        # 2) prepare CLIP at given layer
        clip = clip.clone()
        clip.clip_layer(layer_idx)

        # 3) tokenize or empty
        empty = clip.tokenize("")
        eg, el = empty["g"], empty["l"]
        tg = clip.tokenize(text_g) if text_g.strip() else empty
        tl = clip.tokenize(text_l) if text_l.strip() else empty

        # 4) handle presets
        if preset == "default":
            # joint encode both prompts together
            cond, pooled = clip.encode_from_tokens(
                {"g": tg["g"], "l": tl["l"]}, return_pooled=True
            )

        elif preset == "original":
            # single-pass at mix≈0.5, else split & sum weights=2
            if abs(mix - 0.5) < 1e-6:
                cond, pooled = clip.encode_from_tokens(
                    {"g": tg["g"], "l": tl["l"]}, return_pooled=True
                )
            else:
                cg, pg = clip.encode_from_tokens({"g": tg["g"], "l": el}, return_pooled=True)
                cl, pl = clip.encode_from_tokens({"g": eg,       "l": tl["l"]}, return_pooled=True)
                wg, wl = (1.0 - mix)*2.0, mix*2.0
                cond   = cg*wg + cl*wl
                pooled = pg*wg + pl*wl

        elif preset == "split_only":
            # always split into g-only and l-only, weights sum=2
            cg, pg = clip.encode_from_tokens({"g": tg["g"], "l": el}, return_pooled=True)
            cl, pl = clip.encode_from_tokens({"g": eg,       "l": tl["l"]}, return_pooled=True)
            wg, wl = (1.0 - mix)*2.0, mix*2.0
            cond   = cg*wg + cl*wl
            pooled = pg*wg + pl*wl

        else:  # continuous
            # joint encode
            cj, pj = clip.encode_from_tokens({"g": tg["g"], "l": tl["l"]}, return_pooled=True)
            # split encode
            cg, pg = clip.encode_from_tokens({"g": tg["g"], "l": el}, return_pooled=True)
            cl, pl = clip.encode_from_tokens({"g": eg,       "l": tl["l"]}, return_pooled=True)
            ws_g, ws_l = (1.0 - mix)*2.0, mix*2.0
            cs = cg*ws_g + cl*ws_l
            ps = pg*ws_g + pl*ws_l
            # blend joint vs split
            wj = 1.0 - 2.0 * abs(mix - 0.5)
            ws = 1.0 - wj
            cond   = cs*ws + cj*wj
            pooled = ps*ws + pj*wj

        # 5) wrap for ComfyUI
        return ([[cond, {"pooled_output": pooled,
                         "width": width, "height": height}]],)

NODE_CLASS_MAPPINGS = {
    "DonutClipEncode": DonutClipEncode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutClipEncode": "Donut Clip Encode"
}
