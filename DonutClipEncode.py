import torch
from nodes import MAX_RESOLUTION

class DonutClipEncode:
    class_type = "CLIP"
    aux_id     = "DonutsDelivery/ComfyUI-DonutDetailer"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip":                 ("CLIP",),
            "width":                ("INT",   {"default": 1024, "min": 0,    "max": MAX_RESOLUTION}),
            "height":               ("INT",   {"default": 1024, "min": 0,    "max": MAX_RESOLUTION}),
            "text_g":               ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "text_l":               ("STRING", {"multiline": True, "dynamicPrompts": True}),
            # Mode toggle
            "mode":                (["Mix Mode", "Strength Mode"],),
            # Mix Mode sliders
            "clip_gl_mix":         ("FLOAT", {"default": 0.5,  "min": 0.0, "max": 1.0,   "step": 0.01}),
            "vs_mix":              ("FLOAT", {"default": 0.5,  "min": 0.0, "max": 1.0,   "step": 0.01}),
            # Strength Mode sliders
            "clip_g_strength":     ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 1000.0, "step": 0.01}),
            "clip_l_strength":     ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 1000.0, "step": 0.01}),
            # Strength Blend preset sliders (Mix Mode only)
            "strength_default":    ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 10.0,   "step": 0.01}),
            "strength_split":      ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 10.0,   "step": 0.01}),
            "strength_continuous": ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 10.0,   "step": 0.01}),
            # Mix Mode presets
            "preset":             ([
                                      "Default",
                                      "Split Only",
                                      "Continuous",
                                      "Split vs Pooled",
                                      "Split vs Continuous",
                                      "Default vs Split",
                                      "Default vs Continuous",
                                      "Strength Blend"
                                   ],),
            "size_cond_factor":    ("INT",   {"default": 4,    "min": 1,    "max": 16}),
            "layer_idx":           ("INT",   {"default": -2,   "min": -33,  "max": 33}),
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
                mode,
                clip_gl_mix,
                vs_mix,
                clip_g_strength,
                clip_l_strength,
                strength_default,
                strength_split,
                strength_continuous,
                preset,
                size_cond_factor,
                layer_idx):
        # 1) Upscale resolution
        width  *= size_cond_factor
        height *= size_cond_factor

        # 2) Prepare CLIP and stop at layer
        clip = clip.clone()
        clip.clip_layer(layer_idx)

        # 3) Tokenize prompts and pad to equal length
        empty = clip.tokenize("")
        eg, el = empty['g'], empty['l']
        tg = clip.tokenize(text_g) if text_g.strip() else clip.tokenize("")
        tl = clip.tokenize(text_l) if text_l.strip() else clip.tokenize("")
        # pad shorter token list
        while len(tg['g']) > len(tl['l']):
            tl['l'] += el
        while len(tl['l']) > len(tg['g']):
            tg['g'] += eg
        tokens = {'g': tg['g'], 'l': tl['l']}

        # 4) Joint encoding
        cond_joint, pooled_joint = clip.encode_from_tokens(tokens, return_pooled=True)

                # 5) Split-only branch
        seq_len = len(tokens['g'])
        # pad empties to match joint sequence length
        el_base = el
        eg_base = eg
        def pad(tokens_base):
            if not tokens_base:
                return []
            repeat = (seq_len + len(tokens_base) - 1) // len(tokens_base)
            padded = (tokens_base * repeat)[:seq_len]
            return padded
        empty_l_padded = pad(el_base)
        empty_g_padded = pad(eg_base)

        cond_g, pooled_g = clip.encode_from_tokens(
            {'g': tokens['g'], 'l': empty_l_padded}, return_pooled=True
        )
        cond_l, pooled_l = clip.encode_from_tokens(
            {'g': empty_g_padded, 'l': tokens['l']}, return_pooled=True
        )
        cond_split   = cond_g   * (1 - clip_gl_mix) + cond_l   * clip_gl_mix
        pooled_split = pooled_g * (1 - clip_gl_mix) + pooled_l * clip_gl_mix

        # 6) Continuous branch (gamma-biased) (gamma-biased)
        exp     = 3.0
        alpha_c = clip_gl_mix ** (1.0 / exp)
        cond_cont   = cond_joint   * (1 - alpha_c) + cond_split   * alpha_c
        pooled_cont = pooled_joint * (1 - alpha_c) + pooled_split * alpha_c

        # 7) Strength Mode
        if mode == 'Strength Mode':
            cond   = cond_g * clip_g_strength + cond_l * clip_l_strength
            pooled = pooled_g * clip_g_strength + pooled_l * clip_l_strength

        else:
            # 8) Mix Mode dispatch
            if preset == 'Default':
                cond, pooled = cond_joint, pooled_joint
            elif preset == 'Split Only':
                cond, pooled = cond_split, pooled_split
            elif preset == 'Continuous':
                cond, pooled = cond_cont, pooled_cont
            elif preset == 'Split vs Pooled':
                gamma = 0.3
                alpha = vs_mix ** gamma
                cond   = cond_split
                pooled = pooled_split * alpha + pooled_joint * (1 - alpha)
            elif preset == 'Split vs Continuous':
                cond   = cond_split   * (1 - clip_gl_mix) + cond_cont   * clip_gl_mix
                pooled = pooled_split * (1 - clip_gl_mix) + pooled_cont * clip_gl_mix
            elif preset == 'Default vs Split':
                cond   = cond_joint   * (1 - clip_gl_mix) + cond_split   * clip_gl_mix
                pooled = pooled_joint * (1 - clip_gl_mix) + pooled_split * clip_gl_mix
            elif preset == 'Default vs Continuous':
                cond   = cond_joint   * (1 - clip_gl_mix) + cond_cont   * clip_gl_mix
                pooled = pooled_joint * (1 - clip_gl_mix) + pooled_cont * clip_gl_mix
            elif preset == 'Strength Blend':
                # blend three mix-based embeddings
                w_def = strength_default
                w_sp  = strength_split
                w_ct  = strength_continuous
                total = w_def + w_sp + w_ct
                if total <= 0:
                    cond, pooled = cond_joint, pooled_joint
                else:
                    cond   = (cond_joint* w_def   + cond_split* w_sp   + cond_cont* w_ct)   / total
                    pooled = (pooled_joint* w_def + pooled_split* w_sp + pooled_cont* w_ct) / total
            else:
                cond, pooled = cond_joint, pooled_joint

        # 9) Wrap and return
        return ([[cond, {'pooled_output': pooled, 'width': width, 'height': height}]],)

NODE_CLASS_MAPPINGS = {
    'DonutClipEncode': DonutClipEncode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'DonutClipEncode': 'Donut Clip Encode'
}
