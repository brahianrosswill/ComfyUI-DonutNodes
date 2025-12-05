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
        seq_len = max(len(tg['g']), len(tl['l']))
        # pad tokens
        def pad_tokens(tok_list, pad_elem):
            if not tok_list:
                return [pad_elem] * seq_len
            repeat = (seq_len + len(tok_list) - 1)//len(tok_list)
            return (tok_list * repeat)[:seq_len]
        tokens = {
            'g': pad_tokens(tg['g'], eg),
            'l': pad_tokens(tl['l'], el)
        }

        # 4) Joint encoding
        cond_joint, pooled_joint = clip.encode_from_tokens(tokens, return_pooled=True)

        # 5) Split-only branch
        cond_g, pooled_g = clip.encode_from_tokens({'g': tokens['g'], 'l': pad_tokens(el, el)}, return_pooled=True)
        cond_l, pooled_l = clip.encode_from_tokens({'g': pad_tokens(eg, eg), 'l': tokens['l']}, return_pooled=True)
        cond_split   = cond_g   * (1 - clip_gl_mix) + cond_l   * clip_gl_mix
        pooled_split = pooled_g * (1 - clip_gl_mix) + pooled_l * clip_gl_mix

        # 6) Continuous branch (gamma-biased)
        exp     = 3.0
        alpha_c = clip_gl_mix ** (1.0 / exp)
        cond_cont   = cond_joint   * (1 - alpha_c) + cond_split   * alpha_c
        pooled_cont = pooled_joint * (1 - alpha_c) + pooled_split * alpha_c

        # 7) Choose base embedding
        if mode == 'Strength Mode':
            base_cond = cond_g * clip_g_strength + cond_l * clip_l_strength
            base_pooled = pooled_g * clip_g_strength + pooled_l * clip_l_strength
        else:
            if preset == 'Default':
                base_cond, base_pooled = cond_joint, pooled_joint
            elif preset == 'Split Only':
                base_cond, base_pooled = cond_split, pooled_split
            elif preset == 'Continuous':
                base_cond, base_pooled = cond_cont, pooled_cont
            elif preset == 'Split vs Pooled':
                alpha = vs_mix ** 0.3
                base_cond = cond_split
                base_pooled = pooled_split * alpha + pooled_joint * (1 - alpha)
            elif preset == 'Split vs Continuous':
                base_cond = cond_split * (1 - clip_gl_mix) + cond_cont * clip_gl_mix
                base_pooled = pooled_split * (1 - clip_gl_mix) + pooled_cont * clip_gl_mix
            elif preset == 'Default vs Split':
                base_cond = cond_joint * (1 - clip_gl_mix) + cond_split * clip_gl_mix
                base_pooled = pooled_joint * (1 - clip_gl_mix) + pooled_split * clip_gl_mix
            elif preset == 'Default vs Continuous':
                base_cond = cond_joint * (1 - clip_gl_mix) + cond_cont * clip_gl_mix
                base_pooled = pooled_joint * (1 - clip_gl_mix) + pooled_cont * clip_gl_mix
            elif preset == 'Strength Blend':
                w = torch.tensor([strength_default, strength_split, strength_continuous], device=pooled_joint.device)
                total = w.sum()
                if total <= 0:
                    base_cond, base_pooled = cond_joint, pooled_joint
                else:
                    # normalize weights
                    w = w / total
                    base_cond = cond_joint * w[0] + cond_split * w[1] + cond_cont * w[2]
                    base_pooled = pooled_joint * w[0] + pooled_split * w[1] + pooled_cont * w[2]
            else:
                base_cond, base_pooled = cond_joint, pooled_joint

                        # 8) Auto-adjust CFG (skip for Default preset in Mix Mode)
        if not (mode == 'Mix Mode' and preset == 'Default'):
            eps = 1e-8
            # match sequence embedding norm rather than pooled, for consistent guidance
            joint_norm = cond_joint.norm()
            base_norm  = base_cond.norm()
            scale = joint_norm / (base_norm + eps)
            cond_final = base_cond * scale
            pooled_final = base_pooled * scale
        else:
            cond_final = base_cond
            pooled_final = base_pooled

                # 9) Wrap and return
        return ([[cond_final, {'pooled_output': pooled_final, 'width': width, 'height': height}]],)

NODE_CLASS_MAPPINGS = {
    'DonutClipEncode': DonutClipEncode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'DonutClipEncode': 'Donut Clip Encode'
}
