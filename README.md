This is an experimental node I made to mimick the "adjust" in A1111 Supermerger (https://github.com/hako-mikan/sd-webui-supermerger?tab=readme-ov-file#adjust). It adds more noise and texture to the output, and can also adjust gamma/brightness using the last parameter. The 3 variants are fundamentally doing the same thing, but with different parameter controls. Scale/multiplier multiplies the S1 (weight) and S2 (bias) parameters. 

Donut Detailer: Initial try at making the node, dosent mimick supermerger accurately. 

![image](https://github.com/user-attachments/assets/b0477a38-86c2-42fd-a635-82afdef3b8a4)

Donut Detailer 2: Mimicks closes Supermerger Adjust parameters. 

![image](https://github.com/user-attachments/assets/6d0cc683-e005-481b-abe4-487700686df3)

Donut Detailer 4: Making it more barebone, without the coefficients. 

![image](https://github.com/user-attachments/assets/e1bafb1c-a24a-448e-92f9-f9e27f98157d)

Thanks to epiTune for helping me make this, and ChatGPT. Note: epiTune does not think this is the best solution to adding more texture as it is a a crude way of modifying the model, use it sparingly.

Donut Clip Encode (Mix Only)

A custom ComfyUI node for Stable Diffusion XL that gives you fine-grained control over the balance between the two CLIP branches ("g" and "l"). Instead of separate strength sliders, this node offers a single Mix slider and four handy presets to adjust how prompts are encoded and combined.

Installation

Copy DonutClipEncodeMixOnly.py into your custom_nodes/ComfyUI-DonutDetailer/ folder.

Clear the UI cache:

rm ~/.cache/comfyui/ui_cache_nodes.json

Restart ComfyUI.

In the node picker, search for Donut Clip Encode (Mix Only).

Inputs

clip: The CLIP encoder from your SDXL pipeline.

width / height: Base resolution (scaled internally by size_cond_factor).

text_g / text_l: Guidance and language prompts for the two CLIP branches.

mix: A value between 0.0 → 1.0 controlling the linear blend for the Split Only and Continuous presets.

split_vs_pooled_ratio: A value between 0.0 → 1.0 used only by the Split vs Pooled preset to blend sequence vs pooled embeddings.

preset: Choose one of:

Default: Standard joint encode of both prompts.

Split Only: Two‑pass encode (g‑only + l‑only) blended by mix.

Continuous: Smooth, gamma‑biased interpolation (mix^(1/3)) between joint and split encodings.

Split vs Pooled: Keeps the split sequence embeddings, but blends the pooled summary using an exponential bias (split_vs_pooled_ratio^0.3).

size_cond_factor: Multiplier for internal CLIP resolution.

layer_idx: CLIP stopping layer (e.g. -2 for penultimate).

Presets Explained

Default

Behavior: Encodes both prompts in a single pass (encode_from_tokens({"g": …, "l": …})).

Use case: When you want the out‑of‑the‑box SDXL behavior without any custom mixing.

Split Only

Behavior: Encodes g and l separately, then linearly blends their outputs by mix:

cond_split   = cond_g * (1 - mix) + cond_l * mix
pooled_split = pooled_g * (1 - mix) + pooled_l * mix

Use case: Direct control over how much each branch contributes, without influencing the joint encoding at all.

Continuous

Behavior: Blends joint vs split encodings with a gamma‑root bias on mix:

alpha = mix ** (1/3)
cond   = cond_joint * (1 - alpha) + cond_split * alpha
pooled = pooled_joint * (1 - alpha) + pooled_split * alpha

Use case: Gradual transition from joint to split behavior, with mid‑values weighted towards split for stronger effect.

Split vs Pooled

Behavior: Uses the split sequence embedding unmodified, but blends only the pooled summary between split and joint via:

alpha = split_vs_pooled_ratio ** 0.3
pooled = pooled_split * alpha + pooled_joint * (1 - alpha)
cond   = cond_split

Use case: When you want the full detail from the split sequence but still adjust how much of the pooled (CFG) signal comes from each branch—mid‑slider values heavily biased toward the split summary.

Usage Tips

Rapid prototyping: Try Split Only at mix=0.5 to evenly mix both branches with full CFG strength.

Strong split signal: In Continuous, raising mix above 0.7 quickly shifts you toward the split encoding.

Fine‑tune CFG: In Split vs Pooled, use small adjustments near the extremes (0.8–1.0) to retain most of the split detail while dialing CFG influence.
