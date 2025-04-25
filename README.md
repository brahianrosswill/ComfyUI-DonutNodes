# ComfyUI-DonutDetailer

A collection of **“Donut Detailer”** custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), enabling fine-grained control over model and LoRA weight/bias parameters.

---

## 📦 Installation

1. Clone into your ComfyUI `custom_nodes` folder:  
   ```bash
   cd ~/.config/ComfyUI/custom_nodes
   git clone https://github.com/DonutsDelivery/ComfyUI-DonutDetailer.git

    Restart ComfyUI.

    In the node picker, look under Model Patches and LoRA Patches.

⚙️ Nodes
🍩 Donut Detailer (Model)
Inputs
Name	Type	Description
MODEL	MODEL	The model to patch.
Scale_in	FLOAT	Scale factor for input-group weight adjustments.
Weight_in	FLOAT	Input-group weight multiplier.
Bias_in	FLOAT	Input-group bias multiplier.
Scale_out0	FLOAT	Scale factor for first output-group weight adjustments.
Weight_out0	FLOAT	First output-group weight multiplier.
Bias_out0	FLOAT	First output-group bias multiplier.
Scale_out2	FLOAT	Scale factor for second output-group weight adjustments.
Weight_out2	FLOAT	Second output-group weight multiplier.
Bias_out2	FLOAT	Second output-group bias multiplier.
Effect

    Weights: 1 – Scale × Weight

    Biases: 1 + Scale × Bias

🍩 Donut Detailer 2 (Model)
Inputs
Name	Type	Description
MODEL	MODEL	The model to patch.
K_in	FLOAT	Base coefficient for input-group weight/bias.
S1_in	FLOAT	Weight-scale factor for input-group.
S2_in	FLOAT	Bias-scale factor for input-group.
K_out0	FLOAT	Base coefficient for first output-group weight/bias.
S1_out0	FLOAT	Weight-scale factor for first output-group.
S2_out0	FLOAT	Bias-scale factor for first output-group.
K_out2	FLOAT	Base coefficient for second output-group weight/bias.
S1_out2	FLOAT	Weight-scale factor for second output-group.
S2_out2	FLOAT	Bias-scale factor for second output-group.
Formula

Weight multiplier = 1 – (K × S1 × 0.01)
Bias multiplier   = 1 + (K × S2 × 0.02)

🍩 Donut Detailer 4 (Model)
Inputs
Name	Type	Description
MODEL	MODEL	The model to patch.
Weight_in	FLOAT	Input-group weight multiplier (direct).
Bias_in	FLOAT	Input-group bias multiplier (direct).
Weight_out0	FLOAT	First output-group weight multiplier (direct).
Bias_out0	FLOAT	First output-group bias multiplier (direct).
Weight_out2	FLOAT	Second output-group weight multiplier (direct).
Bias_out2	FLOAT	Second output-group bias multiplier (direct).
Effect

Multiplies each group’s weights and biases by the slider value (default 1 = bypass).
🍩 Donut Detailer LoRA 6 (LoRA)
Inputs
Name	Type	Description
LoRA	LoRA	The LoRA patch to modify.
Weight_down	FLOAT	Down-layer weight multiplier.
Bias_down	FLOAT	Down-layer bias multiplier.
Weight_mid	FLOAT	Mid-layer weight multiplier.
Bias_mid	FLOAT	Mid-layer bias multiplier.
Weight_up	FLOAT	Up-layer weight multiplier.
Bias_up	FLOAT	Up-layer bias multiplier.
Effect

Scales LoRA down/mid/up layers’ weights & biases and lets you save the modified LoRA directly.
🍩 Donut Detailer XL Blocks (Model)
Inputs
Name	Type	Description
MODEL	MODEL	The model to patch.
W_block_*	FLOAT	Weight multiplier for each SDXL UNet block (various).
B_block_*	FLOAT	Bias multiplier for each SDXL UNet block (various).
Effect

Direct control over every major SDXL UNet block for highly granular tweaking.
💡 Tips

    Use Donut Detailer 2 for the closest mimic of Supermerger’s “Adjust.”

    Use Donut Detailer 4 for straightforward weight/bias multipliers.

    Use LoRA 6 to save custom-scaled LoRA patches.

    Use XL Blocks when you need per-block control in SDXL.

# Donut Clip Encode (Mix Only)

A custom ComfyUI node for Stable Diffusion XL that lets you blend the two CLIP branches ("g" and "l") with a single **Mix** slider and convenient presets.

---

## 📦 Installation

1. Copy `DonutClipEncodeMixOnly.py` into your ComfyUI `custom_nodes/ComfyUI-DonutDetailer/` folder.  
2. Clear the UI cache:
   ```bash
   rm ~/.cache/comfyui/ui_cache_nodes.json
   ```
3. Restart ComfyUI.  
4. In the node picker, search for **Donut Clip Encode (Mix Only)** and add it to your graph.

---

## ⚙️ Inputs

| Name                   | Type    | Description                                                                                       |
|------------------------|---------|---------------------------------------------------------------------------------------------------|
| `clip`                 | CLIP    | The CLIP encoder from your SDXL pipeline.                                                         |
| `width` / `height`     | INT     | Base image resolution. Scaled internally by **size_cond_factor**.                                  |
| `text_g` / `text_l`    | STRING  | "Guidance" and "Language" prompts for the two CLIP branches.                                  |
| `mix`                  | FLOAT   | (0.0–1.0) Linear blend used by **Split Only** and **Continuous** presets.                          |
| `split_vs_pooled_ratio`| FLOAT   | (0.0–1.0) Blend used only by the **Split vs Pooled** preset to mix sequence vs pooled embeddings. |
| `preset`               | ENUM    | Choose one of: `Default`, `Split Only`, `Continuous`, `Split vs Pooled`.                           |
| `size_cond_factor`     | INT     | Upscale factor for internal CLIP resolution.                                                      |
| `layer_idx`            | INT     | CLIP layer index to stop at (e.g. `-2` for penultimate).                                         |

---

## 🎛️ Presets

### 🅰️ Default
Encodes both prompts in a single pass with:
```python
cond, pooled = clip.encode_from_tokens({ 'g': tokens_g, 'l': tokens_l }, return_pooled=True)
``` 
**Use case:** Standard SDXL behavior.

### 🅱️ Split Only
Two‐pass encoding:
```python
cond_split   = cond_g * (1-mix) + cond_l * mix
pooled_split = pooled_g * (1-mix) + pooled_l * mix
``` 
**Use case:** Direct control of each branch’s contribution.

### 🆑 Continuous
Smooth interpolation between joint and split:
```python
alpha = mix ** (1/3)
cond   = cond_joint * (1-alpha) + cond_split * alpha
pooled = pooled_joint * (1-alpha) + pooled_split * alpha
``` 
**Use case:** Gradual transition biased toward split.

### 🆂 Split vs Pooled
Keep full split sequence, but bias pooled summary:
```python
alpha = split_vs_pooled_ratio ** 0.3
cond   = cond_split
pooled = pooled_split * alpha + pooled_joint * (1-alpha)
``` 
**Use case:** Full split detail with adjustable CFG strength.

---

## 💡 Tips

- **Even mix:** Set **mix = 0.5** in **Split Only** for a 50/50 blend.  
- **Strong split effect:** In **Continuous**, raise **mix > 0.7**.  
- **High CFG strength:** In **Split vs Pooled**, small slider moves near 1.0 have large impact.

---

## 📜 License
MIT © DonutsDelivery


