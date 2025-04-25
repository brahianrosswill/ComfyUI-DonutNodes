# ComfyUI-DonutDetailer

A collection of **“Donut Detailer”** custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), enabling fine-grained control over model and LoRA weight/bias parameters.

---

## 📦 Installation

1. Clone into your ComfyUI `custom_nodes` folder:  
   ```bash
   cd ~/.config/ComfyUI/custom_nodes
   git clone https://github.com/DonutsDelivery/ComfyUI-DonutDetailer.git
   ```
2. Restart ComfyUI.  
3. In the node picker, look under **Model Patches** and **LoRA Patches**.

---

## ⚙️ Nodes

### 🍩 Donut Detailer (Model)

#### Inputs

| Name          | Type   | Description                                                      |
| ------------- | ------ | ---------------------------------------------------------------- |
| `MODEL`       | MODEL  | The model to patch.                                              |
| `Scale_in`    | FLOAT  | Scale factor for input-group weight adjustments.                 |
| `Weight_in`   | FLOAT  | Input-group weight multiplier.                                   |
| `Bias_in`     | FLOAT  | Input-group bias multiplier.                                     |
| `Scale_out0`  | FLOAT  | Scale factor for first output-group weight adjustments.          |
| `Weight_out0` | FLOAT  | First output-group weight multiplier.                            |
| `Bias_out0`   | FLOAT  | First output-group bias multiplier.                              |
| `Scale_out2`  | FLOAT  | Scale factor for second output-group weight adjustments.         |
| `Weight_out2` | FLOAT  | Second output-group weight multiplier.                           |
| `Bias_out2`   | FLOAT  | Second output-group bias multiplier.                             |

#### Effect

- **Weights:** `1 – Scale × Weight`  
- **Biases:** `1 + Scale × Bias`

---

### 🍩 Donut Detailer 2 (Model)

#### Inputs

| Name         | Type   | Description                                               |
| ------------ | ------ | --------------------------------------------------------- |
| `MODEL`      | MODEL  | The model to patch.                                       |
| `K_in`       | FLOAT  | Base coefficient for input-group weight/bias.             |
| `S1_in`      | FLOAT  | Weight-scale factor for input-group.                      |
| `S2_in`      | FLOAT  | Bias-scale factor for input-group.                        |
| `K_out0`     | FLOAT  | Base coefficient for first output-group weight/bias.      |
| `S1_out0`    | FLOAT  | Weight-scale factor for first output-group.               |
| `S2_out0`    | FLOAT  | Bias-scale factor for first output-group.                 |
| `K_out2`     | FLOAT  | Base coefficient for second output-group weight/bias.     |
| `S1_out2`    | FLOAT  | Weight-scale factor for second output-group.              |
| `S2_out2`    | FLOAT  | Bias-scale factor for second output-group.                |

#### Formula

\`\`\`text
Weight multiplier = 1 – (K × S1 × 0.01)
Bias multiplier   = 1 + (K × S2 × 0.02)
\`\`\`

---

### 🍩 Donut Detailer 4 (Model)

#### Inputs

| Name          | Type   | Description                                    |
| ------------- | ------ | ---------------------------------------------- |
| `MODEL`       | MODEL  | The model to patch.                            |
| `Weight_in`   | FLOAT  | Input-group weight multiplier (direct).        |
| `Bias_in`     | FLOAT  | Input-group bias multiplier (direct).          |
| `Weight_out0` | FLOAT  | First output-group weight multiplier (direct). |
| `Bias_out0`   | FLOAT  | First output-group bias multiplier (direct).   |
| `Weight_out2` | FLOAT  | Second output-group weight multiplier (direct).|
| `Bias_out2`   | FLOAT  | Second output-group bias multiplier (direct).  |

#### Effect

Multiplies each group’s weights and biases by the slider value (default 1 = bypass).

---

### 🍩 Donut Detailer LoRA 6 (LoRA)

#### Inputs

| Name          | Type   | Description                                      |
| ------------- | ------ | ------------------------------------------------ |
| `LoRA`        | LoRA   | The LoRA patch to modify.                        |
| `Weight_down` | FLOAT  | Down-layer weight multiplier.                    |
| `Bias_down`   | FLOAT  | Down-layer bias multiplier.                      |
| `Weight_mid`  | FLOAT  | Mid-layer weight multiplier.                     |
| `Bias_mid`    | FLOAT  | Mid-layer bias multiplier.                       |
| `Weight_up`   | FLOAT  | Up-layer weight multiplier.                      |
| `Bias_up`     | FLOAT  | Up-layer bias multiplier.                        |

#### Effect

Scales LoRA down/mid/up layers’ weights & biases and lets you save the modified LoRA directly.

---

### 🍩 Donut Detailer XL Blocks (Model)

#### Inputs

| Name        | Type   | Description                                             |
| ----------- | ------ | ------------------------------------------------------- |
| `MODEL`     | MODEL  | The model to patch.                                     |
| `W_block_*` | FLOAT  | Weight multiplier for each SDXL UNet block (various).  |
| `B_block_*` | FLOAT  | Bias multiplier for each SDXL UNet block (various).    |

#### Effect

Direct control over every major SDXL UNet block for highly granular tweaking.

---

## 💡 Tips

- Use **Donut Detailer 2** for the closest mimic of Supermerger’s “Adjust.”  
- Use **Donut Detailer 4** for straightforward weight/bias multipliers.  
- Use **LoRA 6** to save custom-scaled LoRA patches.  
- Use **XL Blocks** when you need per-block control in SDXL.

---

## 📜 License

MIT © DonutsDelivery


# Donut Clip Encode (Mix Only)

A custom ComfyUI node for Stable Diffusion XL that gives you both **Mix Mode** and **Strength Mode** to control how the two CLIP branches ("g" and "l") combine.

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

| Name                      | Type    | Description                                                                                              |
|---------------------------|---------|----------------------------------------------------------------------------------------------------------|
| `clip`                    | CLIP    | The CLIP encoder from your SDXL pipeline.                                                                |
| `width` / `height`        | INT     | Base image resolution. Scaled internally by **size_cond_factor**.                                        |
| `text_g` / `text_l`       | STRING  | "Guidance" and "Language" prompts for the two CLIP branches.                                         |
| `mode`                    | ENUM    | `Mix Mode` or `Strength Mode`. Toggles which controls are active.                                        |
| `clip_gl_mix`             | FLOAT   | (0.0–1.0) Mix slider for all mix-based presets when in **Mix Mode**.                                     |
| `vs_mix`                  | FLOAT   | (0.0–1.0) Ratio slider for **Split vs Pooled** preset in **Mix Mode**.                                   |
| `clip_g_strength`         | FLOAT   | Strength for branch **g** (used only in **Strength Mode**).                                              |
| `clip_l_strength`         | FLOAT   | Strength for branch **l** (used only in **Strength Mode**).                                              |
| `strength_default`        | FLOAT   | Weight for **Default** embedding (only for **Strength Blend** preset in **Mix Mode**).                    |
| `strength_split`          | FLOAT   | Weight for **Split Only** embedding (only for **Strength Blend** preset).                                 |
| `strength_continuous`     | FLOAT   | Weight for **Continuous** embedding (only for **Strength Blend** preset).                                |
| `preset`                  | ENUM    | In **Mix Mode**, choose one of: `Default`, `Split Only`, `Continuous`, `Split vs Pooled`,                  |
|                           |         | `Split vs Continuous`, `Default vs Split`, `Default vs Continuous`, or `Strength Blend`.                  |
| `size_cond_factor`        | INT     | Upscale factor for internal CLIP resolution.                                                             |
| `layer_idx`               | INT     | CLIP layer index to stop at (e.g. `-2` for penultimate).                                                |

---

## 🎛️ Modes & Presets

### 🔀 Mix Mode
Uses `clip_gl_mix`, `vs_mix`, and `preset` to control blending.

| Preset                   | Behavior                                                                                      |
|--------------------------|-----------------------------------------------------------------------------------------------|
| **Default**              | Single-pass joint encode.                                                                     |
| **Split Only**           | Two-pass encode, linear blend by `clip_gl_mix`.                                                |
| **Continuous**           | Gamma-blended joint↔split via `clip_gl_mix^(1/3)`.                                              |
| **Split vs Pooled**      | Split sequence full, gamma blend pooled summary by `vs_mix^0.3`.                               |
| **Split vs Continuous**  | Linear blend between split-only and continuous by `clip_gl_mix`.                                |
| **Default vs Split**     | Linear blend joint vs split-only by `clip_gl_mix`.                                              |
| **Default vs Continuous**| Linear blend joint vs continuous by `clip_gl_mix`.                                              |
| **Strength Blend**       | Blend three embeddings (Default/Split/Continuous) by `strength_default`, `strength_split`,     |
|                          | `strength_continuous` (normalized weights).                                                   |

### 🏋️ Strength Mode
Uses `clip_g_strength` and `clip_l_strength` directly to weight the two branches in a single pass:
```python
cond = cond_g * clip_g_strength + cond_l * clip_l_strength
pooled = pooled_g * clip_g_strength + pooled_l * clip_l_strength
```

---

## 💡 Tips

- In **Mix Mode**, try **Split Only** at `clip_gl_mix=0.5` for an even split.  
- In **Continuous**, raise `clip_gl_mix > 0.7` for strong split bias.  
- **Strength Mode** is handy for direct CFG control per branch.  
- **Strength Blend** lets you combine all three core embeddings with custom weights.

---

## 📜 License
MIT © DonutsDelivery


