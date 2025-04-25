This is an experimental node I made to mimick the "adjust" in A1111 Supermerger (https://github.com/hako-mikan/sd-webui-supermerger?tab=readme-ov-file#adjust). It adds more noise and texture to the output, and can also adjust gamma/brightness using the last parameter. The 3 variants are fundamentally doing the same thing, but with different parameter controls. Scale/multiplier multiplies the S1 (weight) and S2 (bias) parameters. 

Donut Detailer: Initial try at making the node, dosent mimick supermerger accurately. 

![image](https://github.com/user-attachments/assets/b0477a38-86c2-42fd-a635-82afdef3b8a4)

Donut Detailer 2: Mimicks closes Supermerger Adjust parameters. 

![image](https://github.com/user-attachments/assets/6d0cc683-e005-481b-abe4-487700686df3)

Donut Detailer 4: Making it more barebone, without the coefficients. 

![image](https://github.com/user-attachments/assets/e1bafb1c-a24a-448e-92f9-f9e27f98157d)

Thanks to epiTune for helping me make this, and ChatGPT. Note: epiTune does not think this is the best solution to adding more texture as it is a a crude way of modifying the model, use it sparingly.

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


