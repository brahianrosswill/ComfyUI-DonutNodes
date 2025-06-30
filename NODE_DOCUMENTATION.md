# Node Parameter Documentation

## DonutClipEncode Node

The `DonutClipEncode` node is designed to process text prompts using a CLIP model, offering advanced control over how the global (text_g) and local (text_l) parts of the prompt are encoded and mixed. It provides different modes and presets to fine-tune the conditioning output.

**Class Type:** `CLIP`
**Category:** `essentials`

### Input Parameters:

1.  **`clip`**:
    *   **Type**: `CLIP`
    *   **Description**: The CLIP model to use for encoding.
    *   **Practical Use**: This is the foundational model that interprets the text prompts.

2.  **`width`**:
    *   **Type**: `INT`
    *   **Default**: `1024`
    *   **Min**: `0`
    *   **Max**: `MAX_RESOLUTION` (ComfyUI's max resolution)
    *   **Description**: The target width for the conditioning. This width is internally multiplied by `size_cond_factor`.
    *   **Practical Use**: Defines the horizontal dimension of the latent space conditioning. Affects the composition and aspect ratio of the generated image.

3.  **`height`**:
    *   **Type**: `INT`
    *   **Default**: `1024`
    *   **Min**: `0`
    *   **Max**: `MAX_RESOLUTION`
    *   **Description**: The target height for the conditioning. This height is internally multiplied by `size_cond_factor`.
    *   **Practical Use**: Defines the vertical dimension of the latent space conditioning. Affects the composition and aspect ratio.

4.  **`text_g`**:
    *   **Type**: `STRING`
    *   **Multiline**: True
    *   **Dynamic Prompts**: True
    *   **Description**: The global prompt text. This typically corresponds to the first part of an SDXL prompt (e.g., before the comma).
    *   **Practical Use**: Used for the primary, overarching concepts and styles in the desired image.

5.  **`text_l`**:
    *   **Type**: `STRING`
    *   **Multiline**: True
    *   **Dynamic Prompts**: True
    *   **Description**: The local prompt text. This typically corresponds to the second part of an SDXL prompt (e.g., after the comma, often used for details or secondary elements).
    *   **Practical Use**: Used for specific details, secondary subjects, or stylistic nuances that complement the global prompt.

6.  **`mode`**:
    *   **Type**: `LIST` (Dropdown)
    *   **Options**: `["Mix Mode", "Strength Mode"]`
    *   **Default**: (Not specified, likely the first option: "Mix Mode")
    *   **Description**: Determines the primary method for combining `text_g` and `text_l` embeddings.
    *   **Practical Use**:
        *   **`Mix Mode`**: Uses presets and mix sliders (`clip_gl_mix`, `vs_mix`) to blend different encoding strategies (joint, split, continuous). Offers more nuanced control over the blend.
        *   **`Strength Mode`**: Directly applies strength multipliers (`clip_g_strength`, `clip_l_strength`) to the `text_g` and `text_l` embeddings before summing them. Offers a more straightforward way to emphasize one prompt over the other.

7.  **`clip_gl_mix`**:
    *   **Type**: `FLOAT`
    *   **Default**: `0.5`
    *   **Min**: `0.0`
    *   **Max**: `1.0`
    *   **Step**: `0.01`
    *   **Description**: (Mix Mode Only) Controls the mix ratio between `text_g` and `text_l` for the "Split Only" and "Continuous" branches. A value of 0 uses only `text_g`, 1 uses only `text_l`, and 0.5 is an even mix.
    *   **Practical Use**: Fine-tunes the balance between the global and local prompts within specific blending presets.

8.  **`vs_mix`**:
    *   **Type**: `FLOAT`
    *   **Default**: `0.5`
    *   **Min**: `0.0`
    *   **Max**: `1.0`
    *   **Step**: `0.01`
    *   **Description**: (Mix Mode, "Split vs Pooled" preset Only) Controls the mix between the pooled output of the "Split Only" branch and the "Joint" (pooled_joint) branch. The mix is alpha-blended with `alpha = vs_mix ** 0.3`.
    *   **Practical Use**: Specifically adjusts the influence of the joint prompt's pooled output versus the split prompt's pooled output when using the "Split vs Pooled" preset.

9.  **`clip_g_strength`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `0.0`
    *   **Max**: `1000.0`
    *   **Step**: `0.01`
    *   **Description**: (Strength Mode Only) Multiplier for the `text_g` embedding.
    *   **Practical Use**: Increases or decreases the influence of the global prompt. Values greater than 1.0 amplify its effect, while values less than 1.0 diminish it.

10. **`clip_l_strength`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `0.0`
    *   **Max**: `1000.0`
    *   **Step**: `0.01`
    *   **Description**: (Strength Mode Only) Multiplier for the `text_l` embedding.
    *   **Practical Use**: Increases or decreases the influence of the local prompt.

11. **`strength_default`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `0.0`
    *   **Max**: `10.0`
    *   **Step**: `0.01`
    *   **Description**: (Mix Mode, "Strength Blend" preset Only) Weight for the "Default" (joint) encoding in the blend.
    *   **Practical Use**: Controls the contribution of the standard joint encoding when using the "Strength Blend" preset.

12. **`strength_split`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `0.0`
    *   **Max**: `10.0`
    *   **Step**: `0.01`
    *   **Description**: (Mix Mode, "Strength Blend" preset Only) Weight for the "Split Only" encoding in the blend.
    *   **Practical Use**: Controls the contribution of the split encoding (g/l mixed by `clip_gl_mix`) when using the "Strength Blend" preset.

13. **`strength_continuous`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `0.0`
    *   **Max**: `10.0`
    *   **Step**: `0.01`
    *   **Description**: (Mix Mode, "Strength Blend" preset Only) Weight for the "Continuous" encoding in the blend.
    *   **Practical Use**: Controls the contribution of the gamma-biased continuous encoding when using the "Strength Blend" preset.

14. **`preset`**:
    *   **Type**: `LIST` (Dropdown)
    *   **Options**: `["Default", "Split Only", "Continuous", "Split vs Pooled", "Split vs Continuous", "Default vs Split", "Default vs Continuous", "Strength Blend"]`
    *   **Default**: (Not specified, likely "Default")
    *   **Description**: (Mix Mode Only) Selects a specific strategy for combining the `text_g` and `text_l` embeddings.
    *   **Practical Use**:
        *   **`Default`**: Uses joint encoding of `text_g` and `text_l`. Standard SDXL behavior.
        *   **`Split Only`**: Encodes `text_g` and `text_l` separately and then mixes their conditionings based on `clip_gl_mix`.
        *   **`Continuous`**: A gamma-biased mix between the joint encoding and the split encoding, influenced by `clip_gl_mix`.
        *   **`Split vs Pooled`**: Uses the split conditioning but blends its pooled output with the joint pooled output, controlled by `vs_mix`.
        *   **`Split vs Continuous`**: Mixes the "Split Only" and "Continuous" encodings based on `clip_gl_mix`.
        *   **`Default vs Split`**: Mixes the "Default" (joint) and "Split Only" encodings based on `clip_gl_mix`.
        *   **`Default vs Continuous`**: Mixes the "Default" (joint) and "Continuous" encodings based on `clip_gl_mix`.
        *   **`Strength Blend`**: Normalizes and blends "Default", "Split Only", and "Continuous" encodings based on `strength_default`, `strength_split`, and `strength_continuous` weights.

15. **`size_cond_factor`**:
    *   **Type**: `INT`
    *   **Default**: `4`
    *   **Min**: `1`
    *   **Max**: `16`
    *   **Description**: Factor by which the input `width` and `height` are multiplied for conditioning.
    *   **Practical Use**: Influences the scale of conditioning features. Higher values mean the conditioning is effectively done at a larger resolution before being applied to the generation process. This is related to how SDXL handles resolution conditioning.

16. **`layer_idx`**:
    *   **Type**: `INT`
    *   **Default**: `-2`
    *   **Min**: `-33`
    *   **Max**: `33`
    *   **Description**: Specifies the CLIP layer from which to extract embeddings. Negative values count from the end.
    *   **Practical Use**: Using earlier layers (e.g., -1, -2) often captures more semantic meaning, while deeper layers might capture more abstract features. `-2` is a common default for SDXL.

### Output:

*   **`CONDITIONING`**: The resulting conditioning tensor, ready to be used by a KSampler or similar node. It includes the final conditioning and pooled output, along with the adjusted width and height.

### Internal Logic Summary:

1.  **Upscale Resolution**: `width` and `height` are multiplied by `size_cond_factor`.
2.  **Prepare CLIP**: The CLIP model is cloned, and processing is stopped at the specified `layer_idx`.
3.  **Tokenize & Pad Prompts**: `text_g` and `text_l` are tokenized. If one is empty, empty tokens are used. Tokens are padded to the same sequence length.
4.  **Joint Encoding**: `text_g` and `text_l` are encoded together (`cond_joint`, `pooled_joint`).
5.  **Split-Only Branch**: `text_g` and `text_l` are encoded separately and then mixed using `clip_gl_mix` (`cond_split`, `pooled_split`).
6.  **Continuous Branch**: A gamma-biased (exponent `3.0`) mix between `cond_joint` and `cond_split` using `clip_gl_mix` (`cond_cont`, `pooled_cont`).
7.  **Choose Base Embedding**:
    *   If **`Strength Mode`**: `base_cond = cond_g * clip_g_strength + cond_l * clip_l_strength`.
    *   If **`Mix Mode`**: The `base_cond` and `base_pooled` are selected based on the chosen `preset`, using the results from steps 4, 5, and 6, and parameters like `clip_gl_mix`, `vs_mix`, `strength_default`, `strength_split`, `strength_continuous`.
8.  **Auto-adjust CFG**: Unless in `Mix Mode` with `preset == 'Default'`, the norm of the `base_cond` is scaled to match the norm of `cond_joint`. This helps maintain consistent guidance strength across different presets.
9.  **Wrap and Return**: The final conditioning and pooled output are packaged with the adjusted width and height.

This node offers a sophisticated way to blend and control prompt embeddings, allowing for nuanced outputs beyond simple concatenation or weighting.

## DonutDetailer Node

The `DonutDetailer` node modifies an SDXL model by applying adjustments to specific parameter groups: the input block, output block 0, and output block 2. It allows for scaling and offsetting of weights and biases in these blocks.

**Class Type:** `MODEL`
**Category:** `Model Patches`

### Input Parameters:

1.  **`model`**:
    *   **Type**: `MODEL`
    *   **Description**: The SDXL model to be patched.
    *   **Practical Use**: This is the base model whose parameters will be modified. The node clones the model to ensure changes are localized.

2.  **`Scale_in`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `-100.0`
    *   **Max**: `100.0`
    *   **Step**: `0.001`
    *   **Description**: A scaling factor primarily used in the input block calculations.
    *   **Practical Use**: Modulates the intensity of `Weight_in` and `Bias_in` effects on the input block.

3.  **`Weight_in`**:
    *   **Type**: `FLOAT`
    *   **Default**: `0.0`
    *   **Min**: `-10.0`
    *   **Max**: `10.0`
    *   **Step**: `0.001`
    *   **Description**: Influences the weight parameters of the input block (e.g., `input_blocks.0.0.weight` or `diffusion_model.input_blocks.0.0.weight`).
    *   **Formula**: `weight = weight * (1 - Scale_in * Weight_in)`
    *   **Practical Use**: Adjusts the weights of the initial convolution layer. A `Weight_in` of `0.0` (default) results in no change to the weight from this term. Positive values decrease the weight magnitude (if `Scale_in` is positive), potentially softening initial features.

4.  **`Bias_in`**:
    *   **Type**: `FLOAT`
    *   **Default**: `0.0`
    *   **Min**: `-10.0`
    *   **Max**: `10.0`
    *   **Step**: `0.001`
    *   **Description**: Influences the bias parameters of the input block (e.g., `input_blocks.0.0.bias` or `diffusion_model.input_blocks.0.0.bias`).
    *   **Formula**: `bias = bias * (1 + Scale_in * Bias_in)`
    *   **Practical Use**: Adjusts the biases of the initial convolution layer. A `Bias_in` of `0.0` (default) results in no change to the bias from this term. Positive values increase the bias (if `Scale_in` is positive), potentially shifting the activation levels.

5.  **`Scale_out0`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `-100.0`
    *   **Max**: `100.0`
    *   **Step**: `0.001`
    *   **Description**: A scaling factor primarily used in the output block 0 calculations.
    *   **Practical Use**: Modulates the intensity of `Weight_out0` and `Bias_out0` effects on output block 0.

6.  **`Weight_out0`**:
    *   **Type**: `FLOAT`
    *   **Default**: `0.0`
    *   **Min**: `-10.0`
    *   **Max**: `10.0`
    *   **Step**: `0.001`
    *   **Description**: Influences the weight parameters of output block 0 (e.g., `out.0.weight` or `diffusion_model.out.0.weight`).
    *   **Formula**: `weight = weight * (1 - Scale_out0 * Weight_out0)`
    *   **Practical Use**: Adjusts the weights of the first part of the final output projection. Similar to `Weight_in`, a `0.0` value means no change from this term.

7.  **`Bias_out0`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `-10.0`
    *   **Max**: `10.0`
    *   **Step**: `0.001`
    *   **Description**: Influences the bias parameters of output block 0 (e.g., `out.0.bias` or `diffusion_model.out.0.bias`).
    *   **Formula**: `bias = bias * (Scale_out0 * Bias_out0)`
    *   **Practical Use**: Adjusts the biases of the first part of the final output projection. Unlike other bias parameters, the default of `1.0` (with `Scale_out0=1.0`) maintains the original bias. Values other than `1.0` will scale the original bias.

8.  **`Scale_out2`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `-100.0`
    *   **Max**: `100.0`
    *   **Step**: `0.001`
    *   **Description**: A scaling factor primarily used in the output block 2 calculations.
    *   **Practical Use**: Modulates the intensity of `Weight_out2` and `Bias_out2` effects on output block 2.

9.  **`Weight_out2`**:
    *   **Type**: `FLOAT`
    *   **Default**: `0.0`
    *   **Min**: `-10.0`
    *   **Max**: `10.0`
    *   **Step**: `0.001`
    *   **Description**: Influences the weight parameters of output block 2 (e.g., `out.2.weight` or `diffusion_model.out.2.weight`).
    *   **Formula**: `weight = weight * (1 - Scale_out2 * Weight_out2)`
    *   **Practical Use**: Adjusts the weights of the second part (final layer) of the output projection. Similar to `Weight_in`, a `0.0` value means no change from this term.

10. **`Bias_out2`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `-10.0`
    *   **Max**: `10.0`
    *   **Step**: `0.001`
    *   **Description**: Influences the bias parameters of output block 2 (e.g., `out.2.bias` or `diffusion_model.out.2.bias`).
    *   **Formula**: `bias = bias * (Scale_out2 * Bias_out2)`
    *   **Practical Use**: Adjusts the biases of the second part (final layer) of the output projection. The default of `1.0` (with `Scale_out2=1.0`) maintains the original bias.

### Output:

*   **`MODEL`**: The patched SDXL model with modified input and output block parameters.

### Behavior with Default Values:

*   **Input Block**:
    *   `Weight_in = 0.0`: `weight * (1 - Scale_in * 0)` => `weight * 1` (no change)
    *   `Bias_in = 0.0`: `bias * (1 + Scale_in * 0)` => `bias * 1` (no change)
*   **Output Block 0**:
    *   `Weight_out0 = 0.0`: `weight * (1 - Scale_out0 * 0)` => `weight * 1` (no change)
    *   `Bias_out0 = 1.0`: `bias * (Scale_out0 * 1.0)`. If `Scale_out0` is `1.0` (default), then `bias * 1` (no change).
*   **Output Block 2**:
    *   `Weight_out2 = 0.0`: `weight * (1 - Scale_out2 * 0)` => `weight * 1` (no change)
    *   `Bias_out2 = 1.0`: `bias * (Scale_out2 * 1.0)`. If `Scale_out2` is `1.0` (default), then `bias * 1` (no change).

With all default values, the node effectively acts as a bypass, making no changes to the model parameters. Deviating from these defaults allows for fine-tuning of the model's initial feature processing and final output generation stages. The prefixes for parameter names (`input_blocks.0.0.`, `out.0.`, `out.2.` vs. `diffusion_model.input_blocks.0.0.`, etc.) are determined automatically by inspecting the model's parameter names.

## DonutDetailer2 Node

The `DonutDetailer2` node adjusts parameters in an SDXL model, specifically targeting an input block and two output blocks (0 and 2). It uses a set of multipliers (`Multiplier_in/out0/out2`, `S1_in/out0/out2`, `S2_in/out0/out2`) to modify weights and biases based on predefined formulas.

**Class Type:** `MODEL`
**Category:** `Model Patches`

### Input Parameters:

1.  **`model`**:
    *   **Type**: `MODEL`
    *   **Description**: The SDXL model to be patched.
    *   **Practical Use**: The base model that will be cloned and modified.

2.  **`Multiplier_in`** (K_in):
    *   **Type**: `FLOAT`
    *   **Default**: `0.0`
    *   **Min**: `-100.0`
    *   **Max**: `100.0`
    *   **Step**: `0.01`
    *   **Description**: General multiplier for the input block adjustments.
    *   **Practical Use**: Scales the overall effect of `S1_in` and `S2_in` on the input block. A value of `0.0` (default) disables adjustments for this block.

3.  **`S1_in`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `-100.0`
    *   **Max**: `100.0`
    *   **Step**: `0.01`
    *   **Description**: Factor used in the weight multiplier formula for the input block.
    *   **Formula (Weight Multiplier)**: `1 - (Multiplier_in * S1_in * 0.01)`
    *   **Practical Use**: Modifies weights in the input block. With `Multiplier_in = 0.0`, this has no effect.

4.  **`S2_in`**:
    *   **Type**: `FLOAT`
    *   **Default**: `2.0` (*Note: Docstring implies default of 0 for bypass, but code default is 2.0. Assuming 0.0 for bypass effect based on docstring intent.* If `Multiplier_in` is 0, this value doesn't matter for bypass.)
    *   **Min**: `-100.0`
    *   **Max**: `100.0`
    *   **Step**: `0.01`
    *   **Description**: Factor used in the bias multiplier formula for the input block.
    *   **Formula (Bias Multiplier)**: `1 + (Multiplier_in * S2_in * 0.02)`
    *   **Practical Use**: Modifies biases in the input block. With `Multiplier_in = 0.0`, this has no effect.

5.  **`Multiplier_out0`** (K_out0):
    *   **Type**: `FLOAT`
    *   **Default**: `0.0`
    *   **Min**: `-100.0`
    *   **Max**: `100.0`
    *   **Step**: `0.01`
    *   **Description**: General multiplier for output block 0 adjustments.
    *   **Practical Use**: Scales the overall effect of `S1_out0` and `S2_out0` on output block 0. A value of `0.0` (default) disables adjustments for this block.

6.  **`S1_out0`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0` (*Note: Docstring implies default of 0 for bypass. Assuming 0.0 for bypass effect with `Multiplier_out0=0.0`.*)
    *   **Min**: `-100.0`
    *   **Max**: `100.0`
    *   **Step**: `0.01`
    *   **Description**: Factor used in the weight multiplier formula for output block 0.
    *   **Formula (Weight Multiplier)**: `1 - (Multiplier_out0 * S1_out0 * 0.01)`
    *   **Practical Use**: Modifies weights in output block 0.

7.  **`S2_out0`**:
    *   **Type**: `FLOAT`
    *   **Default**: `2.0` (*Note: Docstring implies default of 1 for bypass. Assuming 0.0 or that `Multiplier_out0=0.0` achieves bypass.*)
    *   **Min**: `-100.0`
    *   **Max**: `100.0`
    *   **Step**: `0.01`
    *   **Description**: Factor used in the bias multiplier formula for output block 0.
    *   **Formula (Bias Multiplier)**: `1 + (Multiplier_out0 * S2_out0 * 0.02)`
    *   **Practical Use**: Modifies biases in output block 0.

8.  **`Multiplier_out2`** (K_out2):
    *   **Type**: `FLOAT`
    *   **Default**: `0.0`
    *   **Min**: `-100.0`
    *   **Max**: `100.0`
    *   **Step**: `0.01`
    *   **Description**: General multiplier for output block 2 adjustments.
    *   **Practical Use**: Scales the overall effect of `S1_out2` and `S2_out2` on output block 2. A value of `0.0` (default) disables adjustments for this block.

9.  **`S1_out2`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0` (*Note: Docstring implies default of 0 for bypass. Assuming 0.0 for bypass effect with `Multiplier_out2=0.0`.*)
    *   **Min**: `-100.0`
    *   **Max**: `100.0`
    *   **Step**: `0.01`
    *   **Description**: Factor used in the weight multiplier formula for output block 2.
    *   **Formula (Weight Multiplier)**: `1 - (Multiplier_out2 * S1_out2 * 0.01)`
    *   **Practical Use**: Modifies weights in output block 2.

10. **`S2_out2`**:
    *   **Type**: `FLOAT`
    *   **Default**: `2.0` (*Note: Docstring implies default of 1 for bypass. Assuming 0.0 or that `Multiplier_out2=0.0` achieves bypass.*)
    *   **Min**: `-100.0`
    *   **Max**: `100.0`
    *   **Step**: `0.01`
    *   **Description**: Factor used in the bias multiplier formula for output block 2.
    *   **Formula (Bias Multiplier)**: `1 + (Multiplier_out2 * S2_out2 * 0.02)`
    *   **Practical Use**: Modifies biases in output block 2.

### Output:

*   **`MODEL`**: The patched SDXL model.

### Formulas Applied:

For each targeted block (Input, Output 0, Output 2), the following calculations are made:

*   **Weight Multiplier**: `1 - (K * S1 * 0.01)`
    *   `param.data.mul_(weight_multiplier)`
*   **Bias Multiplier**: `1 + (K * S2 * 0.02)`
    *   `param.data.mul_(bias_multiplier)`

Where `K` is the respective `Multiplier_in/out0/out2`, `S1` is `S1_in/out0/out2`, and `S2` is `S2_in/out0/out2`.

### Behavior with Default `Multiplier_` Values:

If `Multiplier_in`, `Multiplier_out0`, and `Multiplier_out2` are all `0.0` (their defaults):
*   **Weight Multiplier**: `1 - (0 * S1 * 0.01) = 1 - 0 = 1`
*   **Bias Multiplier**: `1 + (0 * S2 * 0.02) = 1 + 0 = 1`
In this case, all weights and biases in the targeted blocks are multiplied by 1, resulting in no change (bypass effect), regardless of the `S1` and `S2` values.

The docstring mentions specific default values for `K`, `S1`, and `S2` for each block that would result in a bypass. However, the actual default input values for `Multiplier_in/out0/out2` are `0.0`, which already ensures a bypass. The `S1` and `S2` defaults in the code (`1.0` and `2.0` respectively) only come into play if their corresponding `Multiplier_` is non-zero.

This node allows for complex interactions by adjusting the `Multiplier`, `S1`, and `S2` values for each block, offering fine-grained control over the model's behavior at its input and output stages. The specific parameter name prefixes are determined automatically.

## DonutDetailer4 Node

The `DonutDetailer4` node provides a straightforward way to modify an SDXL model by directly multiplying the weights and biases of specific blocks (input block, output block 0, and output block 2) by user-defined values.

**Class Type:** `MODEL`
**Category:** `Model Patches`

### Input Parameters:

1.  **`model`**:
    *   **Type**: `MODEL`
    *   **Description**: The SDXL model to be patched.
    *   **Practical Use**: This is the base model that will be cloned and whose parameters will be directly scaled.

2.  **`Weight_in`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `-10.0`
    *   **Max**: `10.0`
    *   **Step**: `0.001`
    *   **Description**: Direct multiplier for the weights of the input block (e.g., `input_blocks.0.0.weight`).
    *   **Formula**: `weight = weight * Weight_in`
    *   **Practical Use**: Scales the weights of the initial convolution layer. A value of `1.0` (default) leaves the weights unchanged. Values greater than 1.0 amplify them, less than 1.0 diminish them, and negative values invert them.

3.  **`Bias_in`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `-10.0`
    *   **Max**: `10.0`
    *   **Step**: `0.001`
    *   **Description**: Direct multiplier for the biases of the input block (e.g., `input_blocks.0.0.bias`).
    *   **Formula**: `bias = bias * Bias_in`
    *   **Practical Use**: Scales the biases of the initial convolution layer. A value of `1.0` (default) leaves biases unchanged.

4.  **`Weight_out0`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `-10.0`
    *   **Max**: `10.0`
    *   **Step**: `0.001`
    *   **Description**: Direct multiplier for the weights of output block 0 (e.g., `out.0.weight`).
    *   **Formula**: `weight = weight * Weight_out0`
    *   **Practical Use**: Scales the weights of the first part of the final output projection.

5.  **`Bias_out0`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `-10.0`
    *   **Max**: `10.0`
    *   **Step**: `0.001`
    *   **Description**: Direct multiplier for the biases of output block 0 (e.g., `out.0.bias`).
    *   **Formula**: `bias = bias * Bias_out0`
    *   **Practical Use**: Scales the biases of the first part of the final output projection.

6.  **`Weight_out2`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `-10.0`
    *   **Max**: `10.0`
    *   **Step**: `0.001`
    *   **Description**: Direct multiplier for the weights of output block 2 (e.g., `out.2.weight`).
    *   **Formula**: `weight = weight * Weight_out2`
    *   **Practical Use**: Scales the weights of the second part (final layer) of the output projection.

7.  **`Bias_out2`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `-10.0`
    *   **Max**: `10.0`
    *   **Step**: `0.001`
    *   **Description**: Direct multiplier for the biases of output block 2 (e.g., `out.2.bias`).
    *   **Formula**: `bias = bias * Bias_out2`
    *   **Practical Use**: Scales the biases of the second part (final layer) of the output projection.

### Output:

*   **`MODEL`**: The patched SDXL model with directly scaled parameters in the specified blocks.

### Behavior with Default Values:

With all multiplier parameters at their default value of `1.0`, the node multiplies all targeted weights and biases by 1. This results in no change to the model parameters, effectively acting as a bypass.

This node offers a simpler, more direct method of parameter adjustment compared to `DonutDetailer` and `DonutDetailer2`, as it applies scaling factors directly without intermediate calculations or combined effects. The specific parameter name prefixes are determined automatically.

## DonutDetailerLoRA5 Node

The `DonutDetailerLoRA5` node is designed to apply direct multiplicative adjustments to the parameters of a LoRA (Low-Rank Adaptation) model. It targets weights and biases within the LoRA based on whether their names contain "down", "mid", or "up", allowing for block-specific modifications.

**Class Type:** `LoRA`
**Category:** `LoRA Patches`

### Input Parameters:

1.  **`lora`**:
    *   **Type**: `LoRA`
    *   **Description**: The LoRA model (as a dictionary containing the actual LoRA weights under the key `"lora"`) to be patched.
    *   **Practical Use**: This is the LoRA whose parameters will be scaled. The node creates a deep copy to ensure changes are localized.

2.  **`Weight_down`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.1`
    *   **Min**: `-10.0`
    *   **Max**: `10.0`
    *   **Step**: `0.001`
    *   **Description**: Multiplier for LoRA parameters that are weights and have "down" in their name.
    *   **Formula**: `lora_param = lora_param * Weight_down`
    *   **Practical Use**: Scales the weights of the "down" projection layers within the LoRA. Values other than `1.0` will alter their influence.

3.  **`Bias_down`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `-10.0`
    *   **Max**: `10.0`
    *   **Step**: `0.001`
    *   **Description**: Multiplier for LoRA parameters that are biases and have "down" in their name.
    *   **Formula**: `lora_param = lora_param * Bias_down`
    *   **Practical Use**: Scales the biases of the "down" projection layers within the LoRA.

4.  **`Weight_mid`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.1`
    *   **Min**: `-10.0`
    *   **Max**: `10.0`
    *   **Step**: `0.001`
    *   **Description**: Multiplier for LoRA parameters that are weights and have "mid" in their name.
    *   **Formula**: `lora_param = lora_param * Weight_mid`
    *   **Practical Use**: Scales the weights of "middle" block LoRA layers, if any are named this way.

5.  **`Bias_mid`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `-10.0`
    *   **Max**: `10.0`
    *   **Step**: `0.001`
    *   **Description**: Multiplier for LoRA parameters that are biases and have "mid" in their name.
    *   **Formula**: `lora_param = lora_param * Bias_mid`
    *   **Practical Use**: Scales the biases of "middle" block LoRA layers.

6.  **`Weight_up`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.1`
    *   **Min**: `-10.0`
    *   **Max**: `10.0`
    *   **Step**: `0.001`
    *   **Description**: Multiplier for LoRA parameters that are weights and have "up" in their name.
    *   **Formula**: `lora_param = lora_param * Weight_up`
    *   **Practical Use**: Scales the weights of the "up" projection layers within the LoRA.

7.  **`Bias_up`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `-10.0`
    *   **Max**: `10.0`
    *   **Step**: `0.001`
    *   **Description**: Multiplier for LoRA parameters that are biases and have "up" in their name.
    *   **Formula**: `lora_param = lora_param * Bias_up`
    *   **Practical Use**: Scales the biases of the "up" projection layers within the LoRA.

### Output:

*   **`LoRA`**: The patched LoRA model (as a dictionary) with modified parameters.

### Behavior:

The node iterates through all parameters within the input `lora` (specifically, `lora["lora"]`).
- If a parameter name contains "down":
    - If it also contains "weight", it's multiplied by `Weight_down`.
    - If it also contains "bias", it's multiplied by `Bias_down`.
- If a parameter name contains "mid":
    - If it also contains "weight", it's multiplied by `Weight_mid`.
    - If it also contains "bias", it's multiplied by `Bias_mid`.
- If a parameter name contains "up":
    - If it also contains "weight", it's multiplied by `Weight_up`.
    - If it also contains "bias", it's multiplied by `Bias_up`.

Parameters not matching these conditions (e.g., not containing "down", "mid", or "up", or not being identified as a "weight" or "bias" through string matching) are left unchanged.
The default values of `1.1` for weights and `1.0` for biases suggest a slight emphasis on weight adjustments by default, while biases are initially unchanged.
This node allows for targeted adjustments within a LoRA, potentially fine-tuning its effect on different parts of the model it's applied to.

## DonutDetailerXLBlocks Node

The `DonutDetailerXLBlocks` node provides granular control over an SDXL model by allowing direct multiplication of weights and biases for individual blocks within the U-Net architecture. It dynamically creates input parameters for many distinct blocks.

**Class Type:** `MODEL`
**Category:** `Model Patches`

### Input Parameters:

1.  **`model`**:
    *   **Type**: `MODEL`
    *   **Description**: The SDXL model to be patched.
    *   **Practical Use**: The base model that will be cloned. Its parameters in specified blocks will be scaled by the corresponding input values.

2.  **Dynamically Generated Block Parameters**:
    The node defines parameters for several groups of blocks. For each block in these groups, two float parameters are created: `[block_name]_weight` and `[block_name]_bias`.

    *   **Block Groups and Naming:**
        *   **`input_blocks`**: 9 blocks, named `input_blocks_0` through `input_blocks_8`.
            *   Parameters: `input_blocks_0_weight`, `input_blocks_0_bias`, ..., `input_blocks_8_weight`, `input_blocks_8_bias`.
        *   **`middle_block`**: 3 blocks, named `middle_block_0` through `middle_block_2`.
            *   Parameters: `middle_block_0_weight`, `middle_block_0_bias`, ..., `middle_block_2_weight`, `middle_block_2_bias`.
        *   **`output_blocks`**: 9 blocks, named `output_blocks_0` through `output_blocks_8`.
            *   Parameters: `output_blocks_0_weight`, `output_blocks_0_bias`, ..., `output_blocks_8_weight`, `output_blocks_8_bias`.
        *   **`out`**: 1 block, named `out`.
            *   Parameters: `out_weight`, `out_bias`.

    *   **Common Properties for each Dynamic Parameter (`*_weight`, `*_bias`):**
        *   **Type**: `FLOAT`
        *   **Default**: `1.0`
        *   **Min**: `-10.0`
        *   **Max**: `10.0`
        *   **Step**: `0.001`
        *   **Description**: Direct multiplier for the weights or biases of the specified block.
        *   **Formula**:
            *   `block_weight_tensor = block_weight_tensor * [block_name]_weight`
            *   `block_bias_tensor = block_bias_tensor * [block_name]_bias`
        *   **Practical Use**: Allows individual scaling of weights and biases for almost every block in the SDXL U-Net. A value of `1.0` (default) leaves the parameters of that specific block unchanged. This offers extremely fine-grained control over the model's characteristics at different stages of processing.

### Output:

*   **`MODEL`**: The patched SDXL model with modified parameters according to the numerous input multipliers.

### Behavior:

The node iterates through the parameters of the input model. It checks if a parameter's name starts with a prefix corresponding to one of the defined blocks (e.g., `input_blocks.0.`, `middle_block.1.`, `output_blocks.8.`, `out.`).
*   If the parameter name ends with `.weight`, its data is multiplied by the corresponding `[block_name]_weight` input value.
*   If the parameter name ends with `.bias`, its data is multiplied by the corresponding `[block_name]_bias` input value.

The prefixes for parameter names (e.g., `diffusion_model.input_blocks.0.` vs. `input_blocks.0.`) are determined automatically by inspecting the model's parameter names.

With all dynamically generated parameters at their default value of `1.0`, the node effectively acts as a bypass, making no changes to the model. Deviating from these defaults allows for highly specific adjustments throughout the U-Net architecture. This can be used to subtly or drastically alter the model's learned features and how they are processed at each level of the network.

## DonutWidenMergeUNet and DonutWidenMergeCLIP Nodes

These two nodes implement an advanced model merging technique called "WIDEN" (Weight-Importance Driven Ensemble for Neural networks) specifically adapted for SDXL U-Net models (`DonutWidenMergeUNet`) and CLIP text encoder models (`DonutWidenMergeCLIP`). They merge multiple input models into a base model, using layer-specific importance calculations and thresholds to determine how parameters are combined. Both nodes share a similar set of core parameters controlling the merge process and can accept up to 12 models/CLIPs in total (1 base + 1 other required + 10 optional).

**Common Category:** `donut/merge`

---

### DonutWidenMergeUNet

**Class Type:** `MODEL`

**Inputs:**
*   `model_base`: (`MODEL`) The primary model into which others are merged.
*   `model_other`: (`MODEL`) The second model to be merged.
*   `model_3` through `model_12`: (Optional `MODEL`) Additional models to include in the merge.
*   *(Common merge parameters listed below)*

**Outputs:**
*   `model`: (`MODEL`) The resulting merged U-Net model.
*   `merge_results`: (`STRING`) A textual summary of the merge process, including layer-wise statistics.

---

### DonutWidenMergeCLIP

**Class Type:** `CLIP`

**Inputs:**
*   `clip_base`: (`CLIP`) The primary CLIP model into which others are merged.
*   `clip_other`: (`CLIP`) The second CLIP model to be merged.
*   `clip_3` through `clip_12`: (Optional `CLIP`) Additional CLIP models to include in the merge.
*   *(Common merge parameters listed below)*

**Outputs:**
*   `clip`: (`CLIP`) The resulting merged CLIP model.
*   `merge_results`: (`STRING`) A textual summary of the merge process.

---

### Common Merge Parameters (for both nodes):

1.  **`merge_strength`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `0.1`
    *   **Max**: `3.0`
    *   **Step**: `0.1`
    *   **Description**: Controls the overall intensity of the merge. It scales the calculated difference (delta) between the merged parameter and the base parameter before adding it back to the base parameter.
    *   **Formula (simplified)**: `final_param = base_param + (merged_param_from_WIDEN - base_param) * merge_strength`
    *   **Practical Use**: A value of `1.0` applies the WIDEN merge as calculated. Values less than `1.0` reduce the impact of the merged models, while values greater than `1.0` can exaggerate their features (potentially leading to instability).

2.  **`widen_threshold`**:
    *   **Type**: `FLOAT`
    *   **Default**: `0.5`
    *   **Min**: `0.0`
    *   **Max**: `1.0`
    *   **Step**: `0.01`
    *   **Description**: A crucial parameter for the WIDEN algorithm. It dynamically adjusts the threshold for determining whether a parameter's delta (change from base) is significant enough to be included in the merge. The threshold is scaled exponentially:
        *   `0.0` is highly permissive (very small changes are merged).
        *   `0.5` is a balanced, standard threshold.
        *   `1.0` is highly selective (only very large changes are merged).
    *   **Practical Use**: Controls the "pickiness" of the merge. Lower values result in a more comprehensive merge, incorporating smaller details from other models. Higher values result in a more conservative merge, primarily taking only the most impactful changes. This helps prevent "parameter washing" where subtle details are lost.

3.  **`widen_calibration`**:
    *   **Type**: `FLOAT`
    *   **Default**: `0.0`
    *   **Min**: `0.0`
    *   **Max**: `1.0`
    *   **Step**: `0.01`
    *   **Description**: Affects the importance computation within the WIDEN algorithm. This value is mapped to a range (0.1 to 2.0) and influences how importance scores are weighted.
        *   `0.0` maps to `0.1x` (minimal importance weighting).
        *   `0.5` maps to `1.0x` (standard importance weighting).
        *   `1.0` maps to `2.0x` (maximum importance weighting).
    *   **Practical Use**: Fine-tunes how the calculated significance of parameter changes contributes to the final merged weights. Higher values give more weight to parameters deemed important by the ranking system.

4.  **`renorm_mode`**:
    *   **Type**: `LIST` (Dropdown)
    *   **Options**: `["magnitude", "calibrate", "none"]`
    *   **Default**: `magnitude`
    *   **Description**: Determines the method used to renormalize the merged parameters after they are combined.
    *   **Practical Use**:
        *   **`none`**: No renormalization is applied.
        *   **`magnitude`**: Simple magnitude preservation. The merged parameter is scaled so its L2 norm matches the L2 norm of the original base parameter.
        *   **`calibrate`**: A more conservative, calibration-style renormalization. It analyzes the delta (change) from the base parameter, applies softmax normalization to the absolute delta, and then uses this to adjust the delta before adding it back to the base. This method has its own internal temperature and scaling factors (fixed at `t=0.3`, `s=1.1` in the code for a more conservative effect).
        Renormalization helps to maintain the overall scale and stability of the model after merging.

5.  **`batch_size`**:
    *   **Type**: `INT`
    *   **Default**: `50` (UNet), `75` (CLIP)
    *   **Min**: `10`
    *   **Max**: `500`
    *   **Step**: `10`
    *   **Description**: This parameter seems to be a remnant or misnomer in the context of the "FULL ZERO-ACCUMULATION" WIDEN implementation provided. The current code processes parameters individually to minimize memory, not in batches. Its value might not have a direct effect on the merging logic itself but could be intended for future optimizations or different merge strategies within the class structure.
    *   **Practical Use**: Currently, likely no direct impact on the zero-accumulation WIDEN merge.

### Internal Logic Summary (WIDEN Merge):

The core `widen_merging_sdxl` function performs the following steps for each common parameter across the models:

1.  **Zero-Accumulation Loading**: Only the current parameter being processed is loaded from the base model and each model-to-merge to calculate deltas (differences from base). This drastically reduces memory usage.
2.  **Layer Classification & Metadata**: The parameter is classified (e.g., `time_embedding`, `cross_attention`, `input_conv`) and metadata like base magnitude and average delta magnitude are calculated.
3.  **Threshold Check**: The `should_merge_parameter` method uses `widen_threshold` and layer-specific base thresholds to decide if the parameter's delta is significant enough to merge. If not, the base parameter is kept, and the process skips to the next parameter.
4.  **Magnitude & Direction Computation**: If merging, the magnitude and direction components are computed for the base parameter and for each (base + delta) combination. Differences in these components (mag_diffs, dir_diffs) are calculated.
5.  **Significance Ranking**: `rank_significance_adaptive` ranks these differences.
6.  **Importance Computation**: `compute_importance_sdxl` uses these ranks, `widen_threshold`, and `widen_calibration` to calculate importance scores for magnitude and direction.
7.  **Parameter Merging**: `merge_single_parameter_sdxl` combines the deltas weighted by these importance scores and adds them to the base parameter.
8.  **Strength Application**: The `merge_strength` is applied to the change from the base.
9.  **Renormalization**: The chosen `renorm_mode` is applied.
10. **Update Target Model**: The final parameter is written to the target model.

The nodes also feature a cache (`_MERGE_CACHE`) to store and retrieve results of identical merge operations, preventing redundant processing. Memory monitoring and cleanup utilities are integrated to improve stability.

## DonutFillerModel and DonutFillerClip Nodes

These are utility nodes that provide placeholder or "stub" objects for `MODEL` and `CLIP` types, respectively. They are primarily used to fill optional inputs in other nodes when a full model or CLIP is not needed or desired for a particular workflow branch.

**Common Category:** `utils` (Based on `DonutWidenMerge.py`, though might appear under `donut/utils` or other utility categories in ComfyUI).

---

### DonutFillerModel

**Class Type:** `MODEL`

**Inputs:**
*   None. This node has no input parameters.

**Outputs:**
*   `model`: (`MODEL`) A lightweight filler model object.

**Practical Use:**
*   The output model is a stub object. It has `state_dict()` and `named_parameters()` methods that return empty structures.
*   It has an internal attribute `_is_filler` set to `True`. This allows other nodes (like `DonutWidenMergeUNet`) to identify and potentially ignore or handle these filler models differently (e.g., by not attempting to merge them).
*   Useful for connecting to optional `MODEL` inputs on nodes if you don't want to load an actual model for that input slot, ensuring the workflow can still run.

---

### DonutFillerClip

**Class Type:** `CLIP`

**Inputs:**
*   None. This node has no input parameters.

**Outputs:**
*   `clip`: (`CLIP`) A lightweight filler CLIP object.

**Practical Use:**
*   Similar to `DonutFillerModel`, the output CLIP is a stub object with empty `state_dict()` and `named_parameters()`.
*   It also has an internal attribute `_is_filler` set to `True`, allowing other nodes (like `DonutWidenMergeCLIP`) to recognize it.
*   Useful for connecting to optional `CLIP` inputs when a real CLIP model isn't required for a specific path in the workflow.

---

In essence, these nodes help maintain workflow integrity and flexibility by providing valid, but inert, objects for model and CLIP inputs, preventing errors when an optional input is intentionally left "empty" but still requires a connection.

## DonutLoRAStack Node

The `DonutLoRAStack` node allows users to define a stack of up to three LoRAs. Each LoRA in the stack can be individually enabled/disabled and configured with its own model weight, CLIP weight, and an optional block vector for fine-grained control (though the block vector is applied by `DonutApplyLoRAStack`).

**Class Type:** `CUSTOM` (Specific to ComfyUI custom nodes)
**Category:** `Comfyanonymous/LoRA`

### Input Parameters:

The node defines parameters for three LoRA slots (1, 2, and 3). Parameters for each slot `X` (where `X` is 1, 2, or 3) are:

1.  **`switch_X`**:
    *   **Type**: `LIST` (Dropdown)
    *   **Options**: `["Off", "On"]`
    *   **Default**: (Not specified, likely "Off")
    *   **Description**: Toggles whether the LoRA in this slot is active and included in the output stack.
    *   **Practical Use**: Easily enable or disable a specific LoRA in the stack without removing its configuration.

2.  **`lora_name_X`**:
    *   **Type**: `LIST` (Dropdown, populated with available LoRA file names)
    *   **Options**: `["None"] + list_of_loras`
    *   **Default**: (Not specified, likely "None")
    *   **Description**: The name of the LoRA file to be used for this slot.
    *   **Practical Use**: Selects the LoRA to apply. Must not be "None" if `switch_X` is "On".

3.  **`model_weight_X`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `-1000`
    *   **Max**: `1000`
    *   **Step**: `0.01`
    *   **Description**: The strength of the LoRA's effect on the U-Net model (diffusion model).
    *   **Practical Use**: Controls how much the LoRA alters the main model's weights. `1.0` is full strength, `0.0` is no effect, negative values invert the LoRA's effect.

4.  **`clip_weight_X`**:
    *   **Type**: `FLOAT`
    *   **Default**: `1.0`
    *   **Min**: `-1000`
    *   **Max**: `1000`
    *   **Step**: `0.01`
    *   **Description**: The strength of the LoRA's effect on the CLIP model (text encoder).
    *   **Practical Use**: Controls how much the LoRA alters the text encoder's weights. Similar scaling as `model_weight_X`.

5.  **`block_vector_X`**:
    *   **Type**: `STRING`
    *   **Default**: `"1,1,1,1,1,1,1,1,1,1,1,1"`
    *   **Placeholder**: `"12 comma-sep floats"` or `"optional"`
    *   **Description**: A string of 12 comma-separated float values. This vector is intended for use by an application node (like `DonutApplyLoRAStack`) to apply per-block weighting to the LoRA's U-Net modification.
    *   **Practical Use**: Allows differential weighting of the LoRA across 12 conceptual blocks of the U-Net (e.g., input, middle, output blocks). The exact mapping of these 12 values to U-Net layers is determined by the node that consumes this stack (e.g., `DonutApplyLoRAStack` using `LoraLoaderBlockWeight`). A string of all "1"s implies uniform weighting.

6.  **`lora_stack`** (Optional Input):
    *   **Type**: `LORA_STACK`
    *   **Description**: An existing LoRA stack to which the LoRAs defined in this node will be appended.
    *   **Practical Use**: Allows chaining multiple `DonutLoRAStack` nodes to build larger, more complex LoRA configurations. If not provided, a new stack is created.

### Output:

1.  **`lora_stack`**:
    *   **Type**: `LORA_STACK`
    *   **Description**: A list of tuples. Each tuple represents an active LoRA and contains: `(lora_name, model_weight, clip_weight, block_vector_string)`.
    *   **Practical Use**: This output is designed to be fed into a LoRA application node (e.g., `DonutApplyLoRAStack`) that knows how to interpret this structure.

2.  **`show_help`**:
    *   **Type**: `STRING`
    *   **Description**: A URL string pointing to help documentation for CR LoRA Stack nodes.
    *   **Value**: `"https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/LoRA-Nodes#cr-lora-stack"`

### Behavior:

The node initializes an empty list or takes an existing `lora_stack`. For each of the three slots:
* If `switch_X` is "On" and `lora_name_X` is not "None", it appends a tuple `(lora_name_X, model_weight_X, clip_weight_X, block_vector_X.strip())` to the stack.

The resulting list (stack) is then output. This node itself does not load or apply any LoRAs; it purely defines the configuration for them.

## DonutApplyLoRAStack Node

The `DonutApplyLoRAStack` node takes a model, a CLIP instance, and a LoRA stack (as produced by `DonutLoRAStack`) and applies each LoRA in the stack sequentially. It uses a specific two-step application process for each LoRA: first applying it to the U-Net model with block-specific weights, and then applying it to the CLIP model with a uniform weight.

**Class Type:** `CUSTOM`
**Category:** `Comfyanonymous/LoRA`

### Input Parameters:

1.  **`model`**:
    *   **Type**: `MODEL`
    *   **Description**: The base U-Net model to which the LoRAs will be applied.
    *   **Practical Use**: This model's weights will be modified by the LoRAs in the stack.

2.  **`clip`**:
    *   **Type**: `CLIP`
    *   **Description**: The base CLIP (text encoder) model to which the LoRAs will be applied.
    *   **Practical Use**: This CLIP model's weights will be modified by the LoRAs in the stack.

3.  **`lora_stack`**:
    *   **Type**: `LORA_STACK` (Typically from `DonutLoRAStack` or a compatible node)
    *   **Description**: A list of LoRA configurations. Each item in the list is expected to be a tuple: `(lora_name, model_weight, clip_weight, block_vector_string)`.
    *   **Practical Use**: Defines the sequence of LoRAs to apply and their respective strengths and block configurations. If the stack is empty or not provided, the node acts as a bypass.

### Output:

1.  **`model`**:
    *   **Type**: `MODEL`
    *   **Description**: The U-Net model after all LoRAs in the stack have been applied.
    *   **Practical Use**: The final, LoRA-modified diffusion model.

2.  **`clip`**:
    *   **Type**: `CLIP`
    *   **Description**: The CLIP model after all LoRAs in the stack have been applied.
    *   **Practical Use**: The final, LoRA-modified text encoder.

3.  **`show_help`**:
    *   **Type**: `STRING`
    *   **Description**: A URL string pointing to help documentation for CR Apply LoRA Stack nodes.
    *   **Value**: `"https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/LoRA-Nodes#cr-apply-lora-stack"`

### Behavior:

The node iterates through each LoRA configuration `(name, mw, cw, bv)` in the provided `lora_stack`:

1.  **Load LoRA**: The LoRA file specified by `name` is loaded from disk.
2.  **Prepare Block Vector**: The `bv` (block vector string) is processed. If it's empty, a default vector of 12 "1"s (e.g., "1,1,1,1,1,1,1,1,1,1,1,1") is used, implying uniform weighting across blocks for the U-Net.
3.  **Apply to U-Net (Model)**:
    *   The LoRA is applied to the current `unet` (model) using `LoraLoaderBlockWeight().load_lora_for_models`.
    *   **`strength_model`**: `mw` (model_weight from the stack item) is used.
    *   **`strength_clip`**: `0.0` is used. This is crucial: the CLIP component of the LoRA is *not* applied to the U-Net model during this step.
    *   **`block_vector`**: The prepared `vector` string is used, allowing per-block weighting for the U-Net.
4.  **Apply to CLIP (Text Encoder)**:
    *   The same LoRA is then applied again, this time primarily targeting the `text_enc` (CLIP model), using `comfy.sd.load_lora_for_models` (which typically applies LoRA uniformly).
    *   **`strength_model`**: `0.0` is used. The U-Net component of the LoRA is *not* applied to the U-Net model again in this step.
    *   **`strength_clip`**: `cw` (clip_weight from the stack item) is used.

This two-step process is repeated for every LoRA in the stack. The `unet` and `text_enc` variables are updated iteratively, so each LoRA is applied on top of the modifications from the previous ones.

This distinct application strategy (block-weighted for U-Net, uniform for CLIP, and applied in separate steps) is a key characteristic of this node, offering a particular way to combine LoRA effects.
