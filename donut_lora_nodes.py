import comfy.sd
import comfy.utils
import folder_paths
import os
import re
import shutil
from .lora_block_weight import LoraLoaderBlockWeight

# CivitAI integration imports
try:
    from .shared.lora_hash import get_or_compute_hash
    from .shared.civitai_api import get_cache
    from .shared.config import get_civitai_api_key
    HAS_CIVITAI = True
except ImportError:
    HAS_CIVITAI = False
    def get_civitai_api_key(): return ""
    print("[DonutLoRAStack] CivitAI integration not available")

# ------------------------------------------------------------------------
# DonutLoRAStack: build up to 3 LoRAs with independent model & clip strengths + optional per-block vectors
# Now with CivitAI metadata integration!
# Block weight presets for different model architectures
# Format: "DisplayName:vector_values" - the part after : is the actual vector
BLOCK_PRESETS = [
    "None",  # Auto-detect or use individual block_vector fields
    "SDXL-ALL:" + ",".join(["1"] * 13),  # 1 base + 12 blocks
    "SD15-ALL:" + ",".join(["1"] * 18),  # 1 base + 17 blocks
    "ZIT-ALL:" + ",".join(["1"] * 31),   # 1 base + 30 layers (Z-Image Turbo)
    "FLUX-ALL:" + ",".join(["1"] * 58),  # 1 base + 57 blocks (19 double + 38 single)
]

# ------------------------------------------------------------------------
class DonutLoRAStack:
    class_type = "CUSTOM"
    aux_id     = "DonutsDelivery/ComfyUI-DonutNodes"

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        inputs = {
            "required": {
                "block_preset": (BLOCK_PRESETS, {"default": "None", "tooltip": "Apply preset block weights to all LoRAs. Overrides individual block_vector fields."}),

                "switch_1":       (["Off","On"],),
                "lora_name_1":    (loras,),
                "model_weight_1": ("FLOAT", {"default":1.0,"min":-1000,"max":1000,"step":0.01}),
                "clip_weight_1":  ("FLOAT", {"default":1.0,"min":-1000,"max":1000,"step":0.01}),
                "block_vector_1": ("STRING",{"default":"","placeholder":"SDXL:12, SD1.5:17, ZIT:30 blocks"}),

                "switch_2":       (["Off","On"],),
                "lora_name_2":    (loras,),
                "model_weight_2": ("FLOAT", {"default":1.0,"min":-1000,"max":1000,"step":0.01}),
                "clip_weight_2":  ("FLOAT", {"default":1.0,"min":-1000,"max":1000,"step":0.01}),
                "block_vector_2": ("STRING",{"default":"","placeholder":"SDXL:12, SD1.5:17, ZIT:30 blocks"}),

                "switch_3":       (["Off","On"],),
                "lora_name_3":    (loras,),
                "model_weight_3": ("FLOAT", {"default":1.0,"min":-1000,"max":1000,"step":0.01}),
                "clip_weight_3":  ("FLOAT", {"default":1.0,"min":-1000,"max":1000,"step":0.01}),
                "block_vector_3": ("STRING",{"default":"","placeholder":"SDXL:12, SD1.5:17, ZIT:30 blocks"}),

                "civitai_lookup": (["On", "Off"], {"default": "On", "tooltip": "Fetch LoRA info from CivitAI (requires API key in settings)"}),
            },
            "optional": {
                "lora_stack": ("LORA_STACK",),
            },
        }
        return inputs

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION     = "build_stack"
    CATEGORY     = "donut/LoRA"
    OUTPUT_NODE  = True

    def build_stack(
        self,
        block_preset,
        switch_1, lora_name_1, model_weight_1, clip_weight_1, block_vector_1,
        switch_2, lora_name_2, model_weight_2, clip_weight_2, block_vector_2,
        switch_3, lora_name_3, model_weight_3, clip_weight_3, block_vector_3,
        civitai_lookup="On",
        lora_stack=None
    ):
        stack = list(lora_stack) if lora_stack else []

        # Get preset vector (overrides individual block_vector fields if not "None")
        # Format is "NAME:vector" - extract the vector part after ':'
        preset_vector = ""
        if block_preset and block_preset != "None" and ":" in block_preset:
            preset_vector = block_preset.split(":", 1)[1]

        # CivitAI metadata collection
        lora_info_lines = []
        all_trigger_words = []
        civitai_urls = []
        individual_infos = ["", "", ""]  # Info for each slot
        collage_images = [None, None, None]  # Collage for each slot (indexed by slot)

        # Get API key from settings
        civitai_api_key = get_civitai_api_key() if HAS_CIVITAI else ""

        # Check if CivitAI lookup is enabled but no API key
        if civitai_lookup == "On" and HAS_CIVITAI and not civitai_api_key:
            raise ValueError("CivitAI lookup is enabled but no API key found. Please add your CivitAI API key in Settings -> DonutNodes.CivitAI.ApiKey")

        # Get cache and temp directory if CivitAI enabled
        cache = None
        temp_dir = folder_paths.get_temp_directory()
        if civitai_lookup == "On" and HAS_CIVITAI:
            cache = get_cache()

        def _maybe_add(slot_idx, sw, name, mw, cw, bv):
            if sw == "On" and name != "None":
                # Use preset if set, otherwise use individual block vector
                final_bv = preset_vector if preset_vector else bv.strip()
                stack.append((name, mw, cw, final_bv))

                # Fetch CivitAI metadata
                if cache is not None:
                    lora_path = folder_paths.get_full_path("loras", name)
                    if lora_path and os.path.exists(lora_path):
                        try:
                            file_hash = get_or_compute_hash(lora_path, use_cache=True)
                            info = cache.get_or_fetch_info(
                                file_hash,
                                api_key=civitai_api_key if civitai_api_key else None,
                                download_preview=True
                            )
                            if info:
                                display_name = info.get_display_name()
                                rec_weight = info.recommended_weight
                                weight_hint = f" [rec:{rec_weight}]" if rec_weight != 1.0 else ""
                                lora_info_lines.append(f"{display_name} (w:{mw}{weight_hint})")
                                if info.trained_words:
                                    all_trigger_words.extend(info.trained_words)
                                civitai_urls.append(info.model_url)
                                print(f"[DonutLoRAStack] CivitAI: {name} -> {display_name}")

                                # Build individual info for this slot
                                slot_info = f"{display_name}{weight_hint}"
                                slot_info += f"\n{info.base_model} | ↓{info.download_count}"
                                if info.trained_words:
                                    slot_info += f"\nTriggers: {', '.join(info.trained_words)}"
                                slot_info += f"\n{info.model_url}"
                                individual_infos[slot_idx] = slot_info

                                # Create collage for this LoRA
                                collage_path = cache.create_preview_collage(info, max_images=4)
                                if collage_path:
                                    # Copy to temp dir for ComfyUI to serve
                                    collage_filename = f"lora_preview_{slot_idx}_{file_hash[:8]}.jpg"
                                    temp_path = os.path.join(temp_dir, collage_filename)
                                    shutil.copy2(collage_path, temp_path)
                                    collage_images[slot_idx] = {
                                        "filename": collage_filename,
                                        "subfolder": "",
                                        "type": "temp"
                                    }
                                    print(f"[DonutLoRAStack] Created collage for slot {slot_idx+1}")
                            else:
                                # Not found on CivitAI
                                search_url = f"https://civitai.com/search/models?query={file_hash}"
                                lora_info_lines.append(f"{name} (w:{mw}) [not on CivitAI]")
                                civitai_urls.append(search_url)
                                individual_infos[slot_idx] = f"{name}\n[not on CivitAI]"
                        except Exception as e:
                            print(f"[DonutLoRAStack] CivitAI error for {name}: {e}")
                            lora_info_lines.append(f"{name} (w:{mw})")
                            individual_infos[slot_idx] = name
                    else:
                        lora_info_lines.append(f"{name} (w:{mw})")
                        individual_infos[slot_idx] = name
                else:
                    lora_info_lines.append(f"{name} (w:{mw})")
                    individual_infos[slot_idx] = name

        _maybe_add(0, switch_1, lora_name_1, model_weight_1, clip_weight_1, block_vector_1)
        _maybe_add(1, switch_2, lora_name_2, model_weight_2, clip_weight_2, block_vector_2)
        _maybe_add(2, switch_3, lora_name_3, model_weight_3, clip_weight_3, block_vector_3)

        # Format outputs
        lora_info = "\n".join(lora_info_lines) if lora_info_lines else "No LoRAs selected"
        trigger_words = ", ".join(sorted(set(all_trigger_words))) if all_trigger_words else ""
        urls_output = "\n".join(civitai_urls) if civitai_urls else ""

        print(f"[DonutLoRAStack] Built stack with {len(stack)} LoRAs")
        if all_trigger_words:
            print(f"[DonutLoRAStack] Trigger words: {trigger_words}")

        # Return with UI data for the JavaScript extension to display in-node
        ui_data = {
            "text": [lora_info, trigger_words, urls_output, individual_infos[0], individual_infos[1], individual_infos[2]]
        }
        # Add collage images if any are available (keep as list with None for empty slots)
        if any(img is not None for img in collage_images):
            ui_data["images"] = collage_images

        return {
            "ui": ui_data,
            "result": (stack,)
        }


# ------------------------------------------------------------------------
# DonutApplyLoRAStack: per-block UNet + uniform CLIP merges, always in that order
# ------------------------------------------------------------------------
class DonutApplyLoRAStack:
    class_type = "CUSTOM"
    aux_id     = "DonutsDelivery/ComfyUI-DonutNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "model":      ("MODEL",),
            "clip":       ("CLIP",),
            "lora_stack": ("LORA_STACK",),
        }}

    RETURN_TYPES = ("MODEL","CLIP","STRING")
    RETURN_NAMES = ("model","clip","show_help")
    FUNCTION     = "apply_stack"
    CATEGORY     = "Comfyanonymous/LoRA"

    def apply_stack(self, model, clip, lora_stack=None):
        help_url = (
            "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/"
            "wiki/LoRA-Nodes#cr-apply-lora-stack"
        )
        
        print(f"[DonutApplyLoRAStack] DEBUG: lora_stack type: {type(lora_stack)}")
        print(f"[DonutApplyLoRAStack] DEBUG: lora_stack value: {lora_stack}")
        
        if lora_stack is None:
            print("[DonutApplyLoRAStack] No LoRAs to apply - lora_stack is None")
            return (model, clip, help_url)
            
        if len(lora_stack) == 0:
            print("[DonutApplyLoRAStack] No LoRAs to apply - lora_stack is empty")
            return (model, clip, help_url)

        print(f"[DonutApplyLoRAStack] Applying {len(lora_stack)} LoRAs to model and clip")
        unet, text_enc = model, clip
        loader         = LoraLoaderBlockWeight()

        for i, (name, mw, cw, bv) in enumerate(lora_stack):
            try:
                print(f"[DonutApplyLoRAStack] Processing LoRA {i+1}/{len(lora_stack)}: {name} (model:{mw}, clip:{cw}, vector:{bv})")
                
                if mw == 0.0 and cw == 0.0:
                    print(f"[DonutApplyLoRAStack] WARNING: Both model and clip strengths are 0.0 for {name} - skipping")
                    continue
                
                path = folder_paths.get_full_path("loras", name)
                print(f"[DonutApplyLoRAStack] Loading LoRA from: {path}")
                lora = comfy.utils.load_torch_file(path, safe_load=True)

                # Auto-detect block count from LoRA keys if no vector provided
                if bv:
                    vector = bv
                else:
                    # Count unique block numbers to determine architecture
                    block_nums = set()
                    for k in lora.keys():
                        if "layers." in k:  # Z-Image/Lumina2
                            m = re.search(r'layers\.(\d+)', k)
                            if m:
                                block_nums.add(int(m.group(1)))
                        elif "input_blocks." in k or "output_blocks." in k or "middle_block." in k:
                            # UNet architecture (SD1.5/SDXL)
                            pass
                        elif "double_blocks." in k or "single_blocks." in k:
                            # Flux
                            pass

                    if block_nums:  # Z-Image detected
                        num_blocks = max(block_nums) + 1
                        vector = ",".join(["1"] * (num_blocks + 1))  # +1 for base
                        print(f"[DonutApplyLoRAStack] Auto-detected Z-Image with {num_blocks} layers")
                    else:
                        vector = ",".join(["1"] * 13)  # Default: base + 12 blocks (SDXL)

                print(f"[DonutApplyLoRAStack] Using block vector: {vector}")

                # 1) block-weighted UNet merge (clip_strength=0)
                if mw != 0.0:
                    print(f"[DonutApplyLoRAStack] Applying UNet merge with strength {mw}")
                    print(f"[DonutApplyLoRAStack] DEBUG: UNet model before LoRA: {type(unet)}")
                    print(f"[DonutApplyLoRAStack] DEBUG: LoRA keys in file: {len(lora.keys())}")
                    print(f"[DonutApplyLoRAStack] DEBUG: First few LoRA keys: {list(lora.keys())[:5]}")
                    
                    unet_before_id = id(unet)
                    print(f"[DonutApplyLoRAStack] DEBUG: About to call load_lora_for_models with strength={mw}, A=1.0, B=1.0, vector='{vector}'")
                    unet, text_enc, _ = loader.load_lora_for_models(
                        unet, text_enc, lora,
                        strength_model=    mw,
                        strength_clip=     0.0,
                        inverse=           False,
                        seed=              0,
                        A=                 1.0,
                        B=                 1.0,
                        block_vector=      vector
                    )
                    print(f"[DonutApplyLoRAStack] DEBUG: load_lora_for_models completed")
                    unet_after_id = id(unet)
                    print(f"[DonutApplyLoRAStack] DEBUG: UNet model after LoRA: {type(unet)}")
                    print(f"[DonutApplyLoRAStack] DEBUG: UNet object changed: {unet_before_id != unet_after_id}")
                    
                    # Check if the model actually has patches applied
                    if hasattr(unet, 'patches'):
                        print(f"[DonutApplyLoRAStack] DEBUG: Model patches count: {len(unet.patches)}")
                        if len(unet.patches) > 0:
                            print(f"[DonutApplyLoRAStack] DEBUG: Sample patch keys: {list(unet.patches.keys())[:3]}")
                    else:
                        print(f"[DonutApplyLoRAStack] DEBUG: Model has no patches attribute")

                # 2) uniform CLIP merge (no block control)
                if cw != 0.0:
                    print(f"[DonutApplyLoRAStack] Applying CLIP merge with strength {cw}")
                    clip_before_id = id(text_enc)
                    unet, text_enc = comfy.sd.load_lora_for_models(
                        unet, text_enc, lora,
                        0.0,               # no UNet change
                        cw                 # clip strength
                    )
                    clip_after_id = id(text_enc)
                    print(f"[DonutApplyLoRAStack] DEBUG: CLIP object changed: {clip_before_id != clip_after_id}")
                    
                    # Check CLIP patches
                    if hasattr(text_enc, 'patches'):
                        print(f"[DonutApplyLoRAStack] DEBUG: CLIP patches count: {len(text_enc.patches)}")
                else:
                    print(f"[DonutApplyLoRAStack] DEBUG: Skipping CLIP merge (strength=0.0)")
                    
                print(f"[DonutApplyLoRAStack] Successfully applied LoRA: {name}")
                
            except Exception as e:
                print(f"[DonutApplyLoRAStack] ERROR applying LoRA {name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        print(f"[DonutApplyLoRAStack] Completed applying all {len(lora_stack)} LoRAs")
        
        # CRITICAL DEBUG: Verify the models being returned have patches
        print(f"🔥 [DonutApplyLoRAStack] RETURNING MODEL TYPE: {type(unet)}")
        print(f"🔥 [DonutApplyLoRAStack] RETURNING MODEL ID: {id(unet)}")
        if hasattr(unet, 'patches'):
            print(f"🔥 [DonutApplyLoRAStack] RETURNING MODEL PATCHES COUNT: {len(unet.patches)}")
            if len(unet.patches) > 0:
                print(f"🔥 [DonutApplyLoRAStack] RETURNING MODEL SAMPLE PATCH KEYS: {list(unet.patches.keys())[:3]}")
                # Verify patches have actual content
                first_patch_key = list(unet.patches.keys())[0]
                patch_data = unet.patches[first_patch_key]
                print(f"🔥 [DonutApplyLoRAStack] FIRST PATCH DATA TYPE: {type(patch_data)}")
        else:
            print(f"🔥 [DonutApplyLoRAStack] WARNING: RETURNING MODEL HAS NO PATCHES ATTRIBUTE!")
            
        print(f"🔥 [DonutApplyLoRAStack] RETURNING CLIP TYPE: {type(text_enc)}")
        if hasattr(text_enc, 'patches'):
            print(f"🔥 [DonutApplyLoRAStack] RETURNING CLIP PATCHES COUNT: {len(text_enc.patches)}")
        else:
            print(f"🔥 [DonutApplyLoRAStack] RETURNING CLIP HAS NO PATCHES ATTRIBUTE")
        
        return (unet, text_enc, help_url)


NODE_CLASS_MAPPINGS = {
    "DonutLoRAStack":      DonutLoRAStack,
    "DonutApplyLoRAStack": DonutApplyLoRAStack,
}
NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
