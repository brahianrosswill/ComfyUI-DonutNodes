"""
DonutTiledUpscale - A simple tiled upscale node designed to work with any model type.
Uses the same sampling approach as the core KSampler to ensure compatibility with
Z-Image/Lumina2 and other modern architectures.
"""

import torch
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from comfy import model_management
from nodes import VAEEncode, VAEDecode, VAEDecodeTiled
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def upscale_with_model(upscale_model, image):
    """Upscale image using a model (adapted from ComfyUI's ImageUpscaleWithModel)"""
    device = model_management.get_torch_device()

    memory_required = model_management.module_size(upscale_model.model)
    memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
    memory_required += image.nelement() * image.element_size()
    model_management.free_memory(memory_required, device)

    upscale_model.to(device)
    in_img = image.movedim(-1, -3).to(device)

    tile = 512
    overlap = 32

    oom = True
    while oom:
        try:
            steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
            )
            pbar = comfy.utils.ProgressBar(steps)
            s = comfy.utils.tiled_scale(
                in_img,
                lambda a: upscale_model(a),
                tile_x=tile, tile_y=tile,
                overlap=overlap,
                upscale_amount=upscale_model.scale,
                pbar=pbar
            )
            oom = False
        except model_management.OOM_EXCEPTION as e:
            tile //= 2
            if tile < 128:
                raise e

    upscale_model.to("cpu")
    s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
    return s


def calculate_tiling(original_width, original_height, rescale_factor, feather_percent):
    """
    Calculate tiling based on the original image dimensions and scale factor.

    Each tile will be the size of the original image. The number of tiles
    is determined by the rescale factor (e.g., 2x = 2x2 tiles, 3x = 3x3 tiles).

    feather_percent controls the overlap/blend zone as a percentage of tile size.

    Returns (tile_width, tile_height, overlap_x, overlap_y, num_tiles_x, num_tiles_y, output_width, output_height)
    """
    # Tile size matches original image dimensions
    tile_width = original_width
    tile_height = original_height

    # Number of tiles based on scale factor (round to nearest integer)
    num_tiles_x = max(1, round(rescale_factor))
    num_tiles_y = max(1, round(rescale_factor))

    # Overlap proportional to each dimension to preserve aspect ratio
    overlap_x = int(tile_width * feather_percent / 100)
    overlap_y = int(tile_height * feather_percent / 100)

    # Make sure overlap is divisible by 8 for VAE compatibility
    overlap_x = (overlap_x // 8) * 8
    overlap_y = (overlap_y // 8) * 8

    # Calculate step (tile size minus overlap)
    step_x = tile_width - overlap_x
    step_y = tile_height - overlap_y

    # Output dimensions: first tile full size, then add steps for remaining tiles
    output_width = tile_width + (num_tiles_x - 1) * step_x
    output_height = tile_height + (num_tiles_y - 1) * step_y

    return (tile_width, tile_height, overlap_x, overlap_y, num_tiles_x, num_tiles_y, output_width, output_height)


def tensor_to_pil(tensor, batch_index=0):
    """Convert a tensor [B,H,W,C] to PIL Image"""
    img = tensor[batch_index].cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def pil_to_tensor(image):
    """Convert PIL Image to tensor [1,H,W,C]"""
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def create_debug_image(width, height, tile_width, tile_height, overlap_x, overlap_y, num_tiles_x, num_tiles_y,
                       original_width, original_height, rescale_factor):
    """
    Create a debug visualization showing tile layout, overlap zones, and dimensions.
    Uses separate overlap_x and overlap_y to preserve aspect ratio.
    """
    # Create image with dark background
    debug_img = Image.new('RGB', (width, height), (40, 40, 40))
    draw = ImageDraw.Draw(debug_img)

    step_x = tile_width - overlap_x
    step_y = tile_height - overlap_y

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 11)
        font_large = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
            font_small = font
            font_large = font

    # Color scheme
    tile_border_color = (100, 200, 255)  # Light blue for tile borders
    overlap_color = (255, 150, 50, 128)  # Orange for overlap zones
    feather_color = (255, 100, 100)  # Red for feather gradients
    ruler_color = (200, 200, 200)  # Light gray for rulers
    text_color = (255, 255, 255)  # White for text
    dim_text_color = (180, 180, 180)  # Dimmer text

    # Draw tiles
    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            x1 = tx * step_x
            y1 = ty * step_y
            x2 = x1 + tile_width
            y2 = y1 + tile_height

            # Fill tile with semi-transparent color (alternating for visibility)
            tile_color = (60, 80, 100) if (tx + ty) % 2 == 0 else (80, 60, 100)
            draw.rectangle([x1, y1, x2, y2], fill=tile_color)

            # Draw tile border
            draw.rectangle([x1, y1, x2, y2], outline=tile_border_color, width=2)

            # Draw tile number in center
            tile_num = ty * num_tiles_x + tx + 1
            tile_label = f"Tile {tile_num}"
            bbox = draw.textbbox((0, 0), tile_label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_x = x1 + (tile_width - text_w) // 2
            text_y = y1 + (tile_height - text_h) // 2
            draw.text((text_x, text_y), tile_label, fill=text_color, font=font)

            # Draw tile dimensions below tile number
            dim_label = f"{tile_width}x{tile_height}"
            bbox = draw.textbbox((0, 0), dim_label, font=font_small)
            dim_w = bbox[2] - bbox[0]
            draw.text((x1 + (tile_width - dim_w) // 2, text_y + text_h + 5),
                      dim_label, fill=dim_text_color, font=font_small)

    # Draw overlap zones (horizontal) using overlap_x
    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x - 1):
            x1 = (tx + 1) * step_x
            y1 = ty * step_y
            # Draw overlap region
            for i in range(overlap_x):
                alpha = int(255 * (1 - i / overlap_x)) if overlap_x > 0 else 0
                # Left feather (from previous tile)
                draw.line([(x1 + i, y1), (x1 + i, y1 + tile_height)],
                          fill=(255, 100, 100, alpha), width=1)

    # Draw overlap zones (vertical) using overlap_y
    for ty in range(num_tiles_y - 1):
        for tx in range(num_tiles_x):
            x1 = tx * step_x
            y1 = (ty + 1) * step_y
            # Draw overlap region
            for i in range(overlap_y):
                alpha = int(255 * (1 - i / overlap_y)) if overlap_y > 0 else 0
                draw.line([(x1, y1 + i), (x1 + tile_width, y1 + i)],
                          fill=(255, 100, 100, alpha), width=1)

    # Draw rulers on top and left
    ruler_height = 30
    ruler_width = 30

    # Top ruler background
    draw.rectangle([0, 0, width, ruler_height], fill=(30, 30, 30))
    # Left ruler background
    draw.rectangle([0, 0, ruler_width, height], fill=(30, 30, 30))

    # Draw tick marks and labels for width
    # Major ticks at tile boundaries
    for tx in range(num_tiles_x + 1):
        x = tx * step_x if tx < num_tiles_x else width
        if tx == num_tiles_x:
            x = width
        draw.line([(x, 0), (x, ruler_height)], fill=ruler_color, width=2)
        if tx < num_tiles_x:
            # Label the step distance
            label = f"{step_x}px" if tx > 0 else f"{tile_width}px"
            mid_x = tx * step_x + step_x // 2 if tx > 0 else tile_width // 2
            if tx == 0:
                mid_x = tile_width // 2
            bbox = draw.textbbox((0, 0), label, font=font_small)
            text_w = bbox[2] - bbox[0]
            if mid_x - text_w // 2 > ruler_width:
                draw.text((mid_x - text_w // 2, 8), label, fill=dim_text_color, font=font_small)

    # Draw tick marks and labels for height
    for ty in range(num_tiles_y + 1):
        y = ty * step_y if ty < num_tiles_y else height
        if ty == num_tiles_y:
            y = height
        draw.line([(0, y), (ruler_width, y)], fill=ruler_color, width=2)

    # Total dimensions label (top-right corner)
    total_label = f"Output: {width}x{height}"
    bbox = draw.textbbox((0, 0), total_label, font=font_large)
    text_w = bbox[2] - bbox[0]
    draw.rectangle([width - text_w - 20, 0, width, 25], fill=(60, 60, 60))
    draw.text((width - text_w - 10, 5), total_label, fill=text_color, font=font_large)

    # Info box in bottom-left
    info_lines = [
        f"Original: {original_width}x{original_height}",
        f"Scale: {rescale_factor:.2f}x",
        f"Tiles: {num_tiles_x}x{num_tiles_y} = {num_tiles_x * num_tiles_y}",
        f"Tile size: {tile_width}x{tile_height}",
        f"Overlap: {overlap_x}x{overlap_y}px",
        f"Step: {step_x}x{step_y}px",
    ]

    box_height = len(info_lines) * 18 + 10
    box_width = 180
    box_x = 5
    box_y = height - box_height - 5

    draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height],
                   fill=(20, 20, 20, 200), outline=(100, 100, 100))

    for i, line in enumerate(info_lines):
        draw.text((box_x + 8, box_y + 5 + i * 18), line, fill=text_color, font=font_small)

    # Legend in bottom-right
    legend_x = width - 150
    legend_y = height - 70
    draw.rectangle([legend_x, legend_y, width - 5, height - 5],
                   fill=(20, 20, 20, 200), outline=(100, 100, 100))
    draw.text((legend_x + 5, legend_y + 5), "Legend:", fill=text_color, font=font_small)
    draw.rectangle([legend_x + 5, legend_y + 22, legend_x + 20, legend_y + 32],
                   outline=tile_border_color, width=2)
    draw.text((legend_x + 25, legend_y + 20), "Tile border", fill=dim_text_color, font=font_small)
    draw.rectangle([legend_x + 5, legend_y + 40, legend_x + 20, legend_y + 50],
                   fill=(255, 100, 100))
    draw.text((legend_x + 25, legend_y + 38), "Feather zone", fill=dim_text_color, font=font_small)

    return debug_img


class DonutTiledUpscale:
    """
    Tiled upscale node that processes an image in tiles using img2img.
    Designed to work with any model type including Z-Image/Lumina2.
    """

    @classmethod
    def INPUT_TYPES(cls):
        resampling_methods = ["lanczos", "nearest", "bilinear", "bicubic"]
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rescale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.5, "tooltip": "Upscale factor. 2x = 2x2 tiles, 3x = 3x3 tiles. Each tile matches original image size."}),
                "resampling_method": (resampling_methods, {"default": "lanczos"}),
                "feather": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 50.0, "step": 1.0, "tooltip": "Feather/blend zone as percentage of tile size. Higher = smoother transitions."}),
                "tiled_vae": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("image", "debug_image",)
    FUNCTION = "upscale"
    CATEGORY = "donut/upscale"

    def upscale(self, image, upscale_model, model, positive, negative, vae, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                rescale_factor, resampling_method, feather, tiled_vae):

        # Resampling filter mapping
        resample_filters = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS
        }
        resample_filter = resample_filters.get(resampling_method, Image.LANCZOS)

        # Get image dimensions (B, H, W, C)
        batch_size, img_height, img_width, channels = image.shape

        # Calculate tiling based on original image size and scale factor
        # Each tile will be the size of the original image
        tile_width, tile_height, overlap_x, overlap_y, num_tiles_x, num_tiles_y, output_width, output_height = calculate_tiling(
            img_width, img_height, rescale_factor, feather
        )

        print(f"[DonutTiledUpscale] Input: {img_width}x{img_height}")
        print(f"[DonutTiledUpscale] Scale: {rescale_factor}x -> {num_tiles_x}x{num_tiles_y} tiles")
        print(f"[DonutTiledUpscale] Tile size: {tile_width}x{tile_height} (matches original)")
        print(f"[DonutTiledUpscale] Overlap/Feather: {overlap_x}x{overlap_y}px ({feather}%)")
        print(f"[DonutTiledUpscale] Output: {output_width}x{output_height}")

        # Get upscale factor from the model
        model_scale = upscale_model.scale
        print(f"[DonutTiledUpscale] Upscale model scale: {model_scale}x")

        # First, upscale with the model
        print(f"[DonutTiledUpscale] Upscaling with model...")
        upscaled_image = upscale_with_model(upscale_model, image)
        _, up_height, up_width, _ = upscaled_image.shape
        print(f"[DonutTiledUpscale] After model upscale: {up_width}x{up_height}")

        # Calculate step between tiles
        step_x = tile_width - overlap_x
        step_y = tile_height - overlap_y

        total_tiles = num_tiles_x * num_tiles_y
        print(f"[DonutTiledUpscale] Processing {total_tiles} total tiles")

        # VAE encoder/decoder
        vae_encoder = VAEEncode()
        vae_decoder = VAEDecode() if not tiled_vae else VAEDecodeTiled()

        # Process each batch item
        result_images = []

        for b in range(batch_size):
            # Convert model-upscaled image to PIL and resize to output dimensions
            pil_image = tensor_to_pil(upscaled_image, b)
            pil_image = pil_image.resize((output_width, output_height), resample_filter)

            # Create result image at output dimensions
            result_image = Image.new('RGB', (output_width, output_height))

            # Create weight map for blending
            weight_map = Image.new('L', (output_width, output_height), 0)

            tile_idx = 0
            pbar = comfy.utils.ProgressBar(total_tiles)

            for ty in range(num_tiles_y):
                for tx in range(num_tiles_x):
                    tile_idx += 1

                    # Calculate tile coordinates - tiles fit exactly
                    x1 = tx * step_x
                    y1 = ty * step_y
                    x2 = x1 + tile_width
                    y2 = y1 + tile_height

                    # Crop tile (always exact tile size)
                    tile = pil_image.crop((x1, y1, x2, y2))

                    # Convert to tensor for VAE encoding
                    tile_tensor = pil_to_tensor(tile)

                    # Encode to latent
                    latent = vae_encoder.encode(vae, tile_tensor)[0]

                    # Sample using the same approach as core KSampler
                    latent_image = latent["samples"]
                    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

                    # Prepare noise
                    noise = comfy.sample.prepare_noise(latent_image, seed + tile_idx)

                    # Sample
                    callback = latent_preview.prepare_callback(model, steps)
                    samples = comfy.sample.sample(
                        model, noise, steps, cfg, sampler_name, scheduler,
                        positive, negative, latent_image,
                        denoise=denoise,
                        disable_noise=False,
                        start_step=None,
                        last_step=None,
                        force_full_denoise=False,
                        noise_mask=None,
                        callback=callback,
                        disable_pbar=True,
                        seed=seed + tile_idx
                    )

                    # Decode
                    sampled_latent = {"samples": samples}
                    if tiled_vae:
                        decoded = vae_decoder.decode(vae, sampled_latent, 512)[0]
                    else:
                        decoded = vae_decoder.decode(vae, sampled_latent)[0]

                    # Convert back to PIL
                    sampled_tile = tensor_to_pil(decoded, 0)

                    # Blend tile into result with feathered edges
                    tile_mask = self._create_blend_mask(tile_width, tile_height, overlap_x, overlap_y,
                                                        tx == 0, ty == 0,
                                                        tx == num_tiles_x - 1, ty == num_tiles_y - 1)
                    result_image = self._blend_tile(result_image, sampled_tile, x1, y1, tile_mask, weight_map)

                    pbar.update(1)

            # Convert result back to tensor
            result_tensor = pil_to_tensor(result_image)
            result_images.append(result_tensor)

        # Combine batch
        output = torch.cat(result_images, dim=0)

        # Generate debug image
        debug_pil = create_debug_image(
            output_width, output_height,
            tile_width, tile_height, overlap_x, overlap_y,
            num_tiles_x, num_tiles_y,
            img_width, img_height,
            rescale_factor
        )
        debug_tensor = pil_to_tensor(debug_pil)

        return (output, debug_tensor)

    def _create_blend_mask(self, width, height, overlap_x, overlap_y, is_left, is_top, is_right, is_bottom):
        """Create a feathered blend mask for seamless tiling with separate x/y overlaps"""
        mask = Image.new('L', (width, height), 255)

        if overlap_x <= 0 and overlap_y <= 0:
            return mask

        mask_array = np.array(mask, dtype=np.float32)

        # Feather left edge
        if not is_left and overlap_x > 0:
            for x in range(min(overlap_x, width)):
                mask_array[:, x] *= x / overlap_x

        # Feather right edge
        if not is_right and overlap_x > 0:
            for x in range(min(overlap_x, width)):
                mask_array[:, -(x+1)] *= x / overlap_x

        # Feather top edge
        if not is_top and overlap_y > 0:
            for y in range(min(overlap_y, height)):
                mask_array[y, :] *= y / overlap_y

        # Feather bottom edge
        if not is_bottom and overlap_y > 0:
            for y in range(min(overlap_y, height)):
                mask_array[-(y+1), :] *= y / overlap_y

        return Image.fromarray(mask_array.astype(np.uint8))

    def _blend_tile(self, result, tile, x, y, tile_mask, weight_map):
        """Blend a tile into the result image using weighted averaging"""
        # Get the region from result
        region = result.crop((x, y, x + tile.width, y + tile.height))
        existing_weight = weight_map.crop((x, y, x + tile.width, y + tile.height))

        # Convert to numpy for blending
        region_arr = np.array(region, dtype=np.float32)
        tile_arr = np.array(tile, dtype=np.float32)
        mask_arr = np.array(tile_mask, dtype=np.float32) / 255.0
        existing_arr = np.array(existing_weight, dtype=np.float32)

        # Weighted blend
        new_weight = existing_arr + mask_arr * 255.0

        # Avoid division by zero
        blend_factor = np.where(new_weight > 0,
                                (mask_arr * 255.0) / np.maximum(new_weight, 1.0),
                                1.0)

        # Blend
        blended = region_arr * (1 - blend_factor[:, :, np.newaxis]) + tile_arr * blend_factor[:, :, np.newaxis]
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # Update result and weight map
        result.paste(Image.fromarray(blended), (x, y))
        weight_map.paste(Image.fromarray(np.clip(new_weight, 0, 255).astype(np.uint8)), (x, y))

        return result


NODE_CLASS_MAPPINGS = {
    "DonutTiledUpscale": DonutTiledUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DonutTiledUpscale": "Donut Tiled Upscale",
}
