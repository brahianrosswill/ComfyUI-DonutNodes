"""
DonutNodes server-side API routes.

Provides endpoints for configuration management and LoRA browser
that can be called from the JavaScript frontend.
"""

import json
import os
import hashlib
from aiohttp import web
from pathlib import Path

try:
    from server import PromptServer
    HAS_SERVER = True
except ImportError:
    HAS_SERVER = False

try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except ImportError:
    HAS_FOLDER_PATHS = False

from .config import load_config, save_config, get_config

# Import CivitAI cache for LoRA browser
try:
    from .civitai_api import CivitAICache, get_civitai_cache_dir
    HAS_CIVITAI = True
except ImportError:
    HAS_CIVITAI = False

# Import hash computation
try:
    from .lora_hash import get_or_compute_hash
    HAS_LORA_HASH = True
except ImportError:
    HAS_LORA_HASH = False


def register_routes():
    """Register API routes with ComfyUI's server."""
    if not HAS_SERVER:
        print("[DonutNodes] Server not available, skipping route registration")
        return

    routes = PromptServer.instance.routes

    @routes.get('/donut/config')
    async def get_donut_config(request):
        """Return current DonutNodes configuration."""
        config = load_config(force_reload=True)
        return web.json_response(config)

    @routes.post('/donut/config/civitai_api_key')
    async def set_civitai_api_key(request):
        """Save CivitAI API key to config."""
        try:
            data = await request.json()
            api_key = data.get("api_key", "")

            config = load_config(force_reload=True)
            if "civitai" not in config:
                config["civitai"] = {}
            config["civitai"]["api_key"] = api_key

            if save_config(config):
                return web.json_response({"status": "ok"})
            else:
                return web.json_response({"status": "error", "message": "Failed to save config"}, status=500)
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    @routes.post('/donut/config/civitai_auto_lookup')
    async def set_civitai_auto_lookup(request):
        """Save CivitAI auto-lookup setting."""
        try:
            data = await request.json()
            auto_lookup = data.get("auto_lookup", True)

            config = load_config(force_reload=True)
            if "civitai" not in config:
                config["civitai"] = {}
            config["civitai"]["auto_lookup"] = bool(auto_lookup)

            if save_config(config):
                return web.json_response({"status": "ok"})
            else:
                return web.json_response({"status": "error", "message": "Failed to save config"}, status=500)
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    @routes.post('/donut/config/civitai_download_previews')
    async def set_civitai_download_previews(request):
        """Save CivitAI download previews setting."""
        try:
            data = await request.json()
            download_previews = data.get("download_previews", True)

            config = load_config(force_reload=True)
            if "civitai" not in config:
                config["civitai"] = {}
            config["civitai"]["download_previews"] = bool(download_previews)

            if save_config(config):
                return web.json_response({"status": "ok"})
            else:
                return web.json_response({"status": "error", "message": "Failed to save config"}, status=500)
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    @routes.post('/donut/config/civitai_prefer_sfw')
    async def set_civitai_prefer_sfw(request):
        """Save CivitAI prefer SFW setting."""
        try:
            data = await request.json()
            prefer_sfw = data.get("prefer_sfw", True)

            config = load_config(force_reload=True)
            if "civitai" not in config:
                config["civitai"] = {}
            config["civitai"]["prefer_sfw"] = bool(prefer_sfw)

            if save_config(config):
                return web.json_response({"status": "ok"})
            else:
                return web.json_response({"status": "error", "message": "Failed to save config"}, status=500)
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    # ============================================
    # LoRA Browser Routes
    # ============================================

    def get_lora_hash(filepath: str) -> str:
        """Get hash of a LoRA file for CivitAI lookup.

        Uses full SHA256 hash (first 10 chars) which is what CivitAI accepts.
        The hash is cached for subsequent lookups.
        """
        if HAS_LORA_HASH:
            try:
                # Full SHA256, cached for speed on subsequent lookups
                full_hash = get_or_compute_hash(filepath, hash_type="SHA256", use_cache=True)
                return full_hash[:10]  # CivitAI accepts first 10 chars
            except Exception as e:
                print(f"[DonutNodes] Error computing hash: {e}")
                return ""
        else:
            # Fallback: compute full SHA256 manually
            sha256 = hashlib.sha256()
            try:
                with open(filepath, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b''):
                        sha256.update(chunk)
                return sha256.hexdigest()[:10].upper()
            except:
                return ""

    @routes.get('/donut/loras/list')
    async def list_loras(request):
        """List all available LoRAs with basic info.

        Query params:
            include_meta: If "true", include cached CivitAI metadata (uses cached hashes only)
        """
        if not HAS_FOLDER_PATHS:
            return web.json_response({"error": "folder_paths not available"}, status=500)

        include_meta = request.query.get("include_meta", "false").lower() == "true"

        # Get cache if needed
        cache = None
        if include_meta and HAS_CIVITAI:
            cache = CivitAICache()

        try:
            lora_paths = folder_paths.get_folder_paths("loras")
            loras = []

            for lora_dir in lora_paths:
                if not os.path.exists(lora_dir):
                    continue
                for root, dirs, files in os.walk(lora_dir):
                    for file in files:
                        if file.endswith(('.safetensors', '.pt', '.ckpt')):
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, lora_dir)

                            lora_data = {
                                "name": rel_path,
                                "filename": file,
                                "full_path": full_path
                            }

                            # Add cached metadata if requested (only use cached hash, don't compute)
                            if include_meta and cache and HAS_LORA_HASH:
                                from .lora_hash import get_cached_hash
                                cached_hashes = get_cached_hash(full_path)
                                if cached_hashes and "SHA256" in cached_hashes:
                                    file_hash = cached_hashes["SHA256"][:10]
                                    lora_data["hash"] = file_hash
                                    info = cache.get_cached_info(file_hash)
                                    if info:
                                        lora_data["civitai_name"] = info.model_name
                                        lora_data["civitai_version"] = info.version_name
                                        lora_data["base_model"] = info.base_model
                                        lora_data["has_preview"] = True
                                    else:
                                        lora_data["has_preview"] = False
                                else:
                                    # No cached hash yet
                                    lora_data["has_preview"] = False

                            loras.append(lora_data)

            # Sort alphabetically
            loras.sort(key=lambda x: x["name"].lower())

            return web.json_response({"loras": loras, "count": len(loras)})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @routes.get('/donut/loras/info')
    async def get_lora_info(request):
        """Get CivitAI info for a specific LoRA, fetching if needed."""
        if not HAS_FOLDER_PATHS or not HAS_CIVITAI:
            return web.json_response({"error": "Required modules not available"}, status=500)

        lora_name = request.query.get("name", "")
        if not lora_name:
            return web.json_response({"error": "No lora name provided"}, status=400)

        try:
            # Find the LoRA file
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path or not os.path.exists(lora_path):
                return web.json_response({"error": "LoRA not found"}, status=404)

            # Get hash
            file_hash = get_lora_hash(lora_path)
            if not file_hash:
                return web.json_response({
                    "name": lora_name,
                    "hash": None,
                    "civitai": None,
                    "error": "Could not compute hash"
                })

            # Get cache
            cache = CivitAICache()  # Uses default cache dir if none configured

            # Try to get cached info first
            info = cache.get_cached_info(file_hash)

            # If not cached, fetch from CivitAI
            if info is None:
                info = cache.get_or_fetch_info(file_hash, download_preview=True)

            if info:
                # Get preview images using cache helper methods
                hash_prefix = file_hash[:10]  # AutoV2 format
                preview_paths = []
                for i in range(4):
                    img_path = cache._get_image_path(file_hash, 'jpg', i)
                    if img_path.exists():
                        preview_paths.append(str(img_path))
                    else:
                        # Try other extensions
                        for ext in ['png', 'webp']:
                            img_path = cache._get_image_path(file_hash, ext, i)
                            if img_path.exists():
                                preview_paths.append(str(img_path))
                                break

                # Check for collage
                collage_path = cache.get_preview_collage_path(file_hash)
                if not collage_path:
                    # Create collage if doesn't exist
                    cache.create_preview_collage(info, max_images=4)
                    collage_path = cache.get_preview_collage_path(file_hash)

                return web.json_response({
                    "name": lora_name,
                    "hash": hash_prefix,
                    "civitai": info.to_dict(),
                    "preview_count": len(preview_paths),
                    "has_collage": collage_path is not None
                })
            else:
                return web.json_response({
                    "name": lora_name,
                    "hash": file_hash[:10],
                    "civitai": None,
                    "error": "Not found on CivitAI"
                })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    @routes.get('/donut/loras/preview')
    async def get_lora_preview(request):
        """Get preview image for a LoRA."""
        if not HAS_CIVITAI:
            return web.json_response({"error": "CivitAI module not available"}, status=500)

        hash_prefix = request.query.get("hash", "")
        image_type = request.query.get("type", "collage")  # collage, 0, 1, 2, 3

        if not hash_prefix:
            return web.json_response({"error": "No hash provided"}, status=400)

        try:
            cache = CivitAICache()  # Uses default cache dir if none configured
            images_dir = cache.images_dir

            # Find the image
            if image_type == "collage":
                img_path = images_dir / f"{hash_prefix}_collage.jpg"
            elif image_type == "0":
                # Try different extensions
                for ext in ['jpg', 'png', 'webp']:
                    img_path = images_dir / f"{hash_prefix}.{ext}"
                    if img_path.exists():
                        break
            else:
                for ext in ['jpg', 'png', 'webp']:
                    img_path = images_dir / f"{hash_prefix}_{image_type}.{ext}"
                    if img_path.exists():
                        break

            if img_path.exists():
                # Return the image
                with open(img_path, 'rb') as f:
                    img_data = f.read()

                content_type = 'image/jpeg'
                if str(img_path).endswith('.png'):
                    content_type = 'image/png'
                elif str(img_path).endswith('.webp'):
                    content_type = 'image/webp'

                return web.Response(body=img_data, content_type=content_type)
            else:
                return web.json_response({"error": "Image not found"}, status=404)

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    print("[DonutNodes] Server routes registered")


# Register routes when module is imported
register_routes()
