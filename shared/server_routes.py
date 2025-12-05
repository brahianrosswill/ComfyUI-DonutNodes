"""
DonutNodes server-side API routes.

Provides endpoints for configuration management that can be called
from the JavaScript frontend.
"""

import json
from aiohttp import web

try:
    from server import PromptServer
    HAS_SERVER = True
except ImportError:
    HAS_SERVER = False

from .config import load_config, save_config, get_config


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

    print("[DonutNodes] Server routes registered")


# Register routes when module is imported
register_routes()
