"""
CivitAI API client for fetching LoRA metadata by hash.

Endpoints used:
- GET /api/v1/model-versions/by-hash/:hash - Lookup model by file hash
"""

import json
import os
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
import time

# Import config
try:
    from .config import (
        get_civitai_api_key,
        get_civitai_cache_dir,
        get_civitai_download_previews,
        get_civitai_prefer_sfw,
    )
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    def get_civitai_api_key(): return ""
    def get_civitai_cache_dir(): return None
    def get_civitai_download_previews(): return True
    def get_civitai_prefer_sfw(): return True


CIVITAI_API_BASE = "https://civitai.com/api/v1"
CIVITAI_HASH_ENDPOINT = f"{CIVITAI_API_BASE}/model-versions/by-hash"

# Rate limiting - CivitAI recommends not hammering the API
MIN_REQUEST_INTERVAL = 1.0  # seconds between requests
_last_request_time = 0.0


@dataclass
class CivitAIImage:
    """Represents an image from CivitAI."""
    url: str
    width: int = 0
    height: int = 0
    nsfw: bool = False
    nsfw_level: int = 0
    hash: str = ""

    @classmethod
    def from_api(cls, data: dict) -> "CivitAIImage":
        return cls(
            url=data.get("url", ""),
            width=data.get("width", 0),
            height=data.get("height", 0),
            nsfw=data.get("nsfw", False),
            nsfw_level=data.get("nsfwLevel", 0),
            hash=data.get("hash", "")
        )


@dataclass
class CivitAIModelInfo:
    """Structured information about a model from CivitAI."""
    # Core identifiers
    model_id: int = 0
    model_version_id: int = 0
    model_name: str = ""
    version_name: str = ""

    # Descriptions
    description: str = ""
    trained_words: List[str] = field(default_factory=list)

    # Metadata
    base_model: str = ""
    model_type: str = ""  # LORA, Checkpoint, etc.

    # Creator info
    creator_name: str = ""
    creator_username: str = ""

    # Stats
    download_count: int = 0
    rating: float = 0.0
    rating_count: int = 0

    # URLs
    model_url: str = ""
    download_url: str = ""

    # Images
    images: List[CivitAIImage] = field(default_factory=list)

    # File info
    file_name: str = ""
    file_size_kb: float = 0.0
    file_hashes: Dict[str, str] = field(default_factory=dict)

    # Recommended settings (extracted from example images)
    recommended_weight: float = 1.0
    recommended_cfg: float = 7.0
    recommended_steps: int = 20
    example_sampler: str = ""

    # Cache metadata
    fetched_at: str = ""
    local_hash: str = ""

    @classmethod
    def from_api_response(cls, data: dict, local_hash: str = "") -> "CivitAIModelInfo":
        """Parse API response into structured model info."""
        # Get the model info (parent)
        model_data = data.get("model", {})

        # Get file info (first file usually)
        files = data.get("files", [])
        file_info = files[0] if files else {}
        file_hashes = file_info.get("hashes", {})

        # Get images
        images_data = data.get("images", [])
        images = [CivitAIImage.from_api(img) for img in images_data[:5]]  # Limit to 5 images

        # Get trained words
        trained_words = data.get("trainedWords", [])
        if isinstance(trained_words, str):
            trained_words = [w.strip() for w in trained_words.split(",") if w.strip()]

        # Get stats
        stats = data.get("stats", {})

        # Extract recommended settings from example images metadata
        recommended_weight = 1.0
        recommended_cfg = 7.0
        recommended_steps = 20
        example_sampler = ""

        for img_data in images_data:
            meta = img_data.get("meta", {})
            if meta:
                # Try to extract LoRA weight from generation params
                # Common formats: "lora:name:0.8" or "Lora hance weight: 0.8"
                resources = meta.get("resources", [])
                for res in resources:
                    if res.get("type") == "lora":
                        weight = res.get("weight", 1.0)
                        if weight and weight != 1.0:
                            recommended_weight = float(weight)
                            break

                # Get other generation params
                if meta.get("cfgScale"):
                    recommended_cfg = float(meta["cfgScale"])
                if meta.get("steps"):
                    recommended_steps = int(meta["steps"])
                if meta.get("sampler"):
                    example_sampler = meta["sampler"]

                # Only need first image with metadata
                if recommended_weight != 1.0 or example_sampler:
                    break

        return cls(
            model_id=model_data.get("id", data.get("modelId", 0)),
            model_version_id=data.get("id", 0),
            model_name=model_data.get("name", data.get("name", "Unknown")),
            version_name=data.get("name", ""),
            description=data.get("description", model_data.get("description", "")),
            trained_words=trained_words,
            base_model=data.get("baseModel", ""),
            model_type=model_data.get("type", "LORA"),
            creator_name=model_data.get("creator", {}).get("username", ""),
            creator_username=model_data.get("creator", {}).get("username", ""),
            download_count=stats.get("downloadCount", 0),
            rating=stats.get("rating", 0.0),
            rating_count=stats.get("ratingCount", 0),
            model_url=f"https://civitai.com/models/{model_data.get('id', data.get('modelId', 0))}",
            download_url=data.get("downloadUrl", file_info.get("downloadUrl", "")),
            images=images,
            file_name=file_info.get("name", ""),
            file_size_kb=file_info.get("sizeKB", 0),
            file_hashes=file_hashes,
            recommended_weight=recommended_weight,
            recommended_cfg=recommended_cfg,
            recommended_steps=recommended_steps,
            example_sampler=example_sampler,
            fetched_at=datetime.now().isoformat(),
            local_hash=local_hash
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["images"] = [asdict(img) for img in self.images]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CivitAIModelInfo":
        """Create from dictionary (for loading from cache)."""
        images = [CivitAIImage(**img) for img in data.pop("images", [])]
        return cls(**data, images=images)

    def get_preview_url(self, prefer_sfw: bool = True) -> Optional[str]:
        """Get the best preview image URL."""
        if not self.images:
            return None

        if prefer_sfw:
            # Try to find a SFW image first
            for img in self.images:
                if not img.nsfw and img.nsfw_level <= 1:
                    return img.url
        # Fall back to first image
        return self.images[0].url if self.images else None

    def get_display_name(self) -> str:
        """Get a formatted display name."""
        if self.version_name and self.version_name != self.model_name:
            return f"{self.model_name} ({self.version_name})"
        return self.model_name

    def get_trigger_words_str(self) -> str:
        """Get trigger/trained words as comma-separated string."""
        return ", ".join(self.trained_words) if self.trained_words else ""


def _rate_limit():
    """Enforce rate limiting between API requests."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


def fetch_model_by_hash(file_hash: str, api_key: Optional[str] = None,
                        timeout: int = 30) -> Optional[CivitAIModelInfo]:
    """
    Fetch model information from CivitAI by file hash.

    The public API works without authentication. API key is optional but provides:
    - Higher rate limits
    - Access to restricted/private models

    Args:
        file_hash: SHA256 or other supported hash
        api_key: Optional CivitAI API key for higher rate limits
        timeout: Request timeout in seconds

    Returns:
        CivitAIModelInfo if found, None otherwise
    """
    _rate_limit()

    # API key is optional - public endpoint works without it
    if api_key is None:
        api_key = get_civitai_api_key()

    url = f"{CIVITAI_HASH_ENDPOINT}/{file_hash}"

    headers = {
        "User-Agent": "ComfyUI-DonutNodes/1.0",
        "Accept": "application/json"
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                return CivitAIModelInfo.from_api_response(data, local_hash=file_hash)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"[CivitAI] Model not found for hash: {file_hash[:16]}...")
        elif e.code == 429:
            print(f"[CivitAI] Rate limited - try again later")
        else:
            print(f"[CivitAI] HTTP error {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        print(f"[CivitAI] Connection error: {e.reason}")
    except json.JSONDecodeError:
        print(f"[CivitAI] Invalid JSON response")
    except Exception as e:
        print(f"[CivitAI] Error fetching model info: {e}")

    return None


def download_preview_image(image_url: str, save_path: str, timeout: int = 30) -> bool:
    """
    Download a preview image from CivitAI.

    Args:
        image_url: URL of the image to download
        save_path: Local path to save the image

    Returns:
        True if download successful
    """
    try:
        headers = {
            "User-Agent": "ComfyUI-DonutNodes/1.0"
        }
        request = urllib.request.Request(image_url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            # Create parent directory if needed
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "wb") as f:
                f.write(response.read())
            return True
    except Exception as e:
        print(f"[CivitAI] Error downloading image: {e}")
        return False


class CivitAICache:
    """
    Local cache for CivitAI model metadata.

    Stores metadata JSON files and preview images in a cache directory.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory for cache files. Defaults to
                      custom_nodes/donutnodes/civitai_cache/
        """
        if cache_dir is None:
            # Check config for custom cache dir
            cache_dir = get_civitai_cache_dir()

        if cache_dir is None:
            # Default to a cache dir in the donutnodes folder
            self.cache_dir = Path(__file__).parent.parent / "civitai_cache"
        else:
            self.cache_dir = Path(cache_dir)

        self.metadata_dir = self.cache_dir / "metadata"
        self.images_dir = self.cache_dir / "images"

        # Create directories
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def _get_metadata_path(self, file_hash: str) -> Path:
        """Get path to metadata JSON file for a hash."""
        return self.metadata_dir / f"{file_hash[:16]}.json"

    def _get_image_path(self, file_hash: str, ext: str = "jpg", index: int = 0) -> Path:
        """Get path to preview image file for a hash."""
        if index == 0:
            return self.images_dir / f"{file_hash[:16]}.{ext}"
        return self.images_dir / f"{file_hash[:16]}_{index}.{ext}"

    def get_cached_info(self, file_hash: str) -> Optional[CivitAIModelInfo]:
        """
        Get cached model info if available.

        Args:
            file_hash: The file hash to look up

        Returns:
            CivitAIModelInfo if cached, None otherwise
        """
        metadata_path = self._get_metadata_path(file_hash)
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return CivitAIModelInfo.from_dict(data)
        except (json.JSONDecodeError, OSError, TypeError) as e:
            print(f"[CivitAI Cache] Error loading cached info: {e}")
            return None

    def save_info(self, info: CivitAIModelInfo) -> bool:
        """
        Save model info to cache.

        Args:
            info: The model info to cache

        Returns:
            True if saved successfully
        """
        if not info.local_hash:
            return False

        metadata_path = self._get_metadata_path(info.local_hash)
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(info.to_dict(), f, indent=2, ensure_ascii=False)
            return True
        except OSError as e:
            print(f"[CivitAI Cache] Error saving info: {e}")
            return False

    def get_preview_image_path(self, file_hash: str) -> Optional[str]:
        """
        Get path to cached preview image if it exists.

        Args:
            file_hash: The file hash to look up

        Returns:
            Path to image file if cached, None otherwise
        """
        # Check for various image extensions
        for ext in ["jpg", "jpeg", "png", "webp"]:
            img_path = self._get_image_path(file_hash, ext)
            if img_path.exists():
                return str(img_path)
        return None

    def download_and_cache_preview(self, info: CivitAIModelInfo,
                                   prefer_sfw: bool = True) -> Optional[str]:
        """
        Download and cache preview image for a model.

        Args:
            info: Model info containing image URLs
            prefer_sfw: Prefer SFW images if available

        Returns:
            Path to downloaded image, None if failed
        """
        preview_url = info.get_preview_url(prefer_sfw=prefer_sfw)
        if not preview_url:
            return None

        # Determine extension from URL
        url_lower = preview_url.lower()
        if ".png" in url_lower:
            ext = "png"
        elif ".webp" in url_lower:
            ext = "webp"
        else:
            ext = "jpg"

        save_path = self._get_image_path(info.local_hash, ext)

        if download_preview_image(preview_url, str(save_path)):
            return str(save_path)
        return None

    def download_multiple_previews(self, info: CivitAIModelInfo,
                                   max_images: int = 4,
                                   prefer_sfw: bool = True) -> List[str]:
        """
        Download multiple preview images for a model.

        Args:
            info: Model info containing image URLs
            max_images: Maximum number of images to download
            prefer_sfw: Prefer SFW images if available

        Returns:
            List of paths to downloaded images
        """
        if not info.images:
            return []

        downloaded = []
        images_to_download = []

        # Sort images - SFW first if preferred
        if prefer_sfw:
            sfw_images = [img for img in info.images if not img.nsfw and img.nsfw_level <= 2]
            nsfw_images = [img for img in info.images if img.nsfw or img.nsfw_level > 2]
            images_to_download = (sfw_images + nsfw_images)[:max_images]
        else:
            images_to_download = info.images[:max_images]

        for idx, img in enumerate(images_to_download):
            if not img.url:
                continue

            # Determine extension from URL
            url_lower = img.url.lower()
            if ".png" in url_lower:
                ext = "png"
            elif ".webp" in url_lower:
                ext = "webp"
            else:
                ext = "jpg"

            save_path = self._get_image_path(info.local_hash, ext, index=idx)

            # Check if already downloaded
            if save_path.exists():
                downloaded.append(str(save_path))
                continue

            if download_preview_image(img.url, str(save_path)):
                downloaded.append(str(save_path))

        return downloaded

    def get_preview_collage_path(self, file_hash: str) -> Optional[str]:
        """Get path to collage image if it exists."""
        collage_path = self.images_dir / f"{file_hash[:16]}_collage.jpg"
        if collage_path.exists():
            return str(collage_path)
        return None

    def create_preview_collage(self, info: CivitAIModelInfo,
                               max_images: int = 4,
                               prefer_sfw: bool = True,
                               collage_size: tuple = (512, 512)) -> Optional[str]:
        """
        Create a 2x2 collage from preview images.

        Args:
            info: Model info containing image URLs
            max_images: Maximum images to include (up to 4)
            prefer_sfw: Prefer SFW images
            collage_size: Size of final collage (width, height)

        Returns:
            Path to collage image, None if failed
        """
        try:
            from PIL import Image
        except ImportError:
            print("[CivitAI] PIL not available for collage creation")
            return None

        # Check if collage already exists
        collage_path = self.images_dir / f"{info.local_hash[:16]}_collage.jpg"
        if collage_path.exists():
            return str(collage_path)

        # Download images if needed
        image_paths = self.download_multiple_previews(info, max_images=max_images, prefer_sfw=prefer_sfw)
        if not image_paths:
            return None

        try:
            # Load images
            images = []
            for path in image_paths[:4]:
                try:
                    img = Image.open(path)
                    img = img.convert("RGB")
                    images.append(img)
                except Exception as e:
                    print(f"[CivitAI] Error loading image {path}: {e}")

            if not images:
                return None

            # Create collage - use unique images only
            seen_sizes = set()
            unique_images = []
            for img in images[:4]:
                # Use size as a simple duplicate check
                size_key = (img.width, img.height, img.tobytes()[:1000] if img.width * img.height > 0 else b'')
                img_hash = hash(size_key)
                if img_hash not in seen_sizes:
                    seen_sizes.add(img_hash)
                    unique_images.append(img)

            images = unique_images

            cell_w = collage_size[0] // 2
            cell_h = collage_size[1] // 2
            collage = Image.new("RGB", collage_size, (32, 32, 48))

            positions = [(0, 0), (cell_w, 0), (0, cell_h), (cell_w, cell_h)]

            for idx, img in enumerate(images[:4]):
                # Resize to fit cell while maintaining aspect ratio
                img.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)

                # Center in cell
                x_offset = (cell_w - img.width) // 2
                y_offset = (cell_h - img.height) // 2

                pos_x, pos_y = positions[idx]
                collage.paste(img, (pos_x + x_offset, pos_y + y_offset))

            # Save collage
            collage.save(str(collage_path), "JPEG", quality=85)
            return str(collage_path)

        except Exception as e:
            print(f"[CivitAI] Error creating collage: {e}")
            return None

    def get_or_fetch_info(self, file_hash: str, api_key: Optional[str] = None,
                          download_preview: bool = True,
                          prefer_sfw: bool = True) -> Optional[CivitAIModelInfo]:
        """
        Get model info from cache or fetch from API.

        This is the main method to use - handles caching automatically.

        Args:
            file_hash: SHA256 hash of the LoRA file
            api_key: Optional CivitAI API key
            download_preview: Whether to download preview image
            prefer_sfw: Prefer SFW preview images

        Returns:
            CivitAIModelInfo if found (cached or fetched), None otherwise
        """
        # Check cache first
        cached = self.get_cached_info(file_hash)
        if cached is not None:
            return cached

        # Fetch from API
        info = fetch_model_by_hash(file_hash, api_key=api_key)
        if info is None:
            return None

        # Cache the result
        self.save_info(info)

        # Download preview image
        if download_preview and info.images:
            self.download_and_cache_preview(info, prefer_sfw=prefer_sfw)

        return info

    def clear_cache(self):
        """Clear all cached data."""
        import shutil
        try:
            if self.metadata_dir.exists():
                shutil.rmtree(self.metadata_dir)
            if self.images_dir.exists():
                shutil.rmtree(self.images_dir)
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
            self.images_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"[CivitAI Cache] Error clearing cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        metadata_count = len(list(self.metadata_dir.glob("*.json")))
        image_count = len(list(self.images_dir.glob("*")))

        # Calculate total size
        total_size = 0
        for f in self.cache_dir.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size

        return {
            "metadata_count": metadata_count,
            "image_count": image_count,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir)
        }


# Global cache instance
_global_cache: Optional[CivitAICache] = None


def get_cache() -> CivitAICache:
    """Get or create the global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = CivitAICache()
    return _global_cache
