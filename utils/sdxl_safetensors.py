import os
import logging
from typing import Dict, List, Literal, Optional

import torch
from safetensors.torch import load_file, save_file

TensorDict = Dict[str, torch.Tensor]
DeviceType = Literal["cpu", "cuda"]


def ensure_same_device(tensors: TensorDict, target_device: DeviceType) -> TensorDict:
    """Move all tensors in the dictionary to the target device."""
    return {k: v.to(target_device) for k, v in tensors.items()}


def load_safetensors_model(
    model_path: str,
    device: DeviceType = "cpu",
    dtype: Optional[torch.dtype] = torch.float16,
    logger_info: bool = True,
) -> TensorDict:
    """Load a complete SDXL model from a safetensors file."""
    if logger_info:
        logging.info(f"Loading model from {model_path}...")
    try:
        model_tensors = load_file(model_path, device=device)
        if dtype is not None:
            model_tensors = {k: v.to(dtype=dtype) for k, v in model_tensors.items()}
        model_tensors = ensure_same_device(model_tensors, device)
        if logger_info:
            logging.info(f"Loaded model with {len(model_tensors)} tensors on {device}")
        return model_tensors
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        raise


def get_module_keys(base_model: TensorDict, modules: List[str]) -> Dict[str, List[str]]:
    """Separate SDXL model tensors by component type using key name patterns."""
    all_keys = set(base_model.keys())
    module_keys: Dict[str, List[str]] = {}
    prefix_map: Dict[str, str] = {
        "unet": "model.diffusion_model.",
        "text_encoder": "cond_stage_model.transformer.",
        "text_encoder_alt": "conditioner.embedders.0.transformer.",
        "text_encoder_2": "conditioner.embedders.1.model.",
        "vae": "first_stage_model.",
    }
    for module in modules:
        if module in prefix_map:
            prefix = prefix_map[module]
            filtered_keys = [k for k in all_keys if k.startswith(prefix)]
            if module == "text_encoder" and not filtered_keys:
                alt_prefix = prefix_map["text_encoder_alt"]
                filtered_keys = [k for k in all_keys if k.startswith(alt_prefix)]
                if filtered_keys:
                    logging.info(f"Using alternate text encoder format: {alt_prefix}")
            module_keys[module] = filtered_keys
            logging.info(f"Found {len(filtered_keys)} keys for module {module}")
        else:
            logging.warning(f"Unknown module: {module}")
            module_keys[module] = []
    return module_keys


def save_safetensors_model(model_tensors: TensorDict, output_path: str, ensure_cpu: bool = True) -> None:
    """Save SDXL model tensors to a safetensors file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    try:
        if ensure_cpu:
            cpu_tensors = {k: v.cpu() for k, v in model_tensors.items()}
            save_file(cpu_tensors, output_path)
        else:
            save_file(model_tensors, output_path)
        logging.info(f"Saved model to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save model to {output_path}: {e}")
        raise
