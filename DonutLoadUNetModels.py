import os, glob, torch
from diffusers import StableDiffusionXLPipeline

class DonutLoadUNetModels:
    """
    Loads U-Net modules from all checkpoints in a directory or glob.
    Outputs a MODELLIST of bare nn.Modules (the unets).
    """
    class_type = "CUSTOM"
    aux_id     = "DonutsDelivery/ComfyUI-DonutNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "path":        ("STRING",),
            "torch_dtype": ("STRING", {"default":"float16"}),
        }}
    RETURN_TYPES = ("MODELLIST",)
    FUNCTION     = "execute"
    CATEGORY     = "donut"

    def execute(self, path, torch_dtype):
        # expand folder or glob:
        if os.path.isdir(path):
            files = sorted(
                glob.glob(os.path.join(path, "*.safetensors")) +
                glob.glob(os.path.join(path, "*.ckpt"))
            )
        else:
            files = sorted(glob.glob(path))

        unets = []
        for fn in files:
            pipe = StableDiffusionXLPipeline.from_single_file(
                fn,
                torch_dtype=getattr(torch, torch_dtype),
                safe_serialization=True,
            )
            unet = pipe.unet
            # discard everything else to free GPU
            del pipe.scheduler, pipe.vae, pipe.text_encoder, pipe.text_encoder_2
            torch.cuda.empty_cache()
            unets.append(unet)
        return (unets,)

NODE_CLASS_MAPPINGS = {"DonutLoadUNetModels": DonutLoadUNetModels}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutLoadUNetModels": "Donut Load U-NET Models"}
