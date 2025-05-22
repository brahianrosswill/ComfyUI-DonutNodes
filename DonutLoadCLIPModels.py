import os, glob, torch
from diffusers import StableDiffusionXLPipeline

class DonutLoadCLIPModels:
    """
    Loads CLIP (text-encoder) modules from all checkpoints in a directory or glob.
    Outputs a CLIPLIST of bare nn.Modules.
    """
    class_type = "CUSTOM"
    aux_id     = "DonutsDelivery/ComfyUI-DonutNodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "path":        ("STRING",),
            "torch_dtype": ("STRING", {"default":"float16"}),
        }}
    RETURN_TYPES = ("CLIPLIST",)
    FUNCTION     = "execute"
    CATEGORY     = "donut"

    def execute(self, path, torch_dtype):
        if os.path.isdir(path):
            files = sorted(
                glob.glob(os.path.join(path, "*.safetensors")) +
                glob.glob(os.path.join(path, "*.ckpt"))
            )
        else:
            files = sorted(glob.glob(path))

        clips = []
        for fn in files:
            pipe = StableDiffusionXLPipeline.from_single_file(
                fn,
                torch_dtype=getattr(torch, torch_dtype),
                safe_serialization=True,
            )
            clip = pipe.text_encoder
            # discard the rest
            del pipe.unet, pipe.scheduler, pipe.vae, pipe.text_encoder_2
            torch.cuda.empty_cache()
            clips.append(clip)
        return (clips,)

NODE_CLASS_MAPPINGS = {"DonutLoadCLIPModels": DonutLoadCLIPModels}
NODE_DISPLAY_NAME_MAPPINGS = {"DonutLoadCLIPModels": "Donut Load CLIP Models"}
