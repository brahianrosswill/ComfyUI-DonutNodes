import re
from collections import defaultdict
from tqdm import tqdm
import traceback
import torch
import torch.nn as nn

from .task_vector import TaskVector
from .mask_weights_utils import mask_model_weights
from .utils.utils import get_param_names_to_merge


class MergingMethod:
    def __init__(self, merging_method_name: str, vram_limit_bytes: int = None):
        """
        merging_method_name: name used in logs
        vram_limit_bytes: if set, will only merge on GPU when free VRAM ≥ this
        """
        self.method      = merging_method_name
        self.vram_limit  = vram_limit_bytes

    def _choose_device(self):
        if torch.cuda.is_available() and self.vram_limit is not None:
            dev = torch.cuda.current_device()
            free, _ = torch.cuda.mem_get_info(dev)
            if free >= self.vram_limit:
                return torch.device("cuda")
        return torch.device("cpu")

    def widen_merging(
        self,
        merged_model: nn.Module,
        models_to_merge: list,
        exclude_param_names_regex: list,
        above_average_value_ratio: float = 1.0,
        score_calibration_value: float = 1.0,
    ):
        """
        Widen merging with automatic CPU/GPU offload and per-param fallbacks.
        Returns merged parameter dict.
        """
        device = self._choose_device()
        print(f"[{self.method}] merging on {device}")

        # 1) snapshot original parameters on CPU/float32
        pre_params = {n: p.detach().cpu().float().clone()
                      for n, p in merged_model.named_parameters()}
        finetuned_dicts = [
            {n: p.detach().cpu().float().clone() for n, p in m.named_parameters()}
            for m in models_to_merge
        ]

        # 2) transpose token embeddings if present
        def transpose_tok(d):
            if "model.embed_tokens.weight" in d:
                d["model.embed_tokens.weight"] = d["model.embed_tokens.weight"].T
        transpose_tok(pre_params)
        for d in finetuned_dicts:
            transpose_tok(d)

        # 3) build TaskVectors to extract deltas
        task_vectors = [
            TaskVector(merged_model, m, exclude_param_names_regex)
            for m in models_to_merge
        ]

        # 4) compute magnitude & direction for each param
        def compute_mag_dir(param_dict, desc):
            mags, dirs = {}, {}
            for name, tensor in tqdm(param_dict.items(), desc=desc):
                try:
                    if tensor.dim() not in (2,4):  # only 2D or conv4D
                        continue
                    if tensor.dim() == 4:
                        o,c,h,w = tensor.shape
                        flat = tensor.view(o, -1)
                    else:
                        flat = tensor
                    mag = flat.norm(dim=0)
                    dir = flat / (mag.unsqueeze(0) + 1e-8)
                    dirs[name] = dir.view(tensor.shape) if tensor.dim()==4 else dir
                    mags[name] = mag
                except Exception:
                    continue
            return mags, dirs

        pre_mag, pre_dir = compute_mag_dir(pre_params,   "[mag/dir] pretrained")
        diff_list = []
        for fin in finetuned_dicts:
            fin_mag, fin_dir = compute_mag_dir(fin, "[mag/dir] finetuned")
            mag_diff = {k:(fin_mag[k] - pre_mag[k]).abs() for k in pre_mag if k in fin_mag}
            dir_diff = {k:1 - torch.cosine_similarity(fin_dir[k], pre_dir[k], dim=0)
                        for k in pre_dir if k in fin_dir}
            diff_list.append((mag_diff, dir_diff))

        # 5) helper to rank & score
        def rank_sig(diff: torch.Tensor):
            if diff.ndim != 2:
                raise IndexError
            n,dim = diff.shape
            flat  = diff.reshape(n,-1)
            idx   = torch.argsort(flat, dim=1)
            L     = flat.shape[1]
            sig   = torch.arange(L, device=flat.device)/L
            base  = sig.unsqueeze(0).repeat(n,1)
            return base.scatter(1, idx, sig)

        def importance(sig: torch.Tensor):
            sc = torch.softmax(sig, dim=0)
            avg= sig.mean(1, keepdim=True)
            mask = sig > avg * above_average_value_ratio
            sc[mask] = score_calibration_value
            return sc

        def merge_param(delta, base, mag_rank, dir_rank):
            try:
                ms  = importance(mag_rank)
                ds  = importance(dir_rank)
                w   = 0.5 * (ms + ds)
                if delta.dim() == 3:
                    w = w.unsqueeze(1)
                elif delta.dim() == 5:
                    n,dim = w.shape
                    w = w.view(n,1,dim,1,1)
                else:
                    shape = [1]*(delta.dim()-2)+[w.shape[1]]
                    w = w.view(delta.shape[0], *shape)
                merged = base + (delta * w).sum(0)
                return merged if merged.shape == base.shape else base
            except Exception:
                return base

        merged_params = {}
        fell_back = 0
        common = set(pre_mag.keys())
        for tv in task_vectors:
            common &= set(tv.task_vector_param_dict.keys())

        for name in tqdm(common, desc=f"[{self.method}] merging"):
            try:
                delta = torch.stack([tv.task_vector_param_dict[name] for tv in task_vectors])
                magd  = torch.stack([d[0][name] for d in diff_list])
                dird  = torch.stack([d[1][name] for d in diff_list])
                rankm = rank_sig(magd)
                rankd = rank_sig(dird)
            except Exception:
                merged_params[name] = pre_params[name]
                fell_back += 1
                continue
            merged = merge_param(delta, pre_params[name], rankm, rankd)
            merged_params[name] = merged
            if torch.allclose(merged, pre_params[name]):
                fell_back += 1

        total = len(common)
        print(f"[{self.method}] merged {total - fell_back} / {total} parameters")
