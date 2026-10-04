"""MIGU+FT — MagnItude-based Gradient Updating for continual full fine-tuning.

Ref: wenyudu/MIGU, arXiv:2406.17245. Rehearsal-free and task-id-free. During the
forward pass, cache each linear layer's per-output-feature magnitude; at the
optimizer step, update only the weights tied to the TOP-T fraction of output
features (zero the gradient of the rest). Default T=0.7. The "+FT" variant masks
the model's own linear weights (full fine-tuning), not LoRA.

    n = L1-normalized per-output magnitude ;  k = floor(T * d_out)
    M = top-k(n)  ;  W <- W - lr * (M ⊙ ∇W)   (M broadcasts over input dim)

Deepspeed note: call mask_grads() AFTER backward, BEFORE step(). Under ZeRO-2/3 the
grads may be partitioned/None on .weight.grad — for MIGU+FT run without grad
partitioning (ZeRO-0/1) or hook deepspeed's grad. Verified path: plain DDP/ZeRO-0.
Local now: syntax only.
"""
import torch
import torch.nn as nn


class MIGU:
    def __init__(self, model, ratio=0.7, target_types=(nn.Linear,)):
        self.model = model
        self.ratio = float(ratio)
        self.mags = {}                 # module -> running per-output magnitude (d_out,)
        self.handles = []
        self.layers = [m for m in model.modules() if isinstance(m, target_types)]
        for m in self.layers:
            self.handles.append(m.register_forward_hook(self._hook))

    def _hook(self, module, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        o = o.detach()
        mag = o.abs().reshape(-1, o.shape[-1]).mean(dim=0).float()   # (d_out,)
        mag = mag / (mag.sum() + 1e-8)                               # L1-normalize
        prev = self.mags.get(module)
        self.mags[module] = mag if prev is None else 0.5 * (prev + mag)

    @torch.no_grad()
    def mask_grads(self):
        """Zero grads of output features NOT in the top-T fraction (call after backward)."""
        for m in self.layers:
            mag = self.mags.get(m)
            g = getattr(m.weight, "grad", None)
            if mag is None or g is None:
                continue
            d_out = mag.shape[0]
            k = max(1, int(self.ratio * d_out))
            keep = torch.zeros(d_out, dtype=torch.bool, device=g.device)
            keep[torch.topk(mag.to(g.device), k).indices] = True
            g[~keep, :] = 0.0
            b = getattr(m, "bias", None)
            if b is not None and b.grad is not None:
                b.grad[~keep] = 0.0

    def clear(self):
        self.mags.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
