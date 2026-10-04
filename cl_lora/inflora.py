"""InfLoRA — Interference-Free Low-Rank Adaptation (port of liangyanshuo/InfLoRA).

Per task: one new LoRA branch (old branches frozen + merged). The DOWN-projection
B_t (r x in) is DESIGNED, not learned, so that training the UP-projection A_t
(out x r) updates W only inside a subspace orthogonal to previous tasks' input
space; old-task outputs stay undisturbed. No extra loss term — the interference-free
constraint is baked into the fixed B_t.

Per task t, per target linear layer:
  1. collect the layer's input activations H_t  (in x N)  over a calibration pass.
  2. maintain the old-task input subspace basis M via DualGPM; complement P = I - M Mᵀ.
  3. Ĥ = P·H_t ; SVD Ĥ = U S Vᵀ ; B_t = U[:, :r]ᵀ  (top-r input dirs ⟂ old tasks).
     freeze B_t, train A_t.
  4. after the task, grow M with H_t's principal directions (DualGPM update).

Local now: syntax only; activation-collection + SVD path needs server torch to verify.
"""
import torch
import torch.nn as nn


def orthonormal_residual(existing, candidates, max_dim=None):
    """Return significant candidate directions orthogonal to an existing basis."""
    if candidates is None or candidates.numel() == 0:
        return None
    residual = candidates.float()
    if existing is not None and existing.numel() > 0:
        basis = existing.float()
        residual = residual - basis @ (basis.t() @ residual)
        residual = residual - basis @ (basis.t() @ residual)
    U, S, _ = torch.linalg.svd(residual, full_matrices=False)
    if S.numel() == 0 or not torch.isfinite(S).all():
        return None
    tolerance = max(residual.shape) * torch.finfo(S.dtype).eps * S.max()
    rank = int((S > max(float(tolerance), 1e-6)).sum().item())
    if existing is not None:
        rank = min(rank, residual.shape[0] - existing.shape[1])
    if max_dim is not None:
        current_dim = 0 if existing is None else existing.shape[1]
        rank = min(rank, max_dim - current_dim)
    if rank <= 0:
        return None
    return U[:, :rank].to(candidates.device)


class ActivationCollector:
    """Forward-pre-hooks that accumulate per-layer input covariance H Hᵀ (in x in)."""

    def __init__(self, layers_by_name):
        self.cov = {}                       # name -> (in, in) accumulated XᵀX
        self.count = {}
        self.handles = []
        for name, module in layers_by_name.items():
            self.handles.append(module.register_forward_pre_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, args):
            x = args[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()          # (N, in)
            c = x.t() @ x                                   # (in, in)
            self.cov[name] = c if name not in self.cov else self.cov[name] + c
            self.count[name] = self.count.get(name, 0) + x.shape[0]
        return hook

    def basis(self, name, energy_keep=0.99):
        """Top input directions of the collected covariance -> (in, k)."""
        C = self.cov.get(name)
        if C is None:
            return None
        U, S, _ = torch.linalg.svd(C, full_matrices=False)  # C SPD -> U columns are input dirs
        energy = torch.cumsum(S, dim=0) / (S.sum() + 1e-12)
        k = int((energy < energy_keep).sum().item()) + 1
        return U[:, :k]

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class DualGPM:
    """Per-layer old-task input subspace basis M (in x k). Complement = I - M Mᵀ."""

    def __init__(self, energy_keep=0.95, max_dim=None):
        self.M = {}
        self.energy_keep = energy_keep
        self.max_dim = max_dim

    def complement_project(self, name, H):
        """Project columns of H (in x *) onto the orthogonal complement of M[name]."""
        M = self.M.get(name)
        if M is None:
            return H
        return H - M @ (M.t() @ H)

    @torch.no_grad()
    def grow(self, name, U_new):
        """Add new orthogonal input directions U_new (in x k') not already in M."""
        current = self.M.get(name)
        add = orthonormal_residual(current, U_new, self.max_dim)
        if add is None:
            return
        self.M[name] = add if current is None else torch.cat([current, add], dim=1)

    def state_dict(self):
        return {"M": {name: basis.cpu() for name, basis in self.M.items()}}

    def load_state_dict(self, state_dict, device=None):
        self.M = {
            name: basis.to(device) if device is not None else basis
            for name, basis in state_dict.get("M", {}).items()
        }


@torch.no_grad()
def design_B(name, input_basis, dualgpm, r):
    """Design the fixed down-projection B_t (r x in): top-r input dirs ⟂ old tasks.

    input_basis: (in x k) principal input directions of the current task (from
    ActivationCollector.basis). dualgpm: old-task subspace. Returns (r, in).
    """
    Hc = dualgpm.complement_project(name, input_basis).float()  # (in, k)
    U, S, _ = torch.linalg.svd(Hc, full_matrices=False)
    output = torch.zeros(r, Hc.shape[0], device=input_basis.device, dtype=input_basis.dtype)
    if S.numel() == 0 or not torch.isfinite(S).all():
        return output
    tolerance = max(Hc.shape) * torch.finfo(S.dtype).eps * S.max()
    r_eff = min(r, int((S > max(float(tolerance), 1e-6)).sum().item()))
    if r_eff > 0:
        output[:r_eff] = U[:, :r_eff].t().to(output.dtype)
    return output.contiguous()


def target_linear_layers(model, name_filter=None):
    """Map name -> nn.Linear for the layers LoRA targets (q/k/v/o/gate/up/down proj)."""
    keys = name_filter or ("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj")
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(k in name for k in keys):
            out[name] = module
    return out
