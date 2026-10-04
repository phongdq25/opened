"""GainLoRA — gated integration of per-task LoRA branches (port of liangyanshuo/gainlora).

Expandable LoRA (one frozen-later branch per task), but branches are combined by an
INPUT-CONDITIONED gate instead of a fixed sum:
    W_t(x) = W0 + Σ_i g_i(x) · A_i B_i ,   g_i(x) ∈ [0,1].
Each branch has a small gating MLP designed to output ~0 on old-task inputs (new
branch silent on old tasks) and ~1 on its own task, via:
  (a) f(b) = |2·σ(b) − 1|  so f(0)=0, and
  (b) constraining the gate HEAD to the orthogonal complement of old-task gate-feature
      space (GPM):  Init(G_{L+1}) ⟂ M ,  ΔG_{L+1} ⟂ M.
Two variants: base branches trained with O-LoRA orth loss (see multi_adapter.orth_loss)
OR InfLoRA designed B_t (see inflora.design_B), PLUS this gate.

Local now: syntax only. Forward integration (weight each branch by its gate) needs a
custom multi-adapter forward, wired in the engine/layer — not stock PEFT.
"""
import torch
import torch.nn as nn

from cl_lora.multi_adapter import lora_layers
from cl_lora.inflora import orthonormal_residual


def gain_activation(b):
    """f(b) = |2·sigmoid(b) − 1|  in [0,1], with f(0)=0."""
    return torch.abs(2.0 * torch.sigmoid(b) - 1.0)


class GateNet(nn.Module):
    """Per-branch gate: pooled feature (in_dim) -> scalar gate in [0,1] via L sigmoid layers."""

    def __init__(self, in_dim, hidden=64, n_layers=2):
        super().__init__()
        dims = [in_dim] + [hidden] * n_layers
        self.hidden = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers))
        self.head = nn.Linear(dims[-1], 1)                # G_{L+1}, GPM-constrained

    def features(self, feat):
        p = feat
        for layer in self.hidden:
            p = torch.sigmoid(layer(p))
        return p                                          # (B, hidden) -> head input

    def forward(self, feat):                              # feat: (B, in_dim) pooled tokens
        b = self.head(self.features(feat))                # (B, 1)
        return gain_activation(b).squeeze(-1)             # (B,) gate in [0,1]


class GPMGate:
    """Old-task gate-head input subspace M; keeps the gate head init/updates ⟂ M."""

    def __init__(self):
        self.M = None                                     # (hidden, k) orthonormal basis

    @torch.no_grad()
    def init_head(self, head: nn.Linear):
        """Project the head's initial weight onto the complement of M  (Init ⟂ M)."""
        if self.M is None:
            return
        w = head.weight.data                              # (1, hidden)
        head.weight.data = w - (w @ self.M) @ self.M.t()

    @torch.no_grad()
    def project_grad(self, head: nn.Linear):
        """Remove the component of head.weight.grad inside M  (ΔG ⟂ M). Call after backward."""
        g = getattr(head.weight, "grad", None)
        if self.M is None or g is None:
            return
        head.weight.grad = g - (g @ self.M) @ self.M.t()

    @torch.no_grad()
    def grow(self, feats):
        """Grow M with principal directions of this task's gate-head input feats (N, hidden)."""
        C = feats.t().float() @ feats.float()
        U, S, _ = torch.linalg.svd(C, full_matrices=False)
        energy = torch.cumsum(S, dim=0) / (S.sum() + 1e-12)
        k = int((energy < 0.95).sum().item()) + 1
        newM = U[:, :k]
        if self.M is None:
            self.M = newM
            return
        add = orthonormal_residual(self.M, newM, max_dim=self.M.shape[0])
        if add is not None:
            self.M = torch.cat([self.M, add], dim=1)

    def state_dict(self):
        return {"M": None if self.M is None else self.M.cpu()}

    def load_state_dict(self, state_dict, device=None):
        basis = state_dict.get("M")
        self.M = basis.to(device) if basis is not None and device is not None else basis


def pool_tokens(hidden_states, attention_mask=None):
    """Mean-pool token hidden states -> (B, d) gate input feature."""
    h = hidden_states.float()
    if attention_mask is None:
        return h.mean(dim=1)
    m = attention_mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)


class GainGates:
    """Input-conditioned gating over Family-A branches, realised without patching PEFT.

    Each LoRA layer already outputs  base(x) + Σ_active scaling·B_i A_i x. A forward
    hook recomputes each active branch's delta and adds (g_i(x) − 1)·delta_i, turning
    the sum into  base(x) + Σ_i g_i(x)·A_iB_i x  (GainLoRA). With lora_dropout=0 the
    recomputed delta equals PEFT's, so both the forward and the A/B/gate gradients are
    exact. g_i(x)=GateNet_i(pooled input embedding); the current branch's gate head is
    kept ⟂ the old-task gate-feature subspace (GPM), so a new branch stays silent on
    old inputs.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.in_dim = model.config.hidden_size
        self.gatenets = nn.ModuleDict()          # adapter_name -> GateNet
        self.gpm = GPMGate()                      # shared old-task gate-feature subspace
        self.gates = {}                           # adapter_name -> (B,) gate for current batch
        self.current = None
        self._feat_buf = []                       # current task's gate-head-input features
        self.hooks = []
        for _, mod in lora_layers(model):
            self.hooks.append(mod.register_forward_hook(self._make_hook(mod)))

    def _make_hook(self, module):
        def hook(_module, inputs, output):
            if not self.gates:
                return output
            x = inputs[0]
            out = output
            la = getattr(module, "lora_A", {})
            for adp, g in self.gates.items():
                if adp not in la or g.shape[0] != x.shape[0]:
                    continue                      # skip stale gates (e.g. InfLoRA calibration pass)
                A, B = module.lora_A[adp], module.lora_B[adp]
                # PEFT keeps LoRA weights in fp32 under a bf16 base; cast x in and
                # the correction back out, matching PEFT's own forward.
                delta = B(A(x.to(A.weight.dtype))) * module.scaling.get(adp, 1.0)
                gg = g.to(delta.dtype).view(-1, *([1] * (delta.dim() - 1)))
                out = out + ((gg - 1.0) * delta).to(out.dtype)
            return out
        return hook

    def add_branch(self, name):
        """New per-task gate; init its head ⟂ old-task subspace; train only this gate."""
        net = GateNet(self.in_dim).to(self.device)
        self.gatenets[name] = net
        self.gpm.init_head(net.head)
        self.current = name
        self._feat_buf = []
        for n, g in self.gatenets.items():
            for p in g.parameters():
                p.requires_grad_(n == name)

    def set_batch_gates(self, input_ids, attention_mask):
        """Compute g_i(x) for every branch from the mean-pooled (frozen) input embedding."""
        with torch.no_grad():
            emb = self.model.get_input_embeddings()(input_ids)
        feat = pool_tokens(emb, attention_mask).detach()          # (B, d)
        self.gates = {adp: net(feat) for adp, net in self.gatenets.items()}
        if self.current is not None and self.model.training:
            self._feat_buf.append(self.gatenets[self.current].features(feat).detach())

    def clear_gates(self):
        """Drop any batch gates (call before non-gated forwards, e.g. InfLoRA calibration)."""
        self.gates = {}

    def project_grads(self):
        if self.current is not None:
            self.gpm.project_grad(self.gatenets[self.current].head)

    def end_task(self, name):
        """Grow the GPM subspace with this task's gate features; freeze this gate."""
        if self._feat_buf:
            self.gpm.grow(torch.cat(self._feat_buf, dim=0))
        for p in self.gatenets[name].parameters():
            p.requires_grad_(False)
        self.gates = {}

    def parameters(self):
        return [p for p in self.gatenets.parameters() if p.requires_grad]
