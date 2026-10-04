"""EPI — Efficient Parameter Isolation (port of Dicer-Zz/EPI).

Parameter isolation: each task trains its OWN private adapter (others frozen) and the
adapters are SELECTED at inference, not merged. To pick the right adapter without task
IDs, EPI models each task's frozen-backbone pooled feature as a Gaussian with task mean
μ_k and a shared (tied) covariance Σ, then routes each test input to
    argmin_k (x − μ_k)ᵀ Σ⁻¹ (x − μ_k)      (Mahalanobis task identifier)
and runs that task's adapter.

Original EPI is encoder-classification; here the private module is a per-task LoRA
adapter (managed by multi_adapter.CLLoRAManager) and the router selects which adapter
to activate for each test example before generation.

Local now: syntax only. Eval wiring (per-example adapter switch) goes in the engine.
"""
import torch


def router_training_indices(rows, streams, task_id):
    """Indices for current-task router fitting, excluding old-only replay rows."""
    old_types = {event_type for stream in streams[:task_id] for event_type in stream}
    included = []
    for index, row in enumerate(rows):
        response = row["response"]
        if isinstance(response, str):
            import json
            response = json.loads(response)
        event_types = {
            event[1]
            for event in response.get("events", [])
            if isinstance(event, list) and len(event) >= 2
        }
        if event_types and event_types <= old_types:
            continue
        included.append(index)
    return included


def router_evaluation_indices(rows, task_types):
    """Rows containing at least one event from the task being diagnosed."""
    task_types = set(task_types)
    included = []
    for index, row in enumerate(rows):
        response = row["response"]
        if isinstance(response, str):
            import json
            response = json.loads(response)
        event_types = {
            event[1]
            for event in response.get("events", [])
            if isinstance(event, list) and len(event) >= 2
        }
        if event_types & task_types:
            included.append(index)
    return included


class MahalanobisRouter:
    """Per-task Gaussian means + tied covariance; routes inputs to the nearest task."""

    def __init__(self, shrinkage=1e-3):
        self.means = []            # list of (d,) task means, in task order
        self.scatter = None        # (d,d) accumulated within-task scatter -> tied cov
        self.n_total = 0
        self.d = None
        self.shrinkage = float(shrinkage)
        self._prec = None          # cached Σ⁻¹

    @torch.no_grad()
    def add_task(self, feats):
        """feats: (N, d) frozen-backbone pooled features for the just-finished task."""
        feats = feats.float()
        mu = feats.mean(dim=0)
        centered = feats - mu
        S = centered.t() @ centered
        self.means.append(mu)
        self.scatter = S if self.scatter is None else self.scatter + S
        self.n_total += feats.shape[0]
        self.d = feats.shape[1]
        self._prec = None

    @torch.no_grad()
    def _precision(self):
        if self._prec is None:
            cov = self.scatter / max(1, self.n_total)
            cov = cov + self.shrinkage * torch.eye(self.d, device=cov.device, dtype=cov.dtype)
            self._prec = torch.linalg.inv(cov)
        return self._prec

    @torch.no_grad()
    def predict(self, feat):
        """feat: (d,) or (B,d) frozen-backbone pooled feature -> task id(s) (B,)."""
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        feat = feat.float()
        dev = feat.device                                    # router state may be on CPU
        P = self._precision().to(dev)
        mus = torch.stack(self.means, dim=0).to(dev)         # (T, d)
        diff = feat.unsqueeze(1) - mus.unsqueeze(0)          # (B, T, d)
        dist = torch.einsum("btd,de,bte->bt", diff, P, diff)  # (B, T) Mahalanobis
        return torch.argmin(dist, dim=1)                     # (B,)

    def state_dict(self):
        return {"means": [m.cpu() for m in self.means], "scatter": None if self.scatter is None
                else self.scatter.cpu(), "n_total": self.n_total, "d": self.d}

    def load_state_dict(self, sd, device="cpu"):
        self.means = [mean.to(device) for mean in sd["means"]]
        self.scatter = None if sd["scatter"] is None else sd["scatter"].to(device)
        self.n_total = sd["n_total"]
        self.d = sd["d"]
        self._prec = None
