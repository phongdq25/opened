"""TreeLoRA — hierarchical bandit-routed LoRA regulariser (port of ZinYY/TreeLoRA).

Builds a binary tree over previous tasks' adapter signatures; per training step a
hierarchical LCB bandit picks one relevant previous task PER LoRA layer, and a
regulariser pulls the current adapter toward those selected previous adapters.

Signature = the CURRENT task's lora_A weights, one matrix per LoRA layer, stacked
to (lora_depth, feat), collected BEFORE backward (differentiable). Faithful to the
repo: L1-distance + LCB, median-split tree, consecutive-difference before build,
reg = -Σ_layer <A_cur, A_prev>, normalised to the LM-loss scale × ramped tmp_reg.
Train step:  loss = loss - reg.   reg coefficient = args.reg (script default 0.5).

Local now: syntax only; verify numerics on server.
"""
import copy
import math
import torch


def signature_from_model(model, adapter_name):
    """Stack the given adapter's lora_A weights per layer -> (lora_depth, feat).

    Layers can differ in in_features (q/k/v vs o vs down_proj on Qwen3), so the
    flattened vectors have different lengths. Zero-pad to the max length before
    stacking: the tree/reg only ever dot-products the SAME layer index across
    tasks (identical true length, identically padded), so the padding zeros never
    affect a dot product, and torch.stack gets a rectangular tensor.
    """
    from cl_lora.multi_adapter import lora_layers
    import torch.nn.functional as F
    vecs = []
    for _, module in lora_layers(model):
        la = getattr(module, "lora_A", {})
        if adapter_name in la:
            vecs.append(la[adapter_name].weight.reshape(-1))
    if not vecs:
        raise ValueError(f"no lora_A weights for adapter '{adapter_name}'")
    m = max(v.numel() for v in vecs)
    vecs = [F.pad(v, (0, m - v.numel())) for v in vecs]
    return torch.stack(vecs, dim=0)                     # (lora_depth, max_feat)


class KDTreeNode:
    def __init__(self, task_indices, depth, grads_tensor, lora_depth):
        self.task_indices = list(task_indices)
        self.depth = depth
        self.lora_depth = lora_depth
        self.is_leaf = False
        self.left = self.right = None
        self.mean_vector = None
        self.median_similarity = 1.0
        self._build(grads_tensor)

    def _build(self, grads_tensor):
        if self.depth >= self.lora_depth or len(self.task_indices) <= 1:
            self.is_leaf = True
            return
        cur = grads_tensor[self.task_indices, self.depth, :]          # (N, D)
        self.mean_vector = cur.mean(dim=0)
        sims = torch.mv(cur, self.mean_vector)                        # (N,)
        self.median_similarity = torch.median(sims).item()
        left = [t for i, t in enumerate(self.task_indices) if sims[i] >= self.median_similarity]
        right = [t for i, t in enumerate(self.task_indices) if sims[i] < self.median_similarity]
        if not left or not right:
            self.is_leaf = True
            return
        self.left = KDTreeNode(left, self.depth + 1, grads_tensor, self.lora_depth)
        self.right = KDTreeNode(right, self.depth + 1, grads_tensor, self.lora_depth)


def tree_lora_loss(current_grad, all_grad, prev_id_matrix):
    reg = None
    for depth_id, prev in enumerate(prev_id_matrix):
        term = -(current_grad[depth_id] * all_grad[int(prev)][depth_id]).sum()
        reg = term if reg is None else reg + term
    return reg


class KDLoRATree:
    """Bandit + tree state across a task sequence."""

    def __init__(self, num_tasks, reg=0.5):
        self.num_tasks = int(num_tasks)
        self.reg = float(reg)
        self.all_grad = None            # (t, lora_depth, feat) stored task signatures
        self.current_grad = None        # running mean of current task's signature
        self.sim = None                 # (num_tasks, lora_depth) bandit reward accumulator
        self.num_of_selected = None     # (num_tasks, lora_depth) arm pull counts
        self.kd_tree_root = None
        self.total_rounds = 1
        self.tmp_rounds = 0
        self.tmp_reg = 0.0
        self.lora_depth = None

    def new_epoch_init(self, total_rounds):
        self.total_rounds = max(1, int(total_rounds))
        self.tmp_rounds = 0
        self.current_grad = None

    def step(self):
        self.tmp_rounds += 1
        self.tmp_reg = self.reg * self.tmp_rounds / self.total_rounds

    def insert_grad(self, grad):        # (lora_depth, feat)
        g = grad.detach()
        self.lora_depth = g.shape[0]
        add = g / self.total_rounds
        self.current_grad = add if self.current_grad is None else self.current_grad + add

    def _ensure_bandit(self, device):
        if self.sim is None:
            self.sim = torch.zeros(self.num_tasks, self.lora_depth, device=device)
            self.num_of_selected = torch.zeros(self.num_tasks, self.lora_depth, device=device)

    def update_similarity(self, prev_id_matrix, device):
        for depth_id, prev in enumerate(prev_id_matrix):
            self.sim[int(prev), depth_id] -= torch.sum(torch.abs(
                self.current_grad[depth_id] - self.all_grad[int(prev)][depth_id])).item()

    def tree_search(self, task_id, device):
        self._ensure_bandit(device)
        sim = self.sim.clone()
        n = self.num_of_selected[:task_id, :]
        mask = n > 0
        sim[:task_id][mask] = sim[:task_id][mask] / n[mask]
        bonus = (1.0 / torch.sqrt(2 * n + 1e-5)) * math.sqrt(
            math.log(2 * self.total_rounds * (self.tmp_rounds + 1) * (self.tmp_rounds + 2)))
        sim[:task_id, :] = sim[:task_id, :] - bonus         # LCB (minimising distance)
        sim = -sim                                          # distance -> reward
        sim = sim + torch.min(sim)
        first = torch.multinomial(torch.softmax(torch.sum(sim, dim=1), dim=0), 1, True).item()
        root = self.kd_tree_root
        if root is not None and root.left is not None:
            if first in root.left.task_indices:
                sim[root.left.task_indices] *= min(root.left.median_similarity, 1.5)
            elif root.right is not None:
                sim[root.right.task_indices] *= min(root.right.median_similarity, 1.5)
        sim = sim / (torch.max(sim) - torch.min(sim) + 1e-5)
        sim[task_id:, :] = -float("inf")                    # route only to earlier tasks
        probs = torch.softmax(sim, dim=0)                   # (num_tasks, lora_depth)
        prev = torch.multinomial(probs.T, 1, True).reshape(-1)
        self.num_of_selected[prev, torch.arange(self.lora_depth, device=device)] += 1
        self.update_similarity(prev, device)
        return prev

    def get_loss(self, current_grad, loss, task_id, prev_id_matrix):
        reg = tree_lora_loss(current_grad, self.all_grad, prev_id_matrix)
        reg = reg / (reg.detach().clone() + 1e-5) * loss.detach().clone() * self.tmp_reg
        return reg

    def end_task(self, task_id):
        sig = self.current_grad.detach().unsqueeze(0)       # (1, lora_depth, feat)
        self.all_grad = sig if self.all_grad is None else torch.cat([self.all_grad, sig], dim=0)
        self._rebuild_tree()

    def _rebuild_tree(self):
        if self.all_grad is None:
            self.kd_tree_root = None
            return
        grads = copy.deepcopy(self.all_grad)
        for i in range(grads.shape[0] - 1, 0, -1):          # consecutive differencing
            grads[i] = grads[i] - grads[i - 1]
        ids = list(range(grads.shape[0]))
        self.kd_tree_root = KDTreeNode(ids, 0, grads, self.lora_depth)

    def state_dict(self):
        return {
            "all_grad": self.all_grad,
            "current_grad": self.current_grad,
            "sim": self.sim,
            "num_of_selected": self.num_of_selected,
            "total_rounds": self.total_rounds,
            "tmp_rounds": self.tmp_rounds,
            "tmp_reg": self.tmp_reg,
            "lora_depth": self.lora_depth,
        }

    def load_state_dict(self, state_dict, device=None):
        for name in ("all_grad", "current_grad", "sim", "num_of_selected"):
            value = state_dict.get(name)
            if value is not None and device is not None:
                value = value.to(device)
            setattr(self, name, value)
        self.total_rounds = state_dict.get("total_rounds", 1)
        self.tmp_rounds = state_dict.get("tmp_rounds", 0)
        self.tmp_reg = state_dict.get("tmp_reg", 0.0)
        self.lora_depth = state_dict.get("lora_depth")
        self._rebuild_tree()
