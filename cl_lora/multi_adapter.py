"""Multi-adapter foundation for Family-A CL-LoRA baselines.

Family A (IncLoRA, O-LoRA, InfLoRA, GainLoRA, TreeLoRA) keeps ONE PEFT LoRA
adapter per task: previous adapters frozen, current adapter trainable, and at
inference the branches are summed  W = W0 + Σ_i A_i B_i.  This module wraps a
PeftModel to manage that lifecycle and provides the O-LoRA orthogonality
regulariser.  Method-specific pieces (InfLoRA's designed B, GainLoRA's gate,
TreeLoRA's bandit reg) build on top of this.

Local now: syntax only (torch/peft not installed here; SSH down). Wire + test on server.
"""
import torch

try:                                            # peft moved LoraLayer around across versions
    from peft.tuners.lora import LoraLayer
except Exception:                               # pragma: no cover
    from peft.tuners.lora.layer import LoraLayer


def lora_layers(model):
    """Yield (name, module) for every LoRA-injected layer in a PeftModel."""
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            yield name, module


class CLLoRAManager:
    """Per-task LoRA adapter lifecycle for Family-A continual methods.

    Task adapters are named 'task0', 'task1', ...  The PeftModel is built
    outside with the first adapter ('task0') already present; call
    register_task0() once, then start_new_task() for each subsequent task.
    """

    def __init__(self, model, orth_lambda=0.0):
        self.model = model
        self.orth_lambda = float(orth_lambda)   # >0 => O-LoRA orthogonality penalty
        self.task_adapters = []                 # ordered adapter names, oldest first

    # ---------- task lifecycle ----------
    def register_task0(self, name="task0"):
        self.task_adapters = [name]
        self.set_trainable(name)

    def activate(self, names):
        """Activate one or more adapters (LoraModel sums all active ones in the forward)."""
        if isinstance(names, str):
            names = [names]
        self.model.base_model.set_adapter(list(names))

    def start_new_task(self, task_id, peft_config, isolate=False, train_A=True, train_B=True):
        """Add a fresh adapter for task_id, freeze all previous, train only the new one.

        isolate=False (default): keep ALL branches active so the forward is
        W0 + Σ_i A_iB_i (Family-A sum). isolate=True (EPI): only the new branch is
        active (parameter isolation). train_A/train_B pick which factor of the new
        branch is trainable (InfLoRA freezes lora_A, its designed down-projection).
        """
        name = f"task{task_id}"
        self.model.add_adapter(name, peft_config)
        self.task_adapters.append(name)
        # base_model (LoraModel) accepts a list => all branches active & summed;
        # PeftModel.set_adapter only takes a single name. set_adapter would mark
        # every active adapter trainable, so re-freeze via set_trainable right after.
        self.activate([name] if isolate else self.task_adapters)
        self.set_trainable(name, train_A=train_A, train_B=train_B)
        return name

    def set_trainable(self, current, train_A=True, train_B=True):
        """requires_grad True only on `current` adapter's LoRA params, False elsewhere.

        train_A/train_B gate the two factors independently (InfLoRA trains only B).
        """
        for _, module in lora_layers(self.model):
            for store, flag in ((getattr(module, "lora_A", {}), train_A),
                                (getattr(module, "lora_B", {}), train_B)):
                for adapter_name, sub in store.items():
                    req = (adapter_name == current) and flag
                    for p in sub.parameters():
                        p.requires_grad_(req)

    # ---------- O-LoRA orthogonality loss ----------
    def orth_loss(self, current):
        """λ · Σ_layers Σ_{prev} ‖A_prev · A_curᵀ‖_F²  over frozen previous adapters.

        lora_A.weight has shape (r, in_features); rows span the task's input
        subspace, so we push the cross-Gram A_prev A_curᵀ (r_prev×r_cur) to 0.
        """
        dev = next(self.model.parameters()).device
        if self.orth_lambda <= 0 or len(self.task_adapters) <= 1:
            return torch.zeros((), device=dev)
        prevs = [a for a in self.task_adapters if a != current]
        total = torch.zeros((), device=dev)
        for _, module in lora_layers(self.model):
            la = getattr(module, "lora_A", {})
            if current not in la:
                continue
            A_cur = la[current].weight                       # (r_cur, in)
            for a in prevs:
                if a not in la:
                    continue
                # peft's autocast_adapter_dtype leaves adapters in fp32 or in the base
                # dtype depending on version and on whether the adapter was created by
                # get_peft_model or added later, so the two operands are not guaranteed
                # to match (peft 0.18 mixes fp32 and bf16 here and the matmul raises).
                # Computing the regulariser in fp32 is both version-proof and the more
                # accurate choice; gradients still flow back to A_cur through the cast.
                cross = la[a].weight.float() @ A_cur.float().t()   # (r_prev, r_cur)
                total = total + (cross ** 2).sum()
        return self.orth_lambda * total

    # ---------- inference: sum all task adapters ----------
    def consolidate(self, name="cl_all"):
        """Activate every task adapter so the forward sums them: W = W0 + Σ_i A_iB_i.

        peft applies the sum of all active adapters, which is exactly the IncLoRA /
        O-LoRA / InfLoRA inference rule, so no merge is needed (this also avoids
        add_weighted_adapter, whose 'cat' path is version-fragile in peft).
        """
        self.model.base_model.set_adapter(list(self.task_adapters))
        return name
