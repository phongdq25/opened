"""What vLLM runs when a CL-LoRA model is evaluated (spec §4).

- inclora, olora, inflora and tree evaluate the base plus the sum of every task adapter
  (CLLoRAManager.consolidate). One adapter with A = [A_1; ...; A_n] and B = [s_1 B_1, ..., s_n B_n]
  (rank and lora_alpha both sum(r_i), so its own scaling is 1) computes exactly that sum.
- migu fine-tunes the base weights, so the whole model is saved (engine._vllm_generate).
- gainlora_o, gainlora_inf and epi pick adapters per input (gates, router). They keep Hugging
  Face generation whatever --gen-backend says."""
import os

import torch
from peft import LoraConfig
from peft.tuners.lora import LoraLayer
from safetensors.torch import save_file

SUMMED = ("inclora", "olora", "inflora", "tree")
WHOLE = ("migu",)


def effective_backend(method, requested):
    """The generation backend eval_task uses for `method` under --gen-backend `requested`."""
    return "vllm" if requested == "vllm" and method in SUMMED + WHOLE else "hf"


def export_summed_adapters(model, adapters, base_path, out_dir):
    """Save the sum of the PEFT adapters named in `adapters` as one LoRA adapter in out_dir.
    Returns its rank."""
    state, targets, ranks = {}, set(), set()
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        names = [adapter for adapter in adapters if adapter in module.lora_A]
        if not names:
            continue
        A = torch.cat([module.lora_A[n].weight.detach().float() for n in names], dim=0)
        B = torch.cat([module.lora_B[n].weight.detach().float() * module.scaling[n] for n in names], dim=1)
        state[f"{name}.lora_A.weight"] = A.cpu().contiguous()
        state[f"{name}.lora_B.weight"] = B.cpu().contiguous()
        targets.add(name.rsplit(".", 1)[-1])
        ranks.add(A.size(0))
    if len(ranks) != 1:
        raise ValueError(f"the task adapters cover the layers unevenly (summed ranks {sorted(ranks)})")
    rank = ranks.pop()
    os.makedirs(out_dir, exist_ok=True)
    save_file(state, os.path.join(out_dir, "adapter_model.safetensors"))
    LoraConfig(task_type="CAUSAL_LM", r=rank, lora_alpha=rank, lora_dropout=0.0, target_modules=sorted(targets),
               base_model_name_or_path=base_path).save_pretrained(out_dir)
    return rank
