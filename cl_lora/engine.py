"""CL-LoRA baseline engine (STANDALONE) — runs a whole ACE-CED task sequence in one
process so per-task LoRA adapters persist (Family A), reusing our raw JSONL data and
ed_evaluate for a fair comparison. Deliberately depends only on torch + peft +
transformers + ed_eval + cl_lora (NO deepspeed / utils / arguments), so it runs under
any modern env (the OpenED mta env is gone from A100).

Dispatch on --cl-method:
  inclora      per-task adapter, summed at inference (no constraint)
  olora        + orthogonality loss between adapters
  inflora      designed (frozen) lora_A per task, train lora_B; DualGPM keeps it ⟂ old
  migu         single full-FT model, magnitude-masked gradient (no LoRA)
  tree         TreeLoRA bandit-tree regulariser toward relevant previous adapters
  gainlora_o   input-gated sum of branches (O-LoRA base)
  gainlora_inf input-gated sum of branches (InfLoRA base)
  epi          parameter isolation: per-task adapter, Mahalanobis feature routing at eval

Data: --data-root points at data/<prefix><perm> with per-task subdirs 0..N-1/{train,test}.jsonl.
Each row = {system_prompt, user_prompt, response}. Train on prompt+response (labels
masked on the prompt); eval by greedy-generating from the prompt and scoring trigger-F1.
"""
import argparse
import hashlib
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from ed_eval import ed_evaluate
from cl_lora.multi_adapter import CLLoRAManager, lora_layers
from cl_lora import migu as migu_mod
from cl_lora import treelora as tree_mod
from cl_lora import gainlora as gain_mod
from cl_lora.inflora import ActivationCollector, DualGPM, design_B
from cl_lora.epi import (
    MahalanobisRouter,
    router_evaluation_indices,
    router_training_indices,
)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

ORTH = ("olora", "gainlora_o")
DESIGNED_B = ("inflora", "gainlora_inf")
GATED = ("gainlora_o", "gainlora_inf")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="Qwen/Qwen3-0.6B")
    p.add_argument("--cl-method", required=True,
                   choices=["inclora", "olora", "migu", "tree", "inflora",
                            "gainlora_o", "gainlora_inf", "epi"])
    p.add_argument("--data-root", required=True, help="data/<prefix><perm> with 0..N-1/{train,test}.jsonl")
    p.add_argument("--num-tasks", type=int, default=5)
    p.add_argument("--end-task", type=int, default=None,
                   help="stop after this task boundary; use --resume to continue")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--max-prompt-length", type=int, default=460)
    p.add_argument("--cl-reg", type=float, default=0.5)
    p.add_argument("--cl-migu-ratio", type=float, default=0.7)
    p.add_argument("--calib-samples", type=int, default=64, help="InfLoRA calibration examples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=-1, help="truncate each split to N rows (debug/smoke)")
    p.add_argument("--save", required=True)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


class JsonlED(Dataset):
    """Read {system_prompt,user_prompt,response} jsonl; build prompt+response ids for CLM."""

    def __init__(self, path, tokenizer, max_length, max_prompt_length, split, limit=-1):
        self.tok = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.split = split
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        if limit > 0:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def _prompt_text(self, r):
        msgs = [{"role": "system", "content": r["system_prompt"]},
                {"role": "user", "content": r["user_prompt"]}]
        # tokenize=True returns a BatchEncoding on tf5.x; render to string then encode
        # so we get a plain list[int] (the template already embeds special tokens).
        text = self.tok.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        return text

    def __getitem__(self, i):
        r = self.rows[i]
        prompt_text = self._prompt_text(r)
        full_prompt = self.tok.encode(prompt_text, add_special_tokens=False)
        full_ids = self.tok.encode(
            prompt_text + r["response"], add_special_tokens=False
        ) + [self.tok.eos_token_id]
        p = full_prompt[: self.max_prompt_length]
        resp = full_ids[len(full_prompt):]
        return {"prompt": p, "response": resp, "answer": r["response"]}

    def collate_train(self, batch):
        seqs, labels = [], []
        for b in batch:
            ids = (b["prompt"] + b["response"])[: self.max_length]
            lab = ([-100] * len(b["prompt"]) + b["response"])[: self.max_length]
            seqs.append(ids); labels.append(lab)
        m = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id
        input_ids = torch.tensor([s + [pad] * (m - len(s)) for s in seqs])
        label = torch.tensor([l + [-100] * (m - len(l)) for l in labels])
        attn = (input_ids != pad).long()
        return {"input_ids": input_ids, "attention_mask": attn}, {"label": label}

    def collate_gen(self, batch):
        m = self.max_prompt_length
        pad = self.tok.pad_token_id
        # left-pad prompts for generation
        ids = [[pad] * (m - len(b["prompt"])) + b["prompt"] for b in batch]
        input_ids = torch.tensor(ids)
        attn = (input_ids != pad).long()
        answers = [b["answer"] for b in batch]
        return {"input_ids": input_ids, "attention_mask": attn}, answers


def lora_config(a):
    return LoraConfig(task_type="CAUSAL_LM", r=a.rank, lora_alpha=a.alpha,
                      lora_dropout=a.dropout, target_modules=TARGET_MODULES)


def mean_pool(hidden, attn):
    m = attn.unsqueeze(-1).float()
    return (hidden.float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def optimizer_steps_per_epoch(num_rows, batch_size, grad_accum):
    return num_rows // (batch_size * grad_accum)


def atomic_json_dump(data, path):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=1)
    os.replace(temporary, path)


def files_fingerprint(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.encode("utf-8"))
        with open(path, "rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(data_root, num_tasks):
    paths = [os.path.join(data_root, "streams.json")]
    for task in range(num_tasks):
        paths.extend(
            os.path.join(data_root, str(task), f"{split}.jsonl")
            for split in ("train", "dev", "test")
        )
    return files_fingerprint(paths)


def runtime_fingerprint():
    module_root = os.path.dirname(__file__)
    project_root = os.path.dirname(module_root)
    return files_fingerprint([
        __file__,
        os.path.join(project_root, "ed_eval.py"),
        os.path.join(module_root, "epi.py"),
        os.path.join(module_root, "gainlora.py"),
        os.path.join(module_root, "inflora.py"),
        os.path.join(module_root, "migu.py"),
        os.path.join(module_root, "multi_adapter.py"),
        os.path.join(module_root, "treelora.py"),
    ])


def cpu_state_dict(state_dict):
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in state_dict.items()
    }


def save_task_checkpoint(path, completed_task, model, mgr, tree, dualgpm, router, gates, results):
    checkpoint = {
        "completed_task": completed_task,
        "results": results,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "tree": None if tree is None else cpu_state_dict(tree.state_dict()),
        "dualgpm": None if dualgpm is None else dualgpm.state_dict(),
        "router": None if router is None else router.state_dict(),
        "gates": None if gates is None else {
            "networks": cpu_state_dict(gates.gatenets.state_dict()),
            "gpm": gates.gpm.state_dict(),
        },
    }
    if mgr is None:
        checkpoint["model"] = cpu_state_dict(model.state_dict())
    else:
        checkpoint["adapters"] = {
            adapter: cpu_state_dict(
                get_peft_model_state_dict(model, adapter_name=adapter)
            )
            for adapter in mgr.task_adapters
        }
    temporary = path + ".tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def restore_rng(checkpoint):
    random.setstate(checkpoint["python_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    torch.set_rng_state(checkpoint["torch_rng"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])


# ---------------- InfLoRA: design frozen lora_A, then grow the old-task subspace ----------------
@torch.no_grad()
def _collect_input_cov(a, model, ds, device):
    """Run a small calibration pass; return an ActivationCollector keyed by LoRA-layer name."""
    layers = {name: mod.base_layer for name, mod in lora_layers(model) if hasattr(mod, "base_layer")}
    coll = ActivationCollector(layers)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, collate_fn=ds.collate_train)
    model.eval()
    seen = 0
    for mb, _ in loader:
        mb = {k: v.to(device) for k, v in mb.items()}
        model(**mb, use_cache=False)
        seen += mb["input_ids"].shape[0]
        if seen >= a.calib_samples:
            break
    coll.remove()
    return coll


@torch.no_grad()
def setup_inflora(a, model, mgr, ds, device, current, dualgpm):
    """Design lora_A[current] = top-r input dirs of this task ⟂ old-task subspace; freeze it."""
    coll = _collect_input_cov(a, model, ds, device)
    for name, mod in lora_layers(model):
        if current not in getattr(mod, "lora_A", {}):
            continue
        basis = coll.basis(name)                       # (in, k) current-task input dirs
        if basis is None:
            continue
        Bt = design_B(name, basis.to(device), dualgpm, a.rank)      # (r_eff, in)
        w = mod.lora_A[current].weight                 # (r, in)
        rr = min(Bt.shape[0], w.shape[0])
        w.data.zero_()
        w.data[:rr] = Bt[:rr].to(w.dtype)
    mgr.set_trainable(current, train_A=False, train_B=True)         # A designed+frozen, train B


@torch.no_grad()
def grow_inflora(a, model, ds, device, dualgpm):
    """After a task, add its input principal directions to the DualGPM old-task subspace."""
    coll = _collect_input_cov(a, model, ds, device)
    for name, _ in lora_layers(model):
        basis = coll.basis(name)
        if basis is not None:
            dualgpm.grow(name, basis.to(device))


# ---------------- EPI: frozen-backbone features for the Mahalanobis router ----------------
@torch.no_grad()
def frozen_features(model, tok, ds, device, n_max=512, indices=None):
    feature_data = ds if indices is None else Subset(ds, indices)
    loader = DataLoader(feature_data, batch_size=16, shuffle=False, collate_fn=ds.collate_gen)
    feats = []
    model.eval()
    with model.disable_adapter():                      # base backbone, no adapters
        for mb, _ in loader:
            mb = {k: v.to(device) for k, v in mb.items()}
            out = model(**mb, output_hidden_states=True, use_cache=False)
            feats.append(mean_pool(out.hidden_states[-1], mb["attention_mask"]).cpu())
            if sum(f.shape[0] for f in feats) >= n_max:
                break
    return torch.cat(feats)[:n_max]


@torch.no_grad()
def epi_router_diagnostics(a, model, tok, device, router, streams, upto):
    confusion = [[0 for _ in range(upto + 1)] for _ in range(upto + 1)]
    total = correct = 0
    for task_id in range(upto + 1):
        dataset = JsonlED(
            os.path.join(a.data_root, str(task_id), "dev.jsonl"),
            tok,
            a.max_length,
            a.max_prompt_length,
            "dev",
            limit=a.limit,
        )
        indices = router_evaluation_indices(dataset.rows, streams[task_id])
        if not indices:
            continue
        features = frozen_features(
            model, tok, dataset, device, n_max=128, indices=indices
        )
        predictions = router.predict(features).cpu().tolist()
        for prediction in predictions:
            confusion[task_id][prediction] += 1
            correct += int(prediction == task_id)
            total += 1
    return {
        "accuracy": correct / total if total else None,
        "total": total,
        "confusion": confusion,
    }


def train_task(a, model, ds, device, task_id, mgr, migu, tree, gates):
    sampler = DistributedSampler(
        ds,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=0,
        drop_last=True,
    )
    loader = DataLoader(
        ds, batch_size=a.batch_size, sampler=sampler, collate_fn=ds.collate_train
    )
    params = [p for p in model.parameters() if p.requires_grad]
    if gates is not None:
        params += gates.parameters()                   # gate MLPs live outside `model`
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-2)
    updates_per_epoch = optimizer_steps_per_epoch(len(ds), a.batch_size, a.grad_accum)
    if updates_per_epoch < 1:
        raise ValueError(
            f"task{task_id} has {len(ds)} rows, fewer than effective batch "
            f"{a.batch_size * a.grad_accum}"
        )
    micro_steps_per_epoch = updates_per_epoch * a.grad_accum
    total_updates = updates_per_epoch * a.epochs
    scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=a.warmup_ratio * total_updates,
        num_training_steps=total_updates,
    )
    if tree is not None:
        tree.new_epoch_init(micro_steps_per_epoch * a.epochs)
    cur = mgr.task_adapters[-1] if mgr is not None else None
    model.train()
    opt.zero_grad(set_to_none=True)
    for epoch in range(a.epochs):
        sampler.set_epoch(epoch)
        for step, (mb, nmb) in enumerate(loader):
            if step >= micro_steps_per_epoch:
                break
            mb = {k: v.to(device) for k, v in mb.items()}
            labels = nmb["label"].to(device)
            if gates is not None:
                gates.set_batch_gates(mb["input_ids"], mb["attention_mask"])
            logits = model(**mb, use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(), labels[:, 1:].reshape(-1))
            if mgr is not None and a.cl_method in ORTH:
                loss = loss + mgr.orth_loss(cur)
            if tree is not None:
                tree.step()
                sig = tree_mod.signature_from_model(model, cur)
                tree.insert_grad(sig)
                if task_id > 0:
                    prev = tree.tree_search(task_id, device)
                    loss = loss - tree.get_loss(sig, loss, task_id, prev)
            (loss / a.grad_accum).backward()
            if migu is not None:
                migu.mask_grads()
            if gates is not None:
                gates.project_grads()
            if (step + 1) % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, a.clip_grad)
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                if migu is not None:
                    migu.clear()
    if tree is not None:
        tree.end_task(task_id)
    return updates_per_epoch, total_updates


@torch.no_grad()
def _generate(a, model, tok, mb):
    gen = model.generate(
        **mb,
        max_new_tokens=a.max_length - a.max_prompt_length,
        do_sample=False,
        eos_token_id=[tok.eos_token_id, 151643],
        pad_token_id=tok.pad_token_id,
    )
    return tok.batch_decode(gen[:, mb["input_ids"].size(1):], skip_special_tokens=True)


@torch.no_grad()
def eval_task(a, model, tok, device, upto, mgr, router=None, gates=None):
    ds = JsonlED(os.path.join(a.data_root, str(upto), "test.jsonl"),
                 tok, a.max_length, a.max_prompt_length, "test", limit=a.limit)
    loader = DataLoader(ds, batch_size=a.eval_batch_size, shuffle=False, collate_fn=ds.collate_gen)
    if router is None and mgr is not None:
        mgr.consolidate()                              # activate all branches (summed forward)
    preds, refs = [], []
    model.eval()
    for mb, answers in loader:
        mb = {k: v.to(device) for k, v in mb.items()}
        if router is not None:                         # EPI: route each example to its task adapter
            with model.disable_adapter():
                fout = model(**mb, output_hidden_states=True, use_cache=False)
            ids = router.predict(mean_pool(fout.hidden_states[-1], mb["attention_mask"]))
            out = [None] * len(answers)
            for tid in ids.unique().tolist():
                idx = (ids == tid).nonzero().flatten().tolist()
                mgr.activate([f"task{tid}"])
                sub = {k: v[idx] for k, v in mb.items()}
                dec = _generate(a, model, tok, sub)
                for j, i in enumerate(idx):
                    out[i] = dec[j]
            preds.extend(out)
        else:
            if gates is not None:
                gates.set_batch_gates(mb["input_ids"], mb["attention_mask"])
            preds.extend(_generate(a, model, tok, mb))
        refs.extend([[x] for x in answers])
    metrics = ed_evaluate(preds, refs)
    return metrics, preds, refs


def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.save, exist_ok=True)
    complete_marker = os.path.join(a.save, ".complete")
    if os.path.exists(complete_marker):
        raise FileExistsError(f"run already complete: {a.save}")
    manifest_path = os.path.join(a.save, "run_manifest.json")
    checkpoint_path = os.path.join(a.save, "checkpoint_latest.pt")
    requested_manifest = {
        "method": a.cl_method,
        "data_root": os.path.abspath(a.data_root),
        "seed": a.seed,
        "model": a.model_path,
        "data_sha256": dataset_fingerprint(a.data_root, a.num_tasks),
        "runtime_sha256": runtime_fingerprint(),
        "rank": None if a.cl_method == "migu" else a.rank,
        "alpha": None if a.cl_method == "migu" else a.alpha,
        "dropout": 0.0 if a.cl_method in GATED else a.dropout,
        "micro_batch": a.batch_size,
        "gradient_accumulation": a.grad_accum,
        "effective_batch": a.batch_size * a.grad_accum,
        "epochs": a.epochs,
        "num_tasks": a.num_tasks,
        "row_limit": a.limit,
        "scheduler": "warmup_cosine",
        "warmup_ratio": a.warmup_ratio,
        "prompt_mode": "qwen_chat_template_thinking_disabled",
        "decoding": "greedy",
        "status": "running",
        "completed_task": -1,
    }
    if os.path.exists(manifest_path):
        if not a.resume:
            raise FileExistsError(f"partial run exists; pass --resume or use a new path: {a.save}")
        with open(manifest_path, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        for key in (
            "method", "data_root", "seed", "model", "rank", "alpha", "dropout",
            "data_sha256", "runtime_sha256", "micro_batch", "gradient_accumulation",
            "epochs", "num_tasks", "row_limit", "scheduler", "prompt_mode"
        ):
            if manifest.get(key) != requested_manifest.get(key):
                raise ValueError(
                    f"resume manifest mismatch for {key}: "
                    f"{manifest.get(key)!r} != {requested_manifest.get(key)!r}"
                )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"missing resume checkpoint: {checkpoint_path}")
    elif a.resume:
        raise FileNotFoundError(f"no partial run to resume: {a.save}")
    else:
        manifest = requested_manifest
        atomic_json_dump(manifest, manifest_path)
    tok = AutoTokenizer.from_pretrained(a.model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(a.model_path, dtype=torch.bfloat16)
    method = a.cl_method
    isolate = method == "epi"
    if method in GATED:
        a.dropout = 0.0            # gated hook recomputes deltas; must match PEFT's (no dropout)

    if method == "migu":
        model = base.to(device); mgr = None
    else:
        model = get_peft_model(base, lora_config(a), adapter_name="task0").to(device)
        mgr = CLLoRAManager(model, orth_lambda=(a.cl_reg if method in ORTH else 0.0))
        mgr.register_task0("task0")
        if isolate:
            mgr.activate(["task0"])

    migu = migu_mod.MIGU(model, ratio=a.cl_migu_ratio) if method == "migu" else None
    tree = tree_mod.KDLoRATree(a.num_tasks, reg=a.cl_reg) if method == "tree" else None
    dualgpm = DualGPM() if method in DESIGNED_B else None
    router = MahalanobisRouter() if isolate else None
    gates = gain_mod.GainGates(model, device) if method in GATED else None
    streams = None
    if isolate:
        with open(os.path.join(a.data_root, "streams.json"), encoding="utf-8") as stream_file:
            streams = json.load(stream_file)

    results = {}
    start_task = 0
    end_task = a.num_tasks - 1 if a.end_task is None else a.end_task
    if end_task < 0 or end_task >= a.num_tasks:
        raise ValueError(f"--end-task must be in [0, {a.num_tasks - 1}], got {end_task}")
    if a.resume:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        completed_task = checkpoint["completed_task"]
        if mgr is not None:
            for task_id in range(1, completed_task + 1):
                mgr.start_new_task(
                    task_id,
                    lora_config(a),
                    isolate=isolate,
                    train_A=method not in DESIGNED_B,
                    train_B=True,
                )
            for adapter, adapter_state in checkpoint["adapters"].items():
                set_peft_model_state_dict(model, adapter_state, adapter_name=adapter)
        else:
            model.load_state_dict(checkpoint["model"])
        if tree is not None:
            tree.load_state_dict(checkpoint["tree"], device=device)
        if dualgpm is not None:
            dualgpm.load_state_dict(checkpoint["dualgpm"], device=device)
        if router is not None:
            router.load_state_dict(checkpoint["router"], device="cpu")
        if gates is not None:
            for task_id in range(completed_task + 1):
                gates.add_branch(f"task{task_id}")
            expected_gates = len({
                key.split(".", 1)[0]
                for key in checkpoint["gates"]["networks"]
            })
            if expected_gates != completed_task + 1:
                raise ValueError(
                    f"gate checkpoint count mismatch: expected {completed_task + 1}, "
                    f"found {expected_gates}"
                )
            gates.gatenets.load_state_dict(checkpoint["gates"]["networks"])
            gates.gpm.load_state_dict(checkpoint["gates"]["gpm"], device=device)
            gates.current = None
            gates.clear_gates()
            for parameter in gates.gatenets.parameters():
                parameter.requires_grad_(False)
        results = checkpoint["results"]
        restore_rng(checkpoint)
        start_task = completed_task + 1
        print(f"resuming after task{completed_task} from {checkpoint_path}", flush=True)

    for t in range(start_task, end_task + 1):
        current = f"task{t}"
        if mgr is not None and t > 0:
            mgr.start_new_task(t, lora_config(a), isolate=isolate,
                               train_A=method not in DESIGNED_B, train_B=True)
        train_ds = JsonlED(os.path.join(a.data_root, str(t), "train.jsonl"),
                           tok, a.max_length, a.max_prompt_length, "train", limit=a.limit)
        if gates is not None:
            gates.clear_gates()                # calibration/setup below must run ungated
        if method in DESIGNED_B:
            setup_inflora(a, model, mgr, train_ds, device, current, dualgpm)
        if gates is not None:
            gates.add_branch(current)
        updates_per_epoch, total_updates = train_task(
            a, model, train_ds, device, t, mgr, migu, tree, gates
        )
        if method in DESIGNED_B:
            grow_inflora(a, model, train_ds, device, dualgpm)
        if gates is not None:
            gates.end_task(current)
        router_diagnostics = None
        if isolate:
            router_indices = router_training_indices(train_ds.rows, streams, t)
            excluded = len(train_ds) - len(router_indices)
            print(
                f"[cl:epi] task{t} router fit rows={len(router_indices)} "
                f"excluded_replay={excluded}",
                flush=True,
            )
            router.add_task(
                frozen_features(model, tok, train_ds, device, indices=router_indices)
            )
            router_diagnostics = epi_router_diagnostics(
                a, model, tok, device, router, streams, t
            )
            print(
                f"[cl:epi] task{t} router accuracy="
                f"{router_diagnostics['accuracy']} confusion="
                f"{router_diagnostics['confusion']}",
                flush=True,
            )
        metrics, predictions, references = eval_task(
            a, model, tok, device, t, mgr, router=router, gates=gates
        )
        f1 = metrics["trigger"]["f1"]
        if router_diagnostics is not None:
            metrics["router"] = router_diagnostics
        results[f"task{t}"] = metrics
        print(f"[cl:{method}] task{t} cumulative trigger-F1 = {f1:.4f}", flush=True)
        prediction_dir = os.path.join(a.save, "predictions")
        os.makedirs(prediction_dir, exist_ok=True)
        with open(
            os.path.join(prediction_dir, f"task{t}.jsonl"), "w", encoding="utf-8"
        ) as prediction_file:
            for prediction, reference in zip(predictions, references):
                prediction_file.write(json.dumps({
                    "prediction": prediction,
                    "reference": reference[0],
                }) + "\n")
        atomic_json_dump(results, os.path.join(a.save, "cl_results.json"))
        manifest["completed_task"] = t
        manifest.setdefault("task_updates", {})[f"task{t}"] = {
            "per_epoch": updates_per_epoch,
            "total": total_updates,
        }
        save_task_checkpoint(
            checkpoint_path,
            t,
            model,
            mgr,
            tree,
            dualgpm,
            router,
            gates,
            results,
        )
        atomic_json_dump(manifest, manifest_path)
    manifest["status"] = "complete" if end_task == a.num_tasks - 1 else "partial"
    atomic_json_dump(manifest, manifest_path)
    if end_task == a.num_tasks - 1:
        open(complete_marker, "w", encoding="utf-8").write("done\n")
    final_f1 = {
        task: metrics["trigger"]["f1"]
        for task, metrics in results.items()
    }
    marker = "CL-LORA DONE" if end_task == a.num_tasks - 1 else "CL-LORA PARTIAL"
    print(f"{marker} {method} {final_f1}", flush=True)


if __name__ == "__main__":
    main()
