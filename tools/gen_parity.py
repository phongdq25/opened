"""Hugging Face generate() against vLLM on one model and one test set (spec §8, checks 2, 3, 5).

  python tools/gen_parity.py ced --base DIR [--adapter DIR] --data-dir processed_data/<prefix><perm>/<task>/qwen/ \
      --out DIR [--strict] [--seeds 1 2 3] [--backends hf vllm]
  python tools/gen_parity.py cllora --base DIR --checkpoint <run>/checkpoint_latest.pt --method inclora \
      --data-root data/<ds>_perm<p> --task T --out DIR

Runs in the training environment on one GPU; vLLM comes from VLLM_PY. Prints one JSON line per run
(backend, seed, seconds, trigger F1, and the share of answers identical to the first backend's run
with the same seed), then a SUMMARY line with each backend's mean and spread in F1 points."""
import argparse
import ast
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def last_test_metrics(log_path):
    """The metrics of the last 'test | avg_loss: ... | {...}' line evaluate() wrote."""
    line = [line for line in open(log_path) if line.startswith("test |")][-1]
    return ast.literal_eval(line.split(" | ", 2)[2].strip())


def identical_share(first, second):
    return sum(a == b for a, b in zip(first, second)) / len(first)


def summary(rows):
    """Mean and spread (max - min) of trigger F1 per backend, and the gap between the means, in points."""
    out = {}
    for backend in sorted({row["backend"] for row in rows}):
        f1 = [100 * row["f1"] for row in rows if row["backend"] == backend]
        out[backend] = {"mean": round(statistics.mean(f1), 3), "spread": round(max(f1) - min(f1), 3)}
    if len(out) == 2:
        first, second = (value["mean"] for value in out.values())
        out["gap"] = round(abs(first - second), 3)
    return out


def report(rows, row, first, answers):
    if first is not None:
        row["identical"] = round(identical_share(first, answers), 4)
    rows.append(row)
    print(json.dumps(row), flush=True)


def eval_args(a):
    """evaluate()'s arguments, parsed from the flags run_ced_v2.sh gives the trainers. get_args() reads
    sys.argv and builds a save path from --save, so both are set here; each run then sets args.save."""
    from arguments import get_args
    argv = sys.argv
    sys.argv = ["gen_parity", "--model-path", a.base, "--model-type", "qwen", "--type", "lm", "--data-dir", a.data_dir,
                "--save", a.out, "--max-length", "768", "--max-prompt-length", "460", "--eval-batch-size", "128",
                "--eval-loss-batch-size", "32", "--top-k", "0", "--top-p", "0.95", "--temperature", "0.5",
                "--eval-gen", "--num-workers", "0"] + (["--strict-generation"] if a.strict else [])
    try:
        args = get_args()
    finally:
        sys.argv = argv
    args.dynamic_pad_effective = False
    return args


def ced(a):
    import torch
    import torch.distributed as dist
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from ced_eval import check_gen_backend, evaluate
    from data_utils.lm_datasets import LMTrainDataset
    from utils import get_tokenizer

    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{a.port}", rank=0, world_size=1)
    args = eval_args(a)
    tokenizer = get_tokenizer(args)
    test = LMTrainDataset(args, tokenizer, a.data_dir, "test", -1, 1, random.Random(0))
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).cuda()
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    rows = []
    for seed in a.seeds:
        first = None
        for backend in a.backends:
            args.seed, args.gen_backend = seed, backend
            args.save = os.path.join(a.out, f"{backend}_s{seed}")
            os.makedirs(args.save, exist_ok=True)
            check_gen_backend(args, model, tokenizer)
            torch.manual_seed(seed)
            start = time.time()
            evaluate(args, tokenizer, model, test, "test", 0, "cuda")
            seconds = round(time.time() - start, 1)
            answers = [json.loads(line)["text"] for line in open(os.path.join(args.save, "eval", "0", "answers.jsonl"))]
            f1 = last_test_metrics(os.path.join(args.save, "log.txt"))["trigger"]["f1"]
            report(rows, {"backend": backend, "seed": seed, "rows": len(answers), "seconds": seconds, "f1": f1},
                   first, answers)
            first = first if first is not None else answers
    print("SUMMARY", json.dumps(summary(rows)), flush=True)


def cllora(a):
    import torch
    from peft import get_peft_model, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cl_lora import engine
    from cl_lora.multi_adapter import CLLoRAManager
    from gen_backend import find_vllm_python

    os.makedirs(a.out, exist_ok=True)
    args = argparse.Namespace(cl_method=a.method, rank=16, alpha=64, dropout=0.1, data_root=a.data_root,
                              max_length=768, max_prompt_length=460, limit=-1, eval_batch_size=128,
                              model_path=a.base, save=a.out, gen_backend="hf", vllm_py=None)
    # the engine's own checkpoint, loaded as engine.main() loads it: it holds RNG and manager
    # state that weights_only=True refuses
    checkpoint = torch.load(a.checkpoint, map_location="cuda", weights_only=False)
    tok = AutoTokenizer.from_pretrained(a.base)
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16),
                           engine.lora_config(args), adapter_name="task0").cuda()
    mgr = CLLoRAManager(model, orth_lambda=0.0)
    mgr.register_task0("task0")
    for task_id in range(1, checkpoint["completed_task"] + 1):
        mgr.start_new_task(task_id, engine.lora_config(args))
    for adapter, state in checkpoint["adapters"].items():
        set_peft_model_state_dict(model, state, adapter_name=adapter)
    rows, first = [], None
    for backend in a.backends:
        args.gen_backend = backend
        args.vllm_py = find_vllm_python()[0] if backend == "vllm" else None
        start = time.time()
        metrics, preds, _ = engine.eval_task(args, model, tok, "cuda", a.task, mgr)
        report(rows, {"backend": backend, "rows": len(preds), "seconds": round(time.time() - start, 1),
                      "f1": metrics["trigger"]["f1"]}, first, preds)
        first = first if first is not None else preds
    print("SUMMARY", json.dumps(summary(rows)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("ced")
    c.add_argument("--base", required=True)
    c.add_argument("--adapter", default=None)
    c.add_argument("--data-dir", required=True)
    c.add_argument("--strict", action="store_true")
    c.add_argument("--seeds", type=int, nargs="+", default=[1])
    c.add_argument("--port", type=int, default=29581)
    cl = sub.add_parser("cllora")
    cl.add_argument("--base", required=True)
    cl.add_argument("--checkpoint", required=True)
    cl.add_argument("--method", required=True, choices=["inclora", "olora", "inflora", "tree"])
    cl.add_argument("--data-root", required=True)
    cl.add_argument("--task", type=int, required=True)
    for parser in (c, cl):
        parser.add_argument("--out", required=True)
        parser.add_argument("--backends", nargs="+", default=["hf", "vllm"], choices=["hf", "vllm"])
    a = ap.parse_args()
    ced(a) if a.mode == "ced" else cllora(a)


if __name__ == "__main__":
    main()
