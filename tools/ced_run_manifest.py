#!/usr/bin/env python3
import argparse
import hashlib
import json
import os


IMMUTABLE_FIELDS = (
    "run",
    "method",
    "permutation",
    "seed",
    "data_root",
    "data_sha256",
    "runtime_sha256",
    "model",
    "rank",
    "alpha",
    "dropout",
    "gpu_count",
    "micro_batch",
    "gradient_accumulation",
    "effective_batch",
    "epochs",
    "scheduler",
    "prompt_mode",
    "decoding",
    "parent_task0_run",
    "config",
)


def files_under(path):
    if os.path.isfile(path):
        return [path]
    files = []
    for root, _, names in os.walk(path):
        for name in names:
            files.append(os.path.join(root, name))
    return files


def fingerprint(paths):
    digest = hashlib.sha256()
    files = []
    for path in paths:
        files.extend(files_under(path))
    for path in sorted(files):
        digest.update(os.path.relpath(path).replace("\\", "/").encode("utf-8"))
        with open(path, "rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def atomic_dump(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=1)
    os.replace(temporary, path)


def init_manifest(args):
    data_root = os.path.abspath(args.data_root)
    manifest = {
        "run": args.run,
        "method": args.method,
        "permutation": args.permutation,
        "seed": args.seed,
        "data_root": data_root,
        "data_sha256": fingerprint(args.data_path or [data_root]),
        "runtime_sha256": fingerprint(args.runtime_file),
        "model": args.model,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "gpu_count": args.gpu_count,
        "micro_batch": args.micro_batch,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch": args.micro_batch * args.gradient_accumulation * args.gpu_count,
        "epochs": args.epochs,
        "scheduler": "warmup_cosine",
        "prompt_mode": "qwen_chat_template_thinking_disabled",
        "decoding": "greedy" if args.greedy else "sampling",
        "parent_task0_run": args.parent_task0_run,
        "config": args.config,
        "status": "running",
        "completed_task": args.start_task - 1,
    }
    if os.path.exists(args.output):
        if not args.resume:
            raise FileExistsError(f"run manifest already exists: {args.output}")
        with open(args.output, encoding="utf-8") as manifest_file:
            existing = json.load(manifest_file)
        for key in IMMUTABLE_FIELDS:
            if existing.get(key) != manifest.get(key):
                raise ValueError(
                    f"resume manifest mismatch for {key}: "
                    f"{existing.get(key)!r} != {manifest.get(key)!r}"
                )
        return
    if args.resume:
        raise FileNotFoundError(f"no run manifest to resume: {args.output}")
    atomic_dump(manifest, args.output)


def update_manifest(args):
    with open(args.output, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    manifest["completed_task"] = args.completed_task
    manifest["status"] = args.status
    atomic_dump(manifest, args.output)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--output", required=True)
    init_parser.add_argument("--run", required=True)
    init_parser.add_argument("--method", required=True)
    init_parser.add_argument("--permutation", required=True)
    init_parser.add_argument("--seed", type=int, required=True)
    init_parser.add_argument("--data-root", required=True)
    init_parser.add_argument("--data-path", action="append")
    init_parser.add_argument("--runtime-file", action="append", required=True)
    init_parser.add_argument("--model", required=True)
    init_parser.add_argument("--rank", type=int, required=True)
    init_parser.add_argument("--alpha", type=int, required=True)
    init_parser.add_argument("--dropout", type=float, required=True)
    init_parser.add_argument("--gpu-count", type=int, required=True)
    init_parser.add_argument("--micro-batch", type=int, required=True)
    init_parser.add_argument("--gradient-accumulation", type=int, required=True)
    init_parser.add_argument("--epochs", type=int, required=True)
    init_parser.add_argument("--greedy", action="store_true")
    init_parser.add_argument("--parent-task0-run", default=None)
    init_parser.add_argument("--config", required=True)
    init_parser.add_argument("--start-task", type=int, default=0)
    init_parser.add_argument("--resume", action="store_true")

    update_parser = subparsers.add_parser("update")
    update_parser.add_argument("--output", required=True)
    update_parser.add_argument("--completed-task", type=int, required=True)
    update_parser.add_argument("--status", choices=["running", "partial", "complete"], required=True)

    args = parser.parse_args()
    if args.command == "init":
        init_manifest(args)
    else:
        update_manifest(args)


if __name__ == "__main__":
    main()