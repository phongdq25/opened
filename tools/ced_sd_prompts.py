"""SD: attach the self-distillation teacher prompt to every train row.

The teacher of on-policy self-distillation reads the augmented annotation
(gold + pseudo-labels, i.e. the row's current response) as a reference answer,
using the SDFT template (Shenfeld et al. 2026, main.py). Run AFTER pseudo-labeling
and oversampling so the reference is the final augmented target.

Every row gets t_system_prompt / t_user_prompt: process_data.py writes the
teacher bin only for rows that have them, so a partial set would misalign it.

Usage:
  python tools/ced_sd_prompts.py --data-dir <task dir with train.jsonl> --out <dir>
dev/test copied unchanged.
"""
import argparse
import json
import os
import shutil

TEMPLATE = ("{user_prompt}\n\nThis is an example for a response to the question:\n"
            "{response}\n\nNow answer with a response of your own.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "train.jsonl"))]
    for r in rows:
        r["t_system_prompt"] = r["system_prompt"]
        r["t_user_prompt"] = TEMPLATE.format(user_prompt=r["user_prompt"], response=r["response"])

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for split in ["dev.jsonl", "test.jsonl"]:
        shutil.copy(os.path.join(args.data_dir, split), os.path.join(args.out, split))
    print(f"sd prompts: {len(rows)} train rows")


if __name__ == "__main__":
    main()
