"""Concatenate the CRE groups into a single non-continual dataset.

This is the joint-training upper bound: all relations at once, no task stream,
no replay buffer. Mirrors data/ace/{train,dev,test}.jsonl.

The groups are a disjoint partition of the whole dataset, so the full split is
just their concatenation -- same per-relation train/dev/test boundaries as every
task in every perm, which is what makes the numbers comparable.

Usage (from the OpenED dir):
    python tools/cre_build_full.py --dataset fewrel
    python tools/cre_build_full.py --dataset tacred
"""
import argparse
import json
import os
import random

SHUFFLE_SEED = 2021


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["fewrel", "tacred"], required=True)
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()

    src = f"{args.data_root}/{args.dataset}_groups"
    groups = json.load(open(f"{src}/groups.json"))
    out_dir = f"{args.data_root}/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== {args.dataset} full: {sum(len(g) for g in groups)} relations "
          f"from {len(groups)} groups -> {out_dir}")
    for split in ("train", "dev", "test"):
        rows = []
        for g in range(len(groups)):
            with open(f"{src}/{g}/{split}.jsonl", encoding="utf-8") as f:
                rows += f.read().splitlines()
        # train is shuffled for joint training; dev/test are shuffled because
        # --dev-num/--test-num cap by taking the first N rows, and these files are
        # relation-major, so an unshuffled prefix would only cover a few relations.
        random.Random(SHUFFLE_SEED).shuffle(rows)
        with open(f"{out_dir}/{split}.jsonl", "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        rels = {json.loads(json.loads(r)["response"])["events"][0][1] for r in rows}
        print(f"  {split}: {len(rows)} rows, {len(rels)} relations")


if __name__ == "__main__":
    main()
