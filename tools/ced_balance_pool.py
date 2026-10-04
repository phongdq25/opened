"""F4: build a type-balanced pool for the final calibration epoch of task t.

Pool = up to N sentences per event type, for ALL types learned so far.
Old types come from the task's replay exemplars (memory-budget compliant);
current-task types are sampled from this task's own train rows.
Counters recency bias (new-type over-prediction, fp inflation).

Usage:
  python tools/ced_balance_pool.py --data-dir <task dir> --streams s.json \
      --task-id T --per-type 10 --out <dir>
dev/test copied unchanged.
"""
import argparse
import json
import os
import random
import shutil
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--streams", required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--per-type", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "train.jsonl"))]

    by_type = defaultdict(list)
    for r in rows:
        events = json.loads(r["response"]).get("events", [])
        for ty in {e[1] for e in events}:
            by_type[ty].append(r)

    pool, seen = [], set()
    for ty, cand in by_type.items():
        random.shuffle(cand)
        for r in cand[:args.per_type]:
            key = id(r)
            if key not in seen:
                seen.add(key)
                pool.append(r)
    random.shuffle(pool)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for split in ["dev.jsonl", "test.jsonl"]:
        shutil.copy(os.path.join(args.data_dir, split), os.path.join(args.out, split))
    print(f"balance pool: {len(pool)} rows over {len(by_type)} types "
          f"(<= {args.per_type}/type)")


if __name__ == "__main__":
    main()
