"""F1: oversample replay exemplars in a task's train.jsonl.

Duplicates old-only rows (replay exemplars) K-1 extra times so they are
rehearsed K times per epoch. Memory budget unchanged — no new sentences are
stored, only reuse frequency increases (same principle as SharpSeq rehearsing
its exemplar set every batch).

Usage:
  python tools/ced_oversample.py --data-dir <task dir with train.jsonl> \
      --streams streams.json --task-id T --boost K --out <dir>
dev/test copied unchanged.
"""
import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--streams", required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--boost", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    streams = json.load(open(args.streams))
    old_types = set()
    for s in streams[:args.task_id]:
        old_types.update(s)

    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, "train.jsonl"))]
    out_rows = []
    n_replay = 0
    for r in rows:
        out_rows.append(r)
        events = json.loads(r["response"]).get("events", [])
        types = {e[1] for e in events}
        if types and types <= old_types:
            n_replay += 1
            out_rows.extend([r] * (args.boost - 1))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for split in ["dev.jsonl", "test.jsonl"]:
        shutil.copy(os.path.join(args.data_dir, split), os.path.join(args.out, split))
    print(f"task{args.task_id}: {n_replay} replay rows boosted x{args.boost}; "
          f"train {len(rows)} -> {len(out_rows)}")


if __name__ == "__main__":
    main()
