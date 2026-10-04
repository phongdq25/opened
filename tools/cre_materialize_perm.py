"""Materialize one CRE stream order (perm) from the canonical groups.

Reads data/<ds>_groups (built by build_cre_groups.py) and assembles the
run-ready layout that run_ced_v2.sh expects:
    data/<ds>_perm<p>/streams.json          groups reordered by PERM[p]
    data/<ds>_perm<p>/{t}/{train,dev,test}.jsonl

Per task t (following build_ced_perms.py semantics):
    train = group PERM[p][t] + replay buffer (cap exemplars per past relation)
    dev   = concat of groups PERM[p][0..t]   (cumulative)
    test  = concat of groups PERM[p][0..t]   (cumulative)

Build one perm, run it, then delete the dir -- the groups are the only copy that
needs to stay on disk.

Usage (from the OpenED dir):
    python tools/cre_materialize_perm.py --dataset fewrel --perm 0
    python tools/cre_materialize_perm.py --dataset tacred --perm 3 --cap 10
"""
import argparse
import json
import os
import random

BASE_SEED = 2021  # perm p uses BASE_SEED + 100*p, matching WAVE's 2021..2421


def perm_order(p, n):
    """PERM[0] = identity, PERM[1] = reversed, PERM[2:] = deterministic shuffles."""
    order = list(range(n))
    if p == 1:
        order.reverse()
    elif p > 1:
        random.Random(BASE_SEED + 100 * p).shuffle(order)
    return order


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def rel_of(rec):
    return json.loads(rec["response"])["events"][0][1]


def exemplars(rows, cap):
    """First `cap` rows per relation, in order of first appearance."""
    picked, seen = [], {}
    for r in rows:
        k = rel_of(r)
        seen[k] = seen.get(k, 0)
        if seen[k] < cap:
            seen[k] += 1
            picked.append(r)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["fewrel", "tacred"], required=True)
    ap.add_argument("--perm", type=int, required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--cap", type=int, default=5, help="replay exemplars per past relation")
    args = ap.parse_args()

    src = f"{args.data_root}/{args.dataset}_groups"
    groups = json.load(open(f"{src}/groups.json"))
    order = perm_order(args.perm, len(groups))

    out_dir = f"{args.data_root}/{args.dataset}_perm{args.perm}"
    os.makedirs(out_dir, exist_ok=True)
    json.dump([groups[g] for g in order], open(f"{out_dir}/streams.json", "w"))

    print(f"=== {args.dataset} perm {args.perm}: group order {order} (cap {args.cap}) -> {out_dir}")
    cache = {}
    buffer, dev_cum, test_cum = [], [], []
    rng = random.Random(BASE_SEED + 100 * args.perm)
    for t, g in enumerate(order):
        if g not in cache:
            cache[g] = {s: load_jsonl(f"{src}/{g}/{s}.jsonl") for s in ("train", "dev", "test")}
        grp = cache[g]
        os.makedirs(f"{out_dir}/{t}", exist_ok=True)

        train = grp["train"] + list(buffer)
        rng.shuffle(train)
        dev_cum += grp["dev"]
        test_cum += grp["test"]

        # dev/test are shuffled too: --dev-num/--test-num cap by taking the first N
        # rows, and these files are relation-major, so an unshuffled prefix would
        # only ever cover the earliest relations.
        out = {"train": train}
        for name, rows in [("dev", dev_cum), ("test", test_cum)]:
            out[name] = list(rows)
            random.Random(BASE_SEED + t).shuffle(out[name])
        n = {name: write_jsonl(f"{out_dir}/{t}/{name}.jsonl", rows)
             for name, rows in out.items()}
        buffer += exemplars(grp["train"], args.cap)
        print(f"task {t}: group {g} ({len(groups[g])} rel)  train={n['train']} "
              f"dev={n['dev']} test={n['test']}  buffer_after={len(buffer)}")


if __name__ == "__main__":
    main()
