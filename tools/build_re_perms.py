"""Build continual relation-extraction (CRE) task splits for all 5 permutations.

TACRED / FewRel arrive here already carved into order-independent group shards:
    data/<ds>_groups/groups.json      relation names per group
    data/<ds>_groups/<g>/{train,dev,test}.jsonl
and perm0 was assembled from them. This script reproduces that assembly for any
permutation, so the 5 permutations come from one shared set of shards instead of five
duplicated copies of the corpus.

Assembly rule (reverse-engineered from the shipped tacred_perm0 and verified by --gate):
    train(t) = shard[order[t]].train  +  exemplars of every earlier task
               (first --cap rows per relation, scanning that shard's train in file order)
    dev(t)   = concat of shard[order[0..t]].dev        (cumulative)
    test(t)  = concat of shard[order[0..t]].test       (cumulative)
    streams.json = [relations of shard[order[t]] for each t]

perm0 = identity order, matching the shipped data. perm1..4 are seeded shuffles.
Line order inside train differs from the shipped files (the original shuffled); the
record SET is identical, which is what --gate checks, and the loader shuffles anyway.

Usage:
    python tools/build_re_perms.py --dataset tacred --gate      # verify rule on perm0
    python tools/build_re_perms.py --dataset tacred --perms 1 2 3 4
    python tools/build_re_perms.py --dataset fewrel --perms 1 2 3 4
"""
import argparse
import hashlib
import json
import os
import random


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def relation_of(row):
    """Relation label of a record, or None when the record asserts no relation."""
    resp = row["response"]
    if isinstance(resp, str):
        resp = json.loads(resp)
    events = resp.get("events", [])
    return events[0][1] if events and len(events[0]) >= 2 else None


def exemplars(rows, cap):
    """First `cap` rows per relation, scanning in file order."""
    per_rel, out = {}, []
    for r in rows:
        rel = relation_of(r)
        if rel is None:
            continue
        bucket = per_rel.setdefault(rel, 0)
        if bucket < cap:
            per_rel[rel] = bucket + 1
            out.append(r)
    return out


def record_set_hash(rows):
    keys = sorted(hashlib.md5(json.dumps(r, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                  for r in rows)
    return hashlib.md5("".join(keys).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tacred", "fewrel"])
    ap.add_argument("--cap", type=int, default=5, help="exemplars kept per relation")
    ap.add_argument("--perms", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gate", action="store_true",
                    help="rebuild perm0 in memory and compare record sets with the shipped perm0")
    args = ap.parse_args()

    src = f"data/{args.dataset}_groups"
    groups = json.load(open(f"{src}/groups.json"))
    n = len(groups)
    shards = [{split: load_jsonl(f"{src}/{g}/{split}.jsonl")
               for split in ("train", "dev", "test")} for g in range(n)]
    print(f"{args.dataset}: {n} group, "
          f"train={[len(s['train']) for s in shards]}")

    orders = {0: list(range(n))}
    for p in range(1, 5):
        o = list(range(n))
        random.Random(args.seed + p).shuffle(o)
        orders[p] = o

    def assemble(order, t):
        train = list(shards[order[t]]["train"])
        for prev in order[:t]:
            train.extend(exemplars(shards[prev]["train"], args.cap))
        dev, test = [], []
        for i in order[:t + 1]:
            dev.extend(shards[i]["dev"])
            test.extend(shards[i]["test"])
        return train, dev, test

    if args.gate:
        ok = True
        for t in range(n):
            train, dev, test = assemble(orders[0], t)
            for split, rows in (("train", train), ("dev", dev), ("test", test)):
                ref = load_jsonl(f"data/{args.dataset}_perm0/{t}/{split}.jsonl")
                same = record_set_hash(rows) == record_set_hash(ref)
                if not same or len(rows) != len(ref):
                    print(f"GATE MISMATCH task{t}/{split}: {len(rows)} vs {len(ref)}, "
                          f"record-set {'same' if same else 'DIFFERENT'}")
                    ok = False
        print("GATE PASSED: quy tac khop perm0 goc" if ok else "GATE FAILED")
        if not ok:
            return

    for p in args.perms:
        order = orders[p]
        root = f"data/{args.dataset}_perm{p}"
        os.makedirs(root, exist_ok=True)
        with open(f"{root}/streams.json", "w", encoding="utf-8") as f:
            json.dump([groups[i] for i in order], f)
        print(f"\n=== {args.dataset} perm{p} (order {order}) ===")
        for t in range(n):
            train, dev, test = assemble(order, t)
            out = f"{root}/{t}"
            os.makedirs(out, exist_ok=True)
            # shuffle train so replay rows are not all clustered at the tail
            random.Random(args.seed + p * 100 + t).shuffle(train)
            dump_jsonl(train, f"{out}/train.jsonl")
            dump_jsonl(dev, f"{out}/dev.jsonl")
            dump_jsonl(test, f"{out}/test.jsonl")
            print(f"task{t}: group{order[t]} train={len(train)} dev={len(dev)} test={len(test)}")


if __name__ == "__main__":
    main()
