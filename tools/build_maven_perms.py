"""Build CED task splits for the 5 stream permutations of a generated-format corpus
(MAVEN, RAMS, GENEVA).

Counterpart of build_ced_perms.py (ACE). These corpora ship as a single generated-format
split (data/<ds>/{train,dev,test}.jsonl, records are {system_prompt, user_prompt, response})
instead of the raw ACE schema, so tasks are carved by filtering each record's event list.

Streams follow KT (Yu et al., 2021):
  * MAVEN uses KT's published partition: pass --streams-file tools/streams/maven_kt.json
    (KT's data/MAVEN/streams.json with label ids mapped to our type names).
  * RAMS and GENEVA have no published split, so KT's procedure (prepare_streams.py in the KT
    repo) is applied: shuffle the types, then put each one in the stream with the fewest
    training instances so far. Same seed as KT.

Output matches what the CED runners expect:
    data/<out-prefix><p>/streams.json
    data/<out-prefix><p>/<t>/{train,dev,test}.jsonl

Usage:
    python tools/build_maven_perms.py --src data/maven --streams-file tools/streams/maven_kt.json
    python tools/build_maven_perms.py --src data/rams --out-prefix rams_b10_perm
    python tools/build_maven_perms.py --src data/geneva --out-prefix geneva_b10_perm --dry-run
"""
import argparse
import json
import os
import random
from collections import Counter

import numpy as np

# KT's five task orders (run_train.py in the KT repo), also used by EMP and SharpSeq
PERM = [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [0, 3, 1, 4, 2], [1, 2, 0, 3, 4], [3, 4, 0, 1, 2]]
KT_STREAM_SEED = 2227341903  # prepare_streams.py in the KT repo


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def events_of(row):
    resp = row["response"]
    if isinstance(resp, str):
        resp = json.loads(resp)
    return resp.get("events", [])


def with_events(row, events):
    out = dict(row)
    out["response"] = json.dumps({"events": events}, ensure_ascii=False)
    return out


def keep(row, types):
    """Record restricted to `types`; empty list means the sentence is a negative here."""
    return [e for e in events_of(row) if len(e) >= 2 and e[1] in types]


def type_freq(train_rows):
    """Training mentions per event type, the instance count KT balances on."""
    freq = Counter()
    for row in train_rows:
        for e in events_of(row):
            if len(e) >= 2:
                freq[e[1]] += 1
    return freq


def build_streams(freq, n_streams, seed):
    """KT's prepare_streams.py: shuffle the types, then put each in the stream with the
    fewest training instances so far (first such stream on ties)."""
    np.random.seed(seed)
    labels = [t for t, _ in freq.most_common()]
    shuffled = [labels[i] for i in np.random.permutation(len(labels))]
    buckets, loads = [[] for _ in range(n_streams)], [0] * n_streams
    for t in shuffled:
        i = loads.index(min(loads))
        buckets[i].append(t)
        loads[i] += freq[t]
    return [sorted(b) for b in buckets]


def build_task(train_rows, dev_rows, test_rows, task_types, seen_types, buffer, cap, rnd):
    """One task of one permutation.

    train = sentences having an event of this task's types (events filtered to those types)
            + a 10% sample of sentences with no event of these types (negatives)
            + the replay buffer accumulated from earlier tasks of this permutation
    dev/test = sentences having an event among all types seen so far (cumulative eval)
    Returns (train, dev, test, new_exemplars).
    """
    pos, neg = [], []
    per_type = {}
    for row in train_rows:
        evs = keep(row, task_types)
        if evs:
            rec = with_events(row, evs)
            pos.append(rec)
            for e in evs:                       # first `cap` per type become exemplars
                bucket = per_type.setdefault(e[1], [])
                if len(bucket) < cap:
                    bucket.append(rec)
        else:
            neg.append(with_events(row, []))

    train = list(pos)
    train.extend(rnd.sample(neg, min(len(neg), len(pos) // 10)))
    train.extend(buffer)

    def cumulative(rows):
        out = []
        for row in rows:
            evs = keep(row, seen_types)
            if evs:
                out.append(with_events(row, evs))
        return out

    exemplars = [r for v in per_type.values() for r in v]
    return train, cumulative(dev_rows), cumulative(test_rows), exemplars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/maven")
    ap.add_argument("--out-prefix", default="maven_b10_perm")
    ap.add_argument("--cap", type=int, default=10, help="exemplars kept per event type")
    ap.add_argument("--n-streams", type=int, default=5)
    ap.add_argument("--perms", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=42, help="negative sampling")
    ap.add_argument("--stream-seed", type=int, default=KT_STREAM_SEED)
    ap.add_argument("--streams-file", help="use a published partition instead of building one")
    ap.add_argument("--dry-run", action="store_true", help="print the split, write nothing")
    args = ap.parse_args()

    train_rows = load_jsonl(f"{args.src}/train.jsonl")
    dev_rows = load_jsonl(f"{args.src}/dev.jsonl")
    test_rows = load_jsonl(f"{args.src}/test.jsonl")
    print(f"loaded train={len(train_rows)} dev={len(dev_rows)} test={len(test_rows)}")

    freq = type_freq(train_rows)
    if args.streams_file:
        streams = json.load(open(args.streams_file))
        missing = set(freq) - {t for s in streams for t in s}
        assert not missing, f"{len(missing)} training types are in no stream: {sorted(missing)[:5]}"
    else:
        streams = build_streams(freq, args.n_streams, args.stream_seed)
    loads = [sum(freq[t] for t in s) for s in streams]
    print(f"streams: sizes={[len(s) for s in streams]} train-events={loads}")
    for i, s in enumerate(streams):
        print(f"  stream {i} ({len(s)} types, {loads[i]} events): {s[:6]}{' ...' if len(s) > 6 else ''}")

    if args.dry_run:
        return

    for p in args.perms:
        order = PERM[p]
        rnd = random.Random(args.seed + p)
        out_root = f"data/{args.out_prefix}{p}"
        os.makedirs(out_root, exist_ok=True)
        perm_streams = [streams[i] for i in order]
        with open(f"{out_root}/streams.json", "w", encoding="utf-8") as f:
            json.dump(perm_streams, f)

        buffer, seen = [], set()
        print(f"\n=== perm {p} (stream order {order}, cap {args.cap}) ===")
        for t, task_types in enumerate(perm_streams):
            seen.update(task_types)
            train, dev, test, exemplars = build_task(
                train_rows, dev_rows, test_rows, set(task_types), set(seen),
                buffer, args.cap, rnd)
            out_dir = f"{out_root}/{t}"
            os.makedirs(out_dir, exist_ok=True)
            for name, rows in [("train", train), ("dev", dev), ("test", test)]:
                with open(f"{out_dir}/{name}.jsonl", "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            buffer.extend(exemplars)
            print(f"task {t}: streams[{order[t]}] ({len(task_types)} types) "
                  f"train={len(train)} dev={len(dev)} test={len(test)} buffer_after={len(buffer)}")


if __name__ == "__main__":
    main()
