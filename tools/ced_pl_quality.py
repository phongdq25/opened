"""Precision/recall of a run's pseudo-labels against the full annotation. CPU only.

For each task t >= 1 of a PL run, the records pseudo-labeling added to D_t
(data/stage_<run>/<t>_pl/train.jsonl minus the task data) are matched on (trigger, type)
against the records the task split stripped (the --oracle build minus the task data).
Memory rows are skipped. Recall counts every stripped record, including those in no-event
rows, which ced_pseudo_label.py never looks at. Rates are pooled over the runs of one config.

Needs the oracle data (project_commands.sh builds it for group o):
    python tools/build_ced_perms.py --cap 10 --oracle --out-prefix ace_oracle_b10_perm
Usage (repo root):
    python tools/ced_pl_quality.py results/qwen3/ced/ours_h12_g2_nofilter_perm*_ace_v2_s42 \
        results/qwen3/ced/ours_h12_g2_ground_perm*_ace_v2_s42 results/qwen3/ced/ours_h12_g1_pl_perm*_ace_v2_s42
"""
import argparse
import collections
import json
import os
import re


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def keys(row):
    return {(str(e[0]).lower(), e[1]) for e in json.loads(row["response"]).get("events", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run dirs under results/qwen3/ced")
    ap.add_argument("--oracle-prefix", default="ace_oracle_b10_perm")
    args = ap.parse_args()

    stats = collections.defaultdict(collections.Counter)  # (config, task) -> counts
    for run_dir in args.runs:
        manifest = json.load(open(os.path.join(run_dir, "run_manifest.json")))
        run, perm, data_root = manifest["run"], manifest["permutation"], manifest["data_root"]
        streams = json.load(open(os.path.join(data_root, "streams.json")))
        config = re.sub(r"_perm\d+", "", run)
        for t in range(1, len(streams)):
            pl_file = os.path.join("data", f"stage_{run}", f"{t}_pl", "train.jsonl")
            if not os.path.exists(pl_file):
                continue
            old = {x for s in streams[:t] for x in s}
            task = jl(os.path.join(data_root, str(t), "train.jsonl"))
            pl = jl(pl_file)
            full = jl(os.path.join("data", f"{args.oracle_prefix}{perm}", str(t), "train.jsonl"))
            assert len(pl) == len(task), f"{pl_file}: {len(pl)} rows vs {len(task)} in the task data"
            c = stats[(config, t)]
            for tr, pr, fr in zip(task, pl, full):
                assert tr["user_prompt"] == pr["user_prompt"] == fr["user_prompt"], f"rows out of order in {pl_file}"
                gold = keys(tr)
                if gold and {ty for _, ty in gold} <= old:
                    continue  # memory row
                added, stripped = keys(pr) - gold, keys(fr) - gold
                c["added"] += len(added)
                c["stripped"] += len(stripped)
                c["hit"] += len(added & stripped)
            c["runs"] += 1

    print(f"{'config':45s} {'task':>4s} {'runs':>4s} {'PL added':>8s} {'stripped':>8s} {'precision':>9s} {'recall':>7s}")
    pooled = collections.defaultdict(collections.Counter)
    for (config, t), c in sorted(stats.items()):
        pooled[config] += c
        print(f"{config:45s} {t:4d} {c['runs']:4d} {c['added']:8d} {c['stripped']:8d} "
              f"{100 * c['hit'] / max(c['added'], 1):8.1f}% {100 * c['hit'] / max(c['stripped'], 1):6.1f}%")
    for config, c in pooled.items():
        print(f"{config:45s} {'all':>4s} {'':4s} {c['added']:8d} {c['stripped']:8d} "
              f"{100 * c['hit'] / max(c['added'], 1):8.1f}% {100 * c['hit'] / max(c['stripped'], 1):6.1f}%")


if __name__ == "__main__":
    main()
