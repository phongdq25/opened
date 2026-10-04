"""What goes wrong on OLD event types after task T, from a run's saved predictions. CPU only.

Each gold trigger of an earlier task's type is counted as correct (trigger + type predicted),
wrong type (trigger predicted under another type) or omitted (trigger not predicted at all).
Spurious = predicted old-type records whose trigger matches no gold trigger, per 100 gold.
argF1@ok = argument F1 on the old events found with the right type. unparsed = outputs that
are not valid JSON. Rates are pooled over the runs of one config (names differing only in
_perm<k>).

Usage (repo root):
    python tools/ced_error_breakdown.py --task 4 results/qwen3/ced/ours_h12_g1_ce_perm*_ace_v2_s42 \
        results/qwen3/ced/ours_h12_sd_g1_full_perm*_ace_v2_s42
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics as st


def norm(text):
    return str(text).strip().lower()


def events(text):
    """(valid JSON, {(trigger, type): {(arg text, role)}}), keyed like ed_eval.py"""
    try:
        d = json.loads(text.strip())
    except Exception:
        return False, {}
    out = collections.defaultdict(set)
    try:
        for e in d.get("events", []):
            key = (norm(e[0]), norm(e[1]))
            out[key]
            for a in e[2]:
                out[key].add((a[0], norm(a[1])))
    except Exception:
        pass
    return True, out


def breakdown(preds, refs, old):
    """Counter of old-type outcomes over aligned (prediction, reference) strings"""
    c = collections.Counter()
    for pr, gr in zip(preds, refs):
        ok, pe = events(pr)
        _, ge = events(gr)
        c["rows"] += 1
        c["unparsed"] += int(not ok)
        p_trig = {t for t, _ in pe}
        g_trig = {t for t, _ in ge}
        for (t, ty), gargs in ge.items():
            if ty not in old:
                continue
            c["gold"] += 1
            if (t, ty) in pe:
                c["correct"] += 1
                pargs = pe[(t, ty)]
                c["a_tp"] += len(gargs & pargs)
                c["a_fp"] += len(pargs - gargs)
                c["a_fn"] += len(gargs - pargs)
            elif t in p_trig:
                c["wrong_type"] += 1
            else:
                c["omitted"] += 1
        for t, ty in pe:
            if ty in old and (t, ty) not in ge and t not in g_trig:
                c["spurious"] += 1
    return c


def summarize(tot, omit_per_run):
    g = max(tot["gold"], 1)
    af1 = 2 * tot["a_tp"] / max(2 * tot["a_tp"] + tot["a_fp"] + tot["a_fn"], 1)
    return dict(n=len(omit_per_run), gold=tot["gold"], correct=100 * tot["correct"] / g,
                wrong_type=100 * tot["wrong_type"] / g, omitted=100 * tot["omitted"] / g,
                omit_sd=st.stdev(omit_per_run) if len(omit_per_run) > 1 else 0.0,
                spurious=100 * tot["spurious"] / g, arg_f1_correct=100 * af1,
                unparsed=100 * tot["unparsed"] / max(tot["rows"], 1))


def load_task(run_dir, task):
    """(predictions, references, old types) of the last epoch of one task, or None"""
    args_files = sorted(glob.glob(os.path.join(run_dir, f"task{task}", "**", "args.json"), recursive=True))
    if not args_files:
        return None
    save = os.path.dirname(args_files[-1])
    evals = sorted(glob.glob(os.path.join(save, "eval", "*", "answers.jsonl")),
                   key=lambda p: int(os.path.basename(os.path.dirname(p))))
    if not evals:
        return None
    args = json.load(open(args_files[-1]))
    refs = [json.loads(l)["response"] for l in open(os.path.join(args["data_dir"], "test.jsonl"))]
    preds = [json.loads(l)["text"] for l in open(evals[-1])][:len(refs)]
    manifest = os.path.join(run_dir, "run_manifest.json")
    streams_file = (os.path.join(json.load(open(manifest))["data_root"], "streams.json")
                    if os.path.exists(manifest) else args.get("ced_streams_file"))
    streams = json.load(open(streams_file))
    old = {norm(x) for s in streams[:task] for x in s}
    return preds, refs, old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run dirs under results/qwen3/ced")
    ap.add_argument("--task", type=int, default=4)
    args = ap.parse_args()

    groups = collections.defaultdict(list)
    for r in args.runs:
        groups[re.sub(r"_perm\d+", "", os.path.basename(os.path.normpath(r)))].append(r)

    print(f"after task {args.task}: gold triggers of types from tasks 0..{args.task - 1}")
    print(f"{'config':45s} {'n':>2s} {'gold':>5s} {'correct':>8s} {'wrongtype':>9s} {'omitted':>8s} "
          f"{'omit sd':>7s} {'spurious':>8s} {'argF1@ok':>8s} {'unparsed':>8s}")
    for name, runs in groups.items():
        tot, omit = collections.Counter(), []
        for r in runs:
            loaded = load_task(r, args.task)
            if loaded is None:
                print(f"  skip {r}: no task{args.task} predictions")
                continue
            c = breakdown(*loaded)
            tot += c
            omit.append(100 * c["omitted"] / max(c["gold"], 1))
        if not omit:
            continue
        s = summarize(tot, omit)
        print(f"{name:45s} {s['n']:2d} {s['gold']:5d} {s['correct']:8.1f} {s['wrong_type']:9.1f} "
              f"{s['omitted']:8.1f} {s['omit_sd']:7.1f} {s['spurious']:8.1f} {s['arg_f1_correct']:8.1f} "
              f"{s['unparsed']:8.2f}")


if __name__ == "__main__":
    main()
