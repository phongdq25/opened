#!/usr/bin/env python3
"""Pick the adapter checkpoint with the best DEV trigger-F1 for a CED task.

Reads per-epoch dev evals from the task's log.txt, aligns them with the saved
adapter dirs by count (dropping any leading pre-train eval), and prints the best
epoch's adapter dir to stdout. Diagnostics go to stderr. Falls back to the LAST
adapter dir if selection is ambiguous, so it can never break a run.
"""
import argparse, glob, os, re, sys


def find_adapters(save_dir):
    dirs = {os.path.dirname(p) for p in
            glob.glob(os.path.join(save_dir, "**", "adapter_config.json"), recursive=True)}
    def stepkey(d):
        b = os.path.basename(d)
        return (0, int(b)) if b.isdigit() else (1, 0, b)
    return sorted(dirs, key=stepkey)


def dev_f1s(save_dir):
    # use the finetune log.txt (save_rank), not train.log; pick the log with the
    # most 'dev |' lines under this task dir.
    best = []
    for lp in glob.glob(os.path.join(save_dir, "**", "log.txt"), recursive=True):
        vals = []
        for line in open(lp, errors="ignore"):
            s = line.strip()
            if s.startswith("dev |"):
                m = re.search(r"'trigger':\s*\{[^}]*'f1':\s*([0-9.]+)", s)
                if m:
                    vals.append(float(m.group(1)))
        if len(vals) > len(best):
            best = vals
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", required=True)
    args = ap.parse_args()

    adapters = find_adapters(args.save_dir)
    if not adapters:
        sys.stderr.write("pick_best_ckpt: no adapter dirs found\n")
        sys.exit(1)

    dev = dev_f1s(args.save_dir)
    N = len(adapters)
    if len(dev) < N:
        sys.stderr.write(f"pick_best_ckpt: dev evals ({len(dev)}) < adapters ({N}); "
                         f"FALLBACK to last {adapters[-1]}\n")
        print(adapters[-1])
        return

    dev_real = dev[-N:]  # drop leading pre-train eval(s) so dev_real[i] <-> adapters[i]
    best = max(range(N), key=lambda i: dev_real[i])
    sys.stderr.write(
        f"pick_best_ckpt: dev_f1(last{N})={['%.3f' % x for x in dev_real]} -> "
        f"best idx {best} (dev {dev_real[best]:.3f}) [{os.path.basename(adapters[best])}]; "
        f"last idx {N-1} (dev {dev_real[-1]:.3f})\n")
    print(adapters[best])


if __name__ == "__main__":
    main()
