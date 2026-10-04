"""Build CED task splits for all SharpSeq permutations.

Replicates exp_ace.ipynb (which generated data/ace/{0..4}, perm 0) and applies
the 5 SharpSeq stream-order permutations. Streams are reconstructed from the
existing perm-0 split; prompts are extracted from the existing data so output
format matches exactly.

Usage (on server, ced env):
    python tools/build_ced_perms.py                          # data/ace_perm{p}/{t}/*.jsonl, cap 5
    python tools/build_ced_perms.py --cap 10 --perms 0 --out-prefix ace_b10_perm
    python tools/build_ced_perms.py --cap 0 --out-prefix ace_b0_perm                   # no rehearsal
    python tools/build_ced_perms.py --cap 10 --oracle --out-prefix ace_oracle_b10_perm # no label stripping
"""
import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

from datasets import load_dataset

# repo root: the cwd this is run from, so the script works on any host. OPENED_BASE overrides.
BASE = os.environ.get("OPENED_BASE", os.getcwd())
SRC = "datht/ace-short-generated-dataset"
EXISTING = f"{BASE}/data/ace"

PERM = [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [0, 3, 1, 4, 2], [1, 2, 0, 3, 4], [3, 4, 0, 1, 2]]

cli = argparse.ArgumentParser()
cli.add_argument("--cap", type=int, default=5, help="exemplar buffer cap per event type")
cli.add_argument("--perms", type=int, nargs="+", default=[0, 1, 2, 3, 4])
cli.add_argument("--out-prefix", type=str, default="ace_perm")
cli.add_argument("--seed", type=int, default=42)
cli.add_argument("--oracle", action="store_true",
                 help="train rows keep the events of earlier tasks' types too (no label stripping); "
                      "same sentences, buffer and dev/test as without the flag")
CLI = cli.parse_args()

OUT = f"{BASE}/data/{CLI.out_prefix}{{p}}"
SEED = CLI.seed


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def events_of(d):
    r = d["response"]
    if isinstance(r, str):
        r = json.loads(r)
    return r.get("events", [])


# ---------- 1. reconstruct streams (type names) from existing perm-0 split ----------
def reconstruct_streams():
    cum = []
    for t in range(5):
        ts = set()
        for split in ["train", "dev", "test"]:
            for d in load_jsonl(f"{EXISTING}/{t}/{split}.jsonl"):
                for e in events_of(d):
                    ts.add(e[1])
        cum.append(ts)
    streams = [sorted(cum[0])]
    for t in range(1, 5):
        streams.append(sorted(cum[t] - cum[t - 1]))
    total = sum(len(s) for s in streams)
    assert total == 33, f"expected 33 ACE types, got {total}"
    return streams


# ---------- 2. extract prompt templates from existing data ----------
def extract_templates():
    recs = load_jsonl(f"{EXISTING}/0/train.jsonl")
    system_prompt = recs[0]["system_prompt"]
    assert all(r["system_prompt"] == system_prompt for r in recs)

    up = recs[0]["user_prompt"]
    m = re.match(r"(Given an input text: )(.*?)(\n\nYour task.*)$", up, re.S)
    assert m, "unexpected user_prompt format"
    prefix, _, suffix = m.groups()
    template = prefix + "{input}" + suffix
    # sanity: template must reproduce every existing record
    pat = re.compile(re.escape(prefix) + r"(.*?)" + re.escape(suffix) + r"$", re.S)
    for r in recs[:200]:
        mm = pat.match(r["user_prompt"])
        assert mm and template.format(input=mm.group(1)) == r["user_prompt"]
    return system_prompt, template


# ---------- 3. notebook logic, verbatim (type names instead of label ids) ----------
def process_ace_data(raw_data, buffer, system_prompt, user_template,
                     is_test=False, tasks=(), eval_tasks=(), old_types=()):
    # old_types (--oracle, train only): earlier tasks' types, kept in the response of a
    # sentence the current types already select. They never select a sentence themselves,
    # so with old_types empty this is the notebook logic unchanged.
    tasks = set(tasks)
    eval_tasks = set(eval_tasks)
    old_types = set(old_types)
    data = []
    none_data = []

    for idx, sample in enumerate(raw_data):
        sent_id_to_sentence = {i: c["sentence"] for i, c in enumerate(sample["content"])}
        sent_id_set = set(sent_id_to_sentence.keys())
        sent_to_existing_events = {}
        selected = set()  # sentences with an event of a non-old type
        temp = {}  # marks handled below via b_sent_id_map
        b_sent_id_map = {}

        for event in sample.get("events", []):
            if event.get("type_id", -1) == -1:
                continue
            event_type = event.get("type")
            description = event.get("description", "")
            is_old = event_type in old_types

            if event_type not in tasks and not is_old:
                if not (is_test and event_type in eval_tasks):
                    continue

            if not is_old and event_type not in process_ace_data.temp_buffer:
                process_ace_data.temp_buffer[event_type] = []

            for mention in event.get("mention", []):
                sent_id = mention.get("sent_id")
                if sent_id not in sent_to_existing_events:
                    sent_to_existing_events[sent_id] = []
                args = [[a["text"], a["role"]] for a in mention.get("arguments", [])]
                event_info = [mention.get("trigger_word"), event_type, args, description]
                sent_to_existing_events[sent_id].append(event_info)
                if is_old:
                    continue
                selected.add(sent_id)

                if len(process_ace_data.temp_buffer[event_type]) < CLI.cap:
                    b_sent_id_map[f"{idx}_{sent_id}"] = event_type

        for sent_id, evs in sent_to_existing_events.items():
            if sent_id not in selected:
                continue  # old-type events only: stays a none sentence, labelled below
            sent_txt = sent_id_to_sentence[sent_id]
            sent_id_set.remove(sent_id)
            response = json.dumps({"events": evs})
            data.append({"system_prompt": system_prompt,
                         "user_prompt": user_template.format(input=sent_txt),
                         "response": response})
            key = f"{idx}_{sent_id}"
            if key in b_sent_id_map:
                process_ace_data.temp_buffer[b_sent_id_map[key]].append(data[-1])

        for sent_id in sent_id_set:
            sent_txt = sent_id_to_sentence[sent_id]
            none_data.append({"system_prompt": system_prompt,
                              "user_prompt": user_template.format(input=sent_txt),
                              "response": json.dumps({"events": sent_to_existing_events.get(sent_id, [])})})

    if not is_test:
        data.extend(random.sample(list(none_data), min(len(none_data), len(data) // 10)))
        data.extend(buffer)
        for v in process_ace_data.temp_buffer.values():
            buffer.extend(v)
    return data


def save_task(ds, out_dir, tasks, buffer, eval_tasks, system_prompt, user_template, old_types=()):
    os.makedirs(out_dir, exist_ok=True)
    stats = {}
    process_ace_data.temp_buffer = {}
    train = process_ace_data(ds["train"], buffer, system_prompt, user_template, tasks=tasks,
                             old_types=old_types)
    process_ace_data.temp_buffer = {}
    dev = process_ace_data(ds["validation"], [], system_prompt, user_template,
                           is_test=True, tasks=tasks, eval_tasks=eval_tasks)
    process_ace_data.temp_buffer = {}
    test = process_ace_data(ds["test"], [], system_prompt, user_template,
                            is_test=True, tasks=tasks, eval_tasks=eval_tasks)
    for name, rows in [("train", train), ("dev", dev), ("test", test)]:
        with open(f"{out_dir}/{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[name] = len(rows)
    return stats


def main():
    streams = reconstruct_streams()
    print("reconstructed streams (type counts):", [len(s) for s in streams])
    for i, s in enumerate(streams):
        print(f"  stream {i}: {s}")

    system_prompt, user_template = extract_templates()
    print("prompt templates extracted OK")

    ds = load_dataset(SRC)

    for p in CLI.perms:
        order = PERM[p]
        random.seed(SEED + p)
        perm_streams = [streams[t] for t in order]
        buffer = []
        eval_tasks = []
        print(f"\n=== perm {p} (stream order {order}, cap {CLI.cap}) ===")
        os.makedirs(OUT.format(p=p), exist_ok=True)
        with open(OUT.format(p=p) + "/streams.json", "w") as f:
            json.dump(perm_streams, f)
        for t, tasks in enumerate(perm_streams):
            old_types = list(eval_tasks) if CLI.oracle else []
            eval_tasks.extend(tasks)
            out_dir = OUT.format(p=p) + f"/{t}"
            stats = save_task(ds, out_dir, tasks, buffer, list(eval_tasks),
                              system_prompt, user_template, old_types)
            print(f"task {t}: streams[{order[t]}] ({len(tasks)} types) "
                  f"train={stats['train']} dev={stats['dev']} test={stats['test']} "
                  f"buffer_after={len(buffer)}")


if __name__ == "__main__":
    main()
