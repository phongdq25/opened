"""Build the canonical Continual Relation Extraction relation groups (order-independent).

Source: the WAVE-CRE-PLUS-PLUS release of the standard CRE benchmarks
(FewRel 80 rel x 700, TACRED 40 rel, entity-marker tokens).

Writes ONE copy of the data, split per relation group, with no cumulation and no
replay buffer -- so it is independent of the stream order:
    data/<ds>_groups/groups.json          list of 10 relation-type lists
    data/<ds>_groups/{g}/{train,dev,test}.jsonl

A stream order (perm) is then materialized on demand by cre_materialize_perm.py,
which concatenates these groups. That keeps disk at 1x the dataset instead of 55x
per perm.

The response reuses the CED "events" schema so ed_eval.py, ced_pseudo_label.py
and ced_oversample.py all run untouched:
    {"events": [[<subject span>, <relation type>, [[<object span>, "object"]], <desc>]]}
so ed_eval's "trigger" F1 == (subject, relation) F1 and "argument" F1 == full
triple F1, "trigger_per_type" == per-relation F1.

Usage (from the OpenED dir):
    python tools/build_cre_groups.py --dataset fewrel
    python tools/build_cre_groups.py --dataset tacred
"""
import argparse
import json
import os
import random

GROUPING_SEED = 2021  # fixes relation -> group assignment; shared by every perm

SYSTEM_PROMPT = (
    "You are a relation extraction system. Your task is to identify the relation "
    "that holds between two given entities in a text.\n"
    "IMPORTANT:Output ONLY valid JSON. No explanations, no markdown, no extra text.\n"
    "Output Format (JSON only, no markdown):\n\n"
    '{"events": [[<subject span>, <relation type>, [[<object span>, "object"]], <description>]]}\n\n'
    '- If no relation holds between the two entities, return: {"events": []}'
)

USER_TEMPLATE = (
    "Given an input text: \n<input>\n{input}\n</input>\n\n"
    "Subject entity: {subject}\nObject entity: {object}\n\n"
    "Your task is to identify the relation that holds between the subject entity "
    "and the object entity.\n\n"
    "For each relation:\n"
    "- Identify the relation type\n"
    "- Description: explaining what this relation type means\n\n"
    "Constraints and Guidelines\n"
    "- Do not invent information not supported by the text.\n"
    "- Do not paraphrase the entity spans.\n"
    "- The subject and object spans must exactly match the spans given above.\n"
)


def load_descriptions(path, key_col):
    """relation_description.txt: FewRel = idx \\t name \\t gloss, TACRED = key \\t name \\t gloss."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            key = str(i) if key_col == "index" else parts[0]
            out[key] = (parts[1].strip(), parts[2].strip())
    return out


def split_markers(tokens):
    """Strip [E11]/[E12] (subject) and [E21]/[E22] (object); return sentence + spans."""
    words, subject, obj = [], [], []
    in_sub = in_obj = False
    for tok in tokens:
        if tok == "[E11]":
            in_sub = True
        elif tok == "[E12]":
            in_sub = False
        elif tok == "[E21]":
            in_obj = True
        elif tok == "[E22]":
            in_obj = False
        else:
            words.append(tok)
            if in_sub:
                subject.append(tok)
            if in_obj:
                obj.append(tok)
    return " ".join(words), " ".join(subject), " ".join(obj)


def make_record(tokens, rel_name, description):
    sent, subject, obj = split_markers(tokens)
    if not subject or not obj:
        return None
    response = json.dumps({"events": [[subject, rel_name, [[obj, "object"]], description]]})
    return {"system_prompt": SYSTEM_PROMPT,
            "user_prompt": USER_TEMPLATE.format(input=sent, subject=subject, object=obj),
            "response": response}


def per_relation_split(samples, dataset):
    """FewRel: 420/140/140 (WAVE sampler.py). TACRED: test/dev = min(n//5, 40), train capped 320."""
    n = len(samples)
    if dataset == "fewrel":
        return samples[:420], samples[420:560], samples[560:700]
    k = min(n // 5, 40)
    return samples[2 * k:][:320], samples[k:2 * k], samples[:k]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["fewrel", "tacred"], required=True)
    ap.add_argument("--src", default="../datasets_wave_cre",
                    help="dir holding data_with_marker*.json and FewRel/ TACRED/")
    ap.add_argument("--out-root", default="data")
    ap.add_argument("--num-groups", type=int, default=10)
    args = ap.parse_args()

    if args.dataset == "fewrel":
        data = json.load(open(f"{args.src}/data_with_marker.json", encoding="utf-8"))
        desc = load_descriptions(f"{args.src}/FewRel/relation_description.txt", "index")
        id2rel = json.load(open(f"{args.src}/id2rel.json", encoding="utf-8"))
        # FewRel descriptions are indexed by position in id2rel.json
        rel_meta = {rel: desc[str(i)] for i, rel in enumerate(id2rel)}
    else:
        data = json.load(open(f"{args.src}/data_with_marker_tacred.json", encoding="utf-8"))
        desc = load_descriptions(f"{args.src}/TACRED/relation_description.txt", "key")
        rel_meta = {rel: desc[rel] for rel in data}

    missing = [r for r in data if r not in rel_meta]
    assert not missing, f"no description for: {missing}"
    n_rel = len(data)
    assert n_rel % args.num_groups == 0, f"{n_rel} relations not divisible by {args.num_groups}"
    rel_per_group = n_rel // args.num_groups

    # relation type string used in the response: readable name for FewRel
    # (P931 means nothing to an LM), native key for TACRED (already readable).
    type_name = {r: (rel_meta[r][0] if args.dataset == "fewrel" else r) for r in data}

    order = list(data.keys())
    random.Random(GROUPING_SEED).shuffle(order)

    out_dir = f"{args.out_root}/{args.dataset}_groups"
    os.makedirs(out_dir, exist_ok=True)
    groups = [[type_name[r] for r in order[g * rel_per_group:(g + 1) * rel_per_group]]
              for g in range(args.num_groups)]
    json.dump(groups, open(f"{out_dir}/groups.json", "w"))

    print(f"=== {args.dataset}: {n_rel} relations -> {args.num_groups} groups "
          f"x {rel_per_group} (grouping seed {GROUPING_SEED}) -> {out_dir}")
    for g in range(args.num_groups):
        rels = order[g * rel_per_group:(g + 1) * rel_per_group]
        os.makedirs(f"{out_dir}/{g}", exist_ok=True)
        splits = {"train": [], "dev": [], "test": []}
        for rel in rels:
            recs = [make_record(s["tokens"], type_name[rel], rel_meta[rel][1])
                    for s in data[rel]]
            dropped = sum(r is None for r in recs)
            if dropped:
                print(f"  warn: {rel} dropped {dropped} samples with missing entity markers")
            tr, dv, te = per_relation_split([r for r in recs if r], args.dataset)
            splits["train"] += tr
            splits["dev"] += dv
            splits["test"] += te
        n = {k: write_jsonl(f"{out_dir}/{g}/{k}.jsonl", v) for k, v in splits.items()}
        print(f"group {g}: {groups[g]}\n         "
              f"train={n['train']} dev={n['dev']} test={n['test']}")


if __name__ == "__main__":
    main()
