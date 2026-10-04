"""Omission-mask for on-policy self-distillation (--ced-sd-omission-mask).

The SD teacher reads the augmented answer y~ (gold + pseudo-labels). The pseudo-label filter
keeps precision over recall, so it drops some valid old records, and the teacher, which never
saw them in its reference, gives them low probability. When the student writes such a record
itself, SD would push it away from correct recall. The mask drops the tokens of every sampled
record that
  - has an event type of an earlier task,
  - has a trigger that occurs verbatim in the input sentence,
  - and whose trigger matches no trigger in y~ (same overlap test as the pseudo-label dedup),
from the SD loss. Records whose trigger is in y~ under another type stay in: that is a wrong
type, which the teacher should correct.

Pure Python plus a tokenizer, so it can be tested without a GPU.
"""
import json
import re


def record_spans(text):
    """Char spans [start, end) of each record (inner list) of the "events" array, or [] when
    the text holds no such array. Scans brackets outside JSON strings, so it works on the
    exact string the model wrote, whatever its spacing."""
    m = re.search(r'"events"\s*:\s*\[', text)
    if not m:
        return []
    spans, depth, start, in_str, esc = [], 1, None, False, False
    for i in range(m.end(), len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
            if depth == 2 and ch == "[":
                start = i
        elif ch in "]}":
            depth -= 1
            if depth == 1 and start is not None:
                spans.append((start, i + 1))
                start = None
            elif depth == 0:
                break
    return spans


def answer_triggers(text):
    """lower-cased triggers of a reference answer"""
    try:
        events = json.loads(text.strip()).get("events", [])
    except Exception:
        return set()
    return {str(e[0]).lower() for e in events if isinstance(e, list) and e}


def input_sentence(prompt_text):
    m = re.search(r"Given an input text: ?\n?(.*?)\n\nYour task", prompt_text, re.S)
    return m.group(1) if m else prompt_text


def masked_spans(response_text, reference_text, sentence, old_types):
    """Char spans of the sampled records the mask removes from the SD loss."""
    ref_trig = answer_triggers(reference_text)
    out = []
    for s, e in record_spans(response_text):
        try:
            rec = json.loads(response_text[s:e])
        except Exception:
            continue
        if not (isinstance(rec, list) and len(rec) >= 2 and isinstance(rec[0], str) and isinstance(rec[1], str)):
            continue
        trig, ty = rec[0], rec[1]
        if ty not in old_types or not trig or trig not in sentence:
            continue
        tl = trig.lower()
        if any(tl == g or tl in g or g in tl for g in ref_trig):
            continue
        out.append((s, e))
    return out


def token_char_ends(tokenizer, ids):
    """End char offset of each token in decode(ids). Decoding prefixes rather than single tokens
    keeps multi-byte characters split across tokens intact."""
    return [len(tokenizer.decode(ids[:k + 1], skip_special_tokens=True)) for k in range(len(ids))]


def token_mask(tokenizer, ids, spans):
    """True for each response token that overlaps one of the char spans"""
    ends = token_char_ends(tokenizer, ids)
    starts = [0] + ends[:-1]
    return [any(st < e and en > s for s, e in spans) and en > st for st, en in zip(starts, ends)]
