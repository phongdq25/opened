"""tools/ced_pseudo_label.py with --gen-backend vllm."""
import importlib.util
import json
import os
import sys

import pytest
import torch

from conftest import MODEL_DIR, need, tiny_qwen

_SPEC = importlib.util.spec_from_file_location("ced_pseudo_label", os.path.join("tools", "ced_pseudo_label.py"))
pl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pl)


@pytest.fixture(scope="module")
def teacher_dir(tmp_path_factory):
    """A tiny teacher saved like a merged model, next to the real tokenizer."""
    from transformers import AutoTokenizer
    path = tmp_path_factory.mktemp("teacher")
    tiny_qwen(0).save_pretrained(path)
    AutoTokenizer.from_pretrained(need(MODEL_DIR)).save_pretrained(path)
    return str(path)


def row(sentence, events):
    return {"system_prompt": "You extract events.",
            "user_prompt": f"Given an input text: {sentence}\n\nYour task is to list the events.",
            "response": json.dumps({"events": events})}


def write_tasks(root, task1_rows):
    """Two tasks: task 0 teaches TypeA, task 1 adds TypeB."""
    root.mkdir(parents=True)
    (root / "streams.json").write_text(json.dumps([["TypeA"], ["TypeB"]]))
    for task, rows in ((0, [row("They attack the town.", [["attack", "TypeA", [], ""]])]), (1, task1_rows)):
        (root / str(task)).mkdir()
        for split in ("train", "dev", "test"):
            (root / str(task) / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def run_main(monkeypatch, teacher, data, out):
    monkeypatch.setattr(sys, "argv", ["ced_pseudo_label.py", "--teacher", teacher, "--data-dir", str(data / "1"),
                                      "--streams", str(data / "streams.json"), "--task-id", "1", "--out", str(out),
                                      "--conf-filter", "percentile", "--gen-backend", "vllm"])
    pl.main()


def test_vllms_tokens_rebuild_generates_texts_ids_and_scores(teacher_dir, tmp_path, fake_vllm, monkeypatch):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(teacher_dir, padding_side="left")
    rows = [row(f"Sentence {i}: they meet and attack.", [["meet", "TypeB", [], ""]]) for i in range(4)]
    prompts = pl.chat_prompts(tok, rows, range(4))
    hf = pl.generate_hf(AutoModelForCausalLM.from_pretrained(teacher_dir).eval(), tok, prompts, 8, True, "cpu")
    # vLLM returns what generate() produced: its tokens (without padding) and their log-probabilities
    replay = [{"token_ids": gid, "logprobs": lp[:len(gid)].tolist()} for _, gid, lp in hf]
    (tmp_path / "replay.json").write_text(json.dumps(replay))
    monkeypatch.setenv("FAKE_VLLM_MODE", "replay")
    monkeypatch.setenv("FAKE_VLLM_REPLAY", str(tmp_path / "replay.json"))
    vl = pl.generate_vllm(teacher_dir, tok, prompts, 8, True, str(tmp_path / "work"), fake_vllm.python)
    for (text_h, gid_h, lp_h), (text_v, gid_v, lp_v) in zip(hf, vl):
        assert (text_v, gid_v) == (text_h, gid_h)
        torch.testing.assert_close(lp_v[:len(gid_v)], lp_h[:len(gid_h)])
    call = fake_vllm.calls()[0]
    assert [r["prompt_token_ids"] for r in call["requests"]] == tok(prompts, truncation=True, max_length=1024)["input_ids"]
    assert {r["max_tokens"] for r in call["requests"]} == {8}
    assert call["params"]["temperature"] == 0.0 and call["params"]["logprobs"] is True
    assert call["params"]["stop_token_ids"] == [151645, 151643]          # the teacher's generation_config


def test_main_with_vllm_writes_todays_outputs(teacher_dir, tmp_path, fake_vllm, monkeypatch):
    task1 = [row("They meet.", [["meet", "TypeB", [], ""]]), row("We meet again.", [["meet", "TypeB", [], ""]]),
             row("They attack.", [["attack", "TypeA", [], ""]])]
    data = write_tasks(tmp_path / "data", task1)
    run_main(monkeypatch, teacher_dir, data, tmp_path / "out")
    stats = json.load(open(tmp_path / "out" / "pl_stats.json"))
    assert stats["candidates"] == 2
    calls = fake_vllm.calls()
    assert len(calls) == 1 and len(calls[0]["requests"]) == 2
    # the fake's answers parse to no events: every row is written back unchanged
    assert [json.loads(line) for line in open(tmp_path / "out" / "train.jsonl")] == task1
    assert (tmp_path / "out" / "test.jsonl").exists() and not (tmp_path / "out" / "vllm_tmp").exists()


def test_a_task_without_candidates_starts_no_vllm(teacher_dir, tmp_path, fake_vllm, monkeypatch):
    task1 = [row("They attack.", [["attack", "TypeA", [], ""]]), row("Nothing happens.", [])]
    data = write_tasks(tmp_path / "data", task1)
    run_main(monkeypatch, teacher_dir, data, tmp_path / "out")
    assert fake_vllm.calls() == []
    assert json.load(open(tmp_path / "out" / "pl_stats.json"))["candidates"] == 0
    assert [json.loads(line) for line in open(tmp_path / "out" / "train.jsonl")] == task1
