"""evaluate() with --gen-backend vllm, run against tests/fake_vllm.py."""
import json
import os
import random

import pytest

from conftest import EngineShim, ced_args, need, tiny_qwen
from data_utils.lm_datasets import LMTrainDataset

ACE_TEST = "processed_data/ace_b10_perm0/0/qwen/"


@pytest.fixture(scope="module")
def ace_test(tokenizer):
    return LMTrainDataset(ced_args(max_length=470), tokenizer, need(ACE_TEST), "test", 40, 1, random.Random(0))


def vllm_args(tmp_path, fake_vllm, **overrides):
    os.makedirs(tmp_path / "run", exist_ok=True)
    return ced_args(max_length=470, save=str(tmp_path / "run"), eval_batch_size=16, gen_backend="vllm",
                    vllm_py=fake_vllm.python, model_path=str(tmp_path / "base"), **overrides)


def test_vllm_gets_each_rows_unpadded_prompt_in_order_with_todays_cap(tmp_path, tokenizer, ace_test, fake_vllm):
    from ced_eval import evaluate
    evaluate(vllm_args(tmp_path, fake_vllm), tokenizer, tiny_qwen(0).eval(), ace_test, "test", 3, "cpu")
    call = fake_vllm.calls()[0]
    expected = []
    for i in range(len(ace_test)):
        gen_data = ace_test.collate([ace_test[i]])[2]
        expected.append(gen_data["input_ids"][0][gen_data["attention_mask"][0].bool()].tolist())
    assert [r["prompt_token_ids"] for r in call["requests"]] == expected
    assert {r["max_tokens"] for r in call["requests"]} == {470 - 460}        # max_length - prompt width
    assert len({r["seed"] for r in call["requests"]}) == len(ace_test)


def test_vllms_answers_are_decoded_and_scored_like_generates(tmp_path, tokenizer, ace_test, fake_vllm):
    from ced_eval import evaluate
    evaluate(vllm_args(tmp_path, fake_vllm), tokenizer, tiny_qwen(0).eval(), ace_test, "test", 3, "cpu")
    requests = fake_vllm.calls()[0]["requests"]
    # generate()'s path pads every answer with eos and decodes without special tokens
    expected = [tokenizer.decode([100 + sum(r["prompt_token_ids"]) % 1000] + [tokenizer.pad_token_id] * 3,
                                 skip_special_tokens=True) for r in requests]
    run = tmp_path / "run"
    assert [json.loads(line)["text"] for line in open(run / "eval" / "3" / "answers.jsonl")] == expected
    assert "'trigger'" in open(run / "log.txt").read()
    assert not (run / "vllm_tmp").exists()


def test_vllm_gets_the_settings_generate_would_use(tmp_path, tokenizer, ace_test, fake_vllm):
    from ced_eval import evaluate
    evaluate(vllm_args(tmp_path, fake_vllm), tokenizer, tiny_qwen(0).eval(), ace_test, "test", 0, "cpu")
    params = fake_vllm.calls()[0]["params"]
    assert (params["temperature"], params["top_p"], params["top_k"]) == (0.5, 0.95, 0)  # the Qwen3 fill-in
    assert params["stop_token_ids"] == [151645, 151643]


def test_a_lora_model_goes_to_vllm_as_its_adapter_on_the_starting_model(tmp_path, tokenizer, ace_test,
                                                                        fake_vllm, student):
    from ced_eval import evaluate
    evaluate(vllm_args(tmp_path, fake_vllm), tokenizer, EngineShim(student).eval(), ace_test, "test", 0, "cpu")
    call = fake_vllm.calls()[0]
    assert call["args"]["model"] == str(tmp_path / "base")
    assert call["args"]["lora"] == str(tmp_path / "run" / "vllm_tmp" / "adapter")
    assert call["args"]["max_lora_rank"] == 4
    assert {"adapter_config.json", "adapter_model.safetensors"} <= set(call["lora_files"])


def test_the_startup_check_refuses_more_than_one_process(monkeypatch, tokenizer):
    import torch.distributed as dist
    from ced_eval import check_gen_backend
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="world size 1"):
        check_gen_backend(ced_args(gen_backend="vllm"), tiny_qwen(0), tokenizer)


def test_the_startup_check_refuses_without_vllm(monkeypatch, tmp_path, tokenizer):
    from ced_eval import check_gen_backend
    monkeypatch.setenv("VLLM_PY", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="no vLLM environment"):
        check_gen_backend(ced_args(gen_backend="vllm"), tiny_qwen(0), tokenizer)


def test_the_startup_check_refuses_settings_vllm_cannot_reproduce(fake_vllm, tokenizer):
    from ced_eval import check_gen_backend
    from gen_backend import UnsupportedForVLLM
    model = tiny_qwen(0)
    model.generation_config.no_repeat_ngram_size = 6      # filled in, so generate() would block n-grams
    with pytest.raises(UnsupportedForVLLM, match="no_repeat_ngram_size"):
        check_gen_backend(ced_args(gen_backend="vllm"), model, tokenizer)


def test_the_startup_check_records_the_interpreter(fake_vllm, tokenizer):
    from ced_eval import check_gen_backend
    args = ced_args(gen_backend="vllm")
    check_gen_backend(args, tiny_qwen(0), tokenizer)
    assert (args.vllm_py, args.vllm_version) == (fake_vllm.python, "0.27.1")


def test_the_hf_backend_needs_no_check(monkeypatch, tmp_path, tokenizer):
    from ced_eval import check_gen_backend
    monkeypatch.setenv("VLLM_PY", str(tmp_path / "missing"))
    check_gen_backend(ced_args(), tiny_qwen(0), tokenizer)


def test_both_trainers_check_the_backend_before_training():
    import inspect

    import ced_finetune
    import finetune
    for trainer in (ced_finetune, finetune):
        assert "check_gen_backend(args, model, tokenizer)" in inspect.getsource(trainer.main)
