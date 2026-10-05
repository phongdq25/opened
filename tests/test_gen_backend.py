"""gen_backend.py: which settings vLLM gets, what it is asked, and how its process runs."""
import os

import pytest
import torch
from transformers import GenerationConfig

from conftest import LORA_TARGETS, EngineShim, tiny_qwen

QWEN_EOS = [151645, 151643]
# what evaluate() builds under --greedy 1 (runner flags --top-k 0 --top-p 0.95 --temperature 0.5)
EVAL = dict(do_sample=False, temperature=0.5, top_p=0.95, top_k=0, repetition_penalty=None,
            eos_token_id=QWEN_EOS, pad_token_id=151645)
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
          "stop_token_ids": QWEN_EOS, "logprobs": False}


def request(prompt, max_tokens=4, seed=0):
    return {"prompt_token_ids": prompt, "max_tokens": max_tokens, "seed": seed}


def test_todays_evaluation_resolves_to_sampling_through_the_wrappers(student):
    from gen_backend import resolve_generation_config, vllm_params
    g = resolve_generation_config(EngineShim(student), GenerationConfig(**EVAL))
    assert (g.do_sample, g.temperature, g.top_p, g.top_k) == (True, 0.5, 0.95, 0)   # the Qwen3 fill-in
    assert vllm_params(g) == {**GREEDY, "temperature": 0.5, "top_p": 0.95}


def test_strict_generation_resolves_to_greedy():
    from gen_backend import resolve_generation_config, vllm_params
    g = resolve_generation_config(tiny_qwen(0), GenerationConfig(**EVAL), use_model_defaults=False)
    assert g.do_sample is False
    assert vllm_params(g) == GREEDY


def test_cl_lora_and_pseudo_label_calls_resolve_to_greedy_with_the_models_stop_tokens():
    from gen_backend import resolve_generation_config, vllm_params
    g = resolve_generation_config(tiny_qwen(0), None, do_sample=False, pad_token_id=151643)
    assert vllm_params(g, logprobs=True) == {**GREEDY, "logprobs": True}


@pytest.mark.parametrize("field,value", [("num_beams", 2), ("no_repeat_ngram_size", 6), ("typical_p", 0.9),
                                         ("min_length", 5), ("bad_words_ids", [[7]])])
def test_settings_vllm_cannot_reproduce_are_refused(field, value):
    from gen_backend import UnsupportedForVLLM, resolve_generation_config, vllm_params
    g = resolve_generation_config(tiny_qwen(0), GenerationConfig(**{**EVAL, field: value}))
    with pytest.raises(UnsupportedForVLLM, match=field):
        vllm_params(g)


def test_prompts_are_the_tokens_the_attention_mask_keeps():
    from gen_backend import unpadded_prompts
    ids = torch.tensor([[0, 0, 5, 6, 7], [0, 8, 9, 0, 10]])
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 1]])
    assert unpadded_prompts(ids, mask) == [[5, 6, 7], [8, 9, 10]]


def test_row_seeds_are_reproducible_and_distinct():
    from gen_backend import row_seed
    seeds = {row_seed(42, split, epoch, i) for split in ("dev", "test") for epoch in (0, 1) for i in range(100)}
    assert len(seeds) == 400 and all(0 <= s < 2 ** 31 for s in seeds)
    assert row_seed(42, "test", 1, 7) == row_seed(42, "test", 1, 7)


def test_run_vllm_round_trip_without_torchruns_variables(fake_vllm, tmp_path, monkeypatch):
    from gen_backend import run_vllm
    for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID"):
        monkeypatch.setenv(var, "1")
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    work = tmp_path / "work"
    requests = [request([5, 6], seed=1), request([7, 8, 9], seed=2)]
    out = run_vllm(str(work), "some/model", requests, {**GREEDY, "logprobs": True}, lora_dir="some/adapter",
                   max_lora_rank=16, vllm_py=fake_vllm.python)
    assert out == [{"token_ids": [111, 151645], "logprobs": [-0.5, -0.1]},
                   {"token_ids": [124, 151645], "logprobs": [-0.5, -0.1]}]
    call = fake_vllm.calls()[0]
    assert call["requests"] == requests and call["params"] == {**GREEDY, "logprobs": True}
    assert (call["args"]["model"], call["args"]["lora"], call["args"]["max_lora_rank"]) == ("some/model", "some/adapter", 16)
    assert call["args"]["max_model_len"] == 7                       # the longest prompt plus its max_tokens
    env = call["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "3" and env["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert not {"RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID",
                "PYTHONPATH"} & set(env)
    assert os.listdir(work) == []                                     # the files go once it worked


def test_a_short_answer_file_fails_loudly(fake_vllm, tmp_path, monkeypatch):
    from gen_backend import run_vllm
    monkeypatch.setenv("FAKE_VLLM_MODE", "short")
    with pytest.raises(RuntimeError, match="1 answers for 2 requests"):
        run_vllm(str(tmp_path / "w"), "m", [request([1]), request([2])], GREEDY, vllm_py=fake_vllm.python)


def test_a_vllm_crash_fails_with_its_last_log_lines_and_keeps_them(fake_vllm, tmp_path, monkeypatch):
    from gen_backend import run_vllm
    monkeypatch.setenv("FAKE_VLLM_MODE", "fail")
    work = tmp_path / "w"
    with pytest.raises(RuntimeError, match="(?s)exit 3.*boom"):
        run_vllm(str(work), "m", [request([1])], GREEDY, vllm_py=fake_vllm.python)
    assert "boom" in (work / "vllm.log").read_text()


def test_no_requests_start_no_vllm(fake_vllm, tmp_path):
    from gen_backend import run_vllm
    assert run_vllm(str(tmp_path / "w"), "m", [], GREEDY, vllm_py=fake_vllm.python) == []
    assert fake_vllm.calls() == []


def test_find_vllm_python_takes_vllm_py_and_its_version(fake_vllm):
    from gen_backend import find_vllm_python
    assert find_vllm_python() == (fake_vllm.python, "0.27.1")


def test_find_vllm_python_does_not_fall_back_when_vllm_py_is_set(fake_vllm, monkeypatch, tmp_path):
    import gen_backend
    monkeypatch.setattr(gen_backend, "VLLM_CANDIDATES", (fake_vllm.python,))
    monkeypatch.setenv("VLLM_PY", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="no vLLM environment"):
        gen_backend.find_vllm_python()
    monkeypatch.delenv("VLLM_PY")
    assert gen_backend.find_vllm_python() == (fake_vllm.python, "0.27.1")


def test_an_exported_adapter_reproduces_the_peft_model(tmp_path):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM
    from gen_backend import export_for_vllm
    base = tiny_qwen(0)
    base.save_pretrained(tmp_path / "base")
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=16, lora_dropout=0.0,
                                            target_modules=LORA_TARGETS))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.02)
    model_dir, lora_dir, rank = export_for_vllm(EngineShim(model), str(tmp_path / "base"), str(tmp_path / "x"))
    assert (model_dir, rank) == (str(tmp_path / "base"), 4)
    reloaded = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_dir), lora_dir)
    ids = torch.tensor([[5, 6, 7, 8]])
    torch.testing.assert_close(reloaded(input_ids=ids).logits, model(input_ids=ids).logits)


def test_a_model_without_lora_is_exported_whole(tmp_path):
    from transformers import AutoModelForCausalLM
    from gen_backend import export_for_vllm
    model = tiny_qwen(0)
    model_dir, lora_dir, rank = export_for_vllm(model, "unused", str(tmp_path / "x"))
    assert lora_dir is None and rank == 0
    ids = torch.tensor([[5, 6, 7, 8]])
    torch.testing.assert_close(AutoModelForCausalLM.from_pretrained(model_dir)(input_ids=ids).logits,
                               model(input_ids=ids).logits)


def test_a_weightless_model_resolves_like_the_saved_one(tmp_path):
    from gen_backend import meta_model, resolve_generation_config
    tiny_qwen(0).save_pretrained(tmp_path / "m")
    g = resolve_generation_config(meta_model(str(tmp_path / "m")), None, do_sample=False)
    assert g.eos_token_id == QWEN_EOS and g.do_sample is False
