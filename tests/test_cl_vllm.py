"""CL-LoRA evaluation through vLLM: the exported adapter, eval_task's vLLM path, and the manifest."""
import json
import os
import sys
from argparse import Namespace

import pytest
import torch

from conftest import LORA_TARGETS, MODEL_DIR, need, tiny_qwen

TACRED = "data/tacred_perm0"


def summed_model(n_adapters, tmp_path):
    """A tiny PEFT model with n task adapters, all active (as consolidate() leaves them), its base saved."""
    from peft import LoraConfig, get_peft_model
    from cl_lora.multi_adapter import CLLoRAManager

    def config():
        return LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=16, lora_dropout=0.0, target_modules=LORA_TARGETS)

    base = tiny_qwen(0)
    base.save_pretrained(tmp_path / "base")
    model = get_peft_model(base, config(), adapter_name="task0")
    mgr = CLLoRAManager(model, orth_lambda=0.0)
    mgr.register_task0("task0")
    for task_id in range(1, n_adapters):
        mgr.start_new_task(task_id, config())
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.normal_(0, 0.05)
    mgr.consolidate()
    return model.eval(), mgr


def engine_args(tmp_path, method="inclora", vllm_py=None):
    return Namespace(cl_method=method, data_root=need(TACRED), max_length=768, max_prompt_length=460, limit=6,
                     eval_batch_size=4, gen_backend="vllm", vllm_py=vllm_py, model_path=str(tmp_path / "base"),
                     save=str(tmp_path / "run"))


@pytest.mark.parametrize("n_adapters", [1, 2, 3])
def test_the_concatenated_adapter_computes_the_consolidated_sum(n_adapters, tmp_path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from cl_lora.vllm_export import export_summed_adapters
    model, mgr = summed_model(n_adapters, tmp_path)
    rank = export_summed_adapters(model, mgr.task_adapters, str(tmp_path / "base"), str(tmp_path / "adapter"))
    assert rank == 4 * n_adapters
    reloaded = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(tmp_path / "base"), tmp_path / "adapter")
    ids = torch.tensor([[5, 6, 7, 8, 9]])
    torch.testing.assert_close(reloaded(input_ids=ids).logits, model(input_ids=ids).logits, atol=1e-5, rtol=0)


@pytest.mark.parametrize("method,requested,backend", [
    ("inclora", "vllm", "vllm"), ("olora", "vllm", "vllm"), ("inflora", "vllm", "vllm"), ("tree", "vllm", "vllm"),
    ("migu", "vllm", "vllm"), ("gainlora_o", "vllm", "hf"), ("gainlora_inf", "vllm", "hf"), ("epi", "vllm", "hf"),
    ("inclora", "hf", "hf")])
def test_gated_and_routed_methods_keep_hugging_face_generation(method, requested, backend):
    from cl_lora.vllm_export import effective_backend
    assert effective_backend(method, requested) == backend


def test_eval_task_sends_the_unpadded_prompts_and_decodes_vllms_answers(tmp_path, fake_vllm):
    from transformers import AutoTokenizer
    from cl_lora.engine import JsonlED, eval_task
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model, mgr = summed_model(2, tmp_path)
    a = engine_args(tmp_path, vllm_py=fake_vllm.python)
    os.makedirs(a.save)
    metrics, preds, refs = eval_task(a, model, tok, "cpu", 1, mgr)
    call = fake_vllm.calls()[0]
    ds = JsonlED(os.path.join(a.data_root, "1", "test.jsonl"), tok, 768, 460, "test", limit=6)
    assert [r["prompt_token_ids"] for r in call["requests"]] == [ds[i]["prompt"] for i in range(6)]
    assert {r["max_tokens"] for r in call["requests"]} == {768 - 460}
    assert call["params"]["temperature"] == 0.0 and call["params"]["stop_token_ids"] == [tok.eos_token_id, 151643]
    assert call["args"]["model"] == a.model_path and call["args"]["max_lora_rank"] == 8
    assert call["args"]["lora"] == os.path.join(a.save, "vllm_tmp", "adapter")
    assert preds == [tok.decode([100 + sum(r["prompt_token_ids"]) % 1000], skip_special_tokens=True)
                     for r in call["requests"]]
    assert refs == [[ds[i]["answer"]] for i in range(6)]
    assert "trigger" in metrics and not os.path.exists(os.path.join(a.save, "vllm_tmp"))


def test_migu_is_handed_to_vllm_whole(tmp_path, fake_vllm):
    from transformers import AutoTokenizer
    from cl_lora.engine import eval_task
    a = engine_args(tmp_path, method="migu", vllm_py=fake_vllm.python)
    os.makedirs(a.save)
    eval_task(a, tiny_qwen(0).eval(), AutoTokenizer.from_pretrained(MODEL_DIR), "cpu", 1, None)
    call = fake_vllm.calls()[0]
    assert call["args"]["model"] == os.path.join(a.save, "vllm_tmp", "model") and call["args"]["lora"] is None


def test_a_resume_under_the_other_backend_is_refused(tmp_path, fake_vllm, monkeypatch):
    from cl_lora import engine
    save = tmp_path / "run"
    argv = ["engine.py", "--cl-method", "inclora", "--data-root", need(TACRED), "--num-tasks", "10",
            "--model-path", MODEL_DIR, "--save", str(save)]
    monkeypatch.setattr(sys, "argv", argv + ["--gen-backend", "hf"])
    a = engine.parse_args()
    a.loss_group_size = a.batch_size                         # as main() sets it
    save.mkdir()
    (save / "run_manifest.json").write_text(json.dumps(engine.manifest_for(a)))
    monkeypatch.setattr(sys, "argv", argv + ["--gen-backend", "vllm", "--resume"])
    with pytest.raises(ValueError, match="resume manifest mismatch for gen_backend"):
        engine.main()


def test_a_vllm_run_without_vllm_is_refused_before_its_run_directory_exists(tmp_path, monkeypatch):
    from cl_lora import engine
    monkeypatch.setenv("VLLM_PY", str(tmp_path / "missing"))
    monkeypatch.setattr(sys, "argv", ["engine.py", "--cl-method", "inclora", "--data-root", need(TACRED),
                                      "--model-path", MODEL_DIR, "--save", str(tmp_path / "run"),
                                      "--gen-backend", "vllm"])
    with pytest.raises(RuntimeError, match="no vLLM environment"):
        engine.main()
    assert not (tmp_path / "run").exists()
