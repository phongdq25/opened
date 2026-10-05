"""Shared test fixtures. The tests run on the box that holds the data and the model, on
CPU in fp32 (see docs/superpowers/plans/2026-10-05-h200-throughput.md)."""
import json
import os
import sys
from argparse import Namespace

import pytest
import torch
import torch.distributed as dist

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
MODEL_DIR = os.path.join(REPO, "models", "Qwen3-0.6B")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def need(path):
    if not os.path.exists(path):
        pytest.skip(f"needs {path} (run on the box that holds the data)")
    return path


@pytest.fixture(scope="session", autouse=True)
def single_process_group():
    """LMTrainDataset and evaluate() ask torch.distributed for the rank and world size."""
    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29591", rank=0, world_size=1)
    yield


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(need(MODEL_DIR), padding_side="right")
    tok.eos_token_id = 151645          # what utils.get_tokenizer sets for qwen
    tok.pad_token_id = tok.eos_token_id
    tok.pad_token = tok.eos_token
    return tok


def tiny_qwen(seed, layers=4):
    """A randomly initialized 4-layer Qwen3 with the real vocabulary and generation defaults."""
    from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=2048, tie_word_embeddings=True, attn_implementation="sdpa")
    model = Qwen3ForCausalLM(cfg).float()
    model.generation_config = GenerationConfig.from_pretrained(need(MODEL_DIR))
    return model


@pytest.fixture
def student():
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(tiny_qwen(0), LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=16,
                                                     lora_dropout=0.0, target_modules=LORA_TARGETS))
    # LoRA B starts at zero, which would leave half the LoRA gradients trivially equal
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.02)
    return model


@pytest.fixture
def teacher():
    model = tiny_qwen(1)
    model.eval()
    return model


class EngineShim(torch.nn.Module):
    """The parts of DeepSpeedEngine that the training step touches."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.module.generate(*args, **kwargs)

    @property
    def base_model(self):
        return self.module.base_model


def ced_args(**overrides):
    """The argparse namespace ced_finetune.py works with, with test-sized defaults."""
    base = dict(
        type="rkl", kd_ratio=0.9, skew_alpha=0.1, w_span_loss=0.0, span_metric="cosine",
        student_layer_mapping=[2, 3, 4], teacher_layer_mapping=[2, 3, 4],
        ced_replay_mode="ce_kd", ced_kd_scope="replay", ced_kd_ratio_new=0.0, ced_kd_new_cap=0.3,
        ced_sd=False, ced_sd_weight=1.0, ced_sd_ema_mu=0.99, ced_sd_temperature=1.0, ced_sd_div="fkl",
        ced_sd_omission_mask=False, ced_sd_mix="sum", ced_sd_probe=0, ced_sd_top_p=1.0,
        ced_sd_skip_unparsed=False, ced_sd_warmup=0.0, ced_streams_file=None, ced_task_id=None,
        ced_smoke_rows=0, lm_data_dir=None, lm_coef=1.0, model_parallel=False, model_type="qwen",
        max_length=768, max_prompt_length=460, t_max_length=1024, total_iters=100, seed=42,
        student_gen=False, amid_alpha=0.5, amid_lam=0.5, amid_div_name="fkl", amid_div_order="pr",
        capacity=1000, mixed_alpha=0.5, replay_ratio="decreasing", init_threshold=0.0, loss_eps=0.1,
        gen_do_sample=True, gen_top_p=1.0, gen_temperature=1.0, gen_num_beams=1,
        batch_size=2, loss_group_size=2, dynamic_pad=False, dynamic_pad_effective=False,
        strict_generation=False, eval_batch_size=32, eval_loss_batch_size=32, eval_gen=True,
        eval_gen_mode="every", do_sample=False, top_p=0.95, top_k=0, temperature=0.5,
        repetition_penalty=None, num_workers=0, save=None, gen_backend="hf", vllm_py=None, model_path=None,
        compile_generation=False,
    )
    base.update(overrides)
    return Namespace(**base)


FAKE_VLLM = os.path.join(REPO, "tests", "fake_vllm.py")


class FakeVLLM:
    """What the fake_vllm fixture hands a test: the VLLM_PY to pass on, and the recorded calls."""

    def __init__(self, python, log):
        self.python, self.log = python, log

    def calls(self):
        if not os.path.exists(self.log):
            return []
        return [json.loads(line) for line in open(self.log)]


@pytest.fixture
def fake_vllm(monkeypatch, tmp_path):
    """gen_backend runs tests/fake_vllm.py in place of tools/vllm_generate.py, through a VLLM_PY
    that tells find_vllm_python() it holds vLLM 0.27.1."""
    import gen_backend
    python = tmp_path / "vllm_python"
    python.write_text('#!/bin/bash\nif [ "$1" = "-c" ]; then echo "VLLM_VERSION 0.27.1"; exit 0; fi\n'
                      f'exec {sys.executable} "$@"\n')
    python.chmod(0o755)
    monkeypatch.setattr(gen_backend, "VLLM_SCRIPT", FAKE_VLLM)
    monkeypatch.setenv("VLLM_PY", str(python))
    monkeypatch.setenv("FAKE_VLLM_LOG", str(tmp_path / "fake_vllm.log"))
    return FakeVLLM(str(python), str(tmp_path / "fake_vllm.log"))
