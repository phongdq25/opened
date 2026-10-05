from argparse import Namespace

import pytest
import torch
from transformers import GenerationConfig

from conftest import ced_args, tiny_qwen

EVAL = dict(do_sample=False, temperature=0.5, top_p=0.95, top_k=0)   # evaluate() under --greedy 1
SD = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0)       # sd_prepare() with SDFT defaults


def resolved(cfg, strict):
    from gen_config import generation_kwargs
    g, _ = tiny_qwen(0)._prepare_generation_config(GenerationConfig(**cfg),
                                                   **generation_kwargs(Namespace(strict_generation=strict)))
    return g


def test_default_keeps_todays_resolution():
    assert resolved(EVAL, False).do_sample is True                   # the "greedy" eval samples
    g = resolved(SD, False)
    assert (g.temperature, g.top_p) == (0.6, 0.95)


def test_strict_uses_every_config_as_written():
    assert resolved(EVAL, True).do_sample is False
    g = resolved(SD, True)
    assert (g.temperature, g.top_p) == (1.0, 1.0)


class Stop(Exception):
    pass


class RecordingModel(torch.nn.Module):
    def __init__(self, seen):
        super().__init__()
        self.seen = seen

    def generate(self, **kwargs):
        self.seen.update(kwargs)
        raise Stop


def test_sd_prepare_passes_the_switch(tokenizer):
    import ced_losses
    seen = {}
    gen_data = {"input_ids": torch.full((1, 4), 7), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    with pytest.raises(Stop):
        ced_losses.sd_prepare(ced_args(strict_generation=True), tokenizer, RecordingModel(seen), {},
                              gen_data, {"t_prompt_ids": []}, "cpu")
    assert seen["use_model_defaults"] is False


def test_student_generation_passes_the_switch():
    from distillm.sampler import SampleGenerator
    args = ced_args(strict_generation=True, max_length=20, max_prompt_length=10)
    sampler = SampleGenerator(args, Namespace(pad_token_id=0, eos_token_id=1))
    seen = {}
    gen_data = {"input_ids": torch.ones(1, 10, dtype=torch.long), "attention_mask": torch.ones(1, 10, dtype=torch.long)}
    with pytest.raises(Stop):
        sampler.run_sample(RecordingModel(seen), gen_data)
    assert seen["use_model_defaults"] is False
