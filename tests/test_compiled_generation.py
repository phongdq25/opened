"""--compile-generation: a static KV cache for the generate() calls inside training steps."""
import inspect
from argparse import Namespace

import pytest
import torch

from conftest import ced_args


class Stop(Exception):
    pass


class RecordingModel(torch.nn.Module):
    def __init__(self, seen):
        super().__init__()
        self.seen = seen

    def generate(self, **kwargs):
        self.seen.update(kwargs)
        raise Stop


@pytest.mark.parametrize("strict,compiled,expected", [
    (False, False, {}),
    (False, True, {"cache_implementation": "static"}),
    (True, False, {"use_model_defaults": False}),
])
def test_train_generation_kwargs_add_the_static_cache_only_when_compiling(strict, compiled, expected):
    from gen_config import train_generation_kwargs
    assert train_generation_kwargs(Namespace(strict_generation=strict, compile_generation=compiled)) == expected


def test_strict_and_compiled_generation_combine():
    from gen_config import train_generation_kwargs
    assert train_generation_kwargs(Namespace(strict_generation=True, compile_generation=True)) == \
        {"use_model_defaults": False, "cache_implementation": "static"}


@pytest.mark.parametrize("compiled", [False, True])
def test_self_distillation_samples_with_a_static_cache_when_compiling(tokenizer, compiled):
    import ced_losses
    seen = {}
    gen_data = {"input_ids": torch.full((1, 4), 7), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    with pytest.raises(Stop):
        ced_losses.sd_prepare(ced_args(compile_generation=compiled), tokenizer, RecordingModel(seen), {},
                              gen_data, {"t_prompt_ids": []}, "cpu")
    assert seen.get("cache_implementation") == ("static" if compiled else None)


@pytest.mark.parametrize("compiled", [False, True])
def test_student_generation_stays_uncompiled(compiled):
    """DistiLLM/AMiD generate for a varying number of rows per step, and every new batch size
    recompiles: measured slower (plan 2026-10-06-compiled-sd-generation, Task 3), so it keeps
    generation_kwargs whatever --compile-generation says."""
    from distillm.sampler import SampleGenerator
    args = ced_args(compile_generation=compiled, max_length=20, max_prompt_length=10)
    sampler = SampleGenerator(args, Namespace(pad_token_id=0, eos_token_id=1))
    seen = {}
    gen_data = {"input_ids": torch.full((1, 10), 7), "attention_mask": torch.ones(1, 10, dtype=torch.long)}
    with pytest.raises(Stop):
        sampler.run_sample(RecordingModel(seen), gen_data)
    assert "cache_implementation" not in seen


def test_evaluation_keeps_its_own_generation_kwargs():
    import ced_eval
    source = inspect.getsource(ced_eval)
    assert "train_generation_kwargs" not in source and "**generation_kwargs(args)" in source
