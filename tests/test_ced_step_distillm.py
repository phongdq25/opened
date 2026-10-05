"""DistiLLM/AMiD student generation in the grouped step against 4257d86, with scripted
generations and seeded draws."""
import random

import numpy as np
import torch

import ced_losses
import reference_ced_loss as ref
from conftest import EngineShim
from distillm import ReplayBuffer, SampleGenerator
from test_ced_step import FEWREL, add_hooks, collate, load, lora_grads
from test_ced_step_sd import scripted_generate

ADAPTIVE = 0.9      # draws below it generate or sample
CAPACITY = 6        # a full buffer: draws in [0.45, 0.9) then sample instead of generating


def mixed_seed():
    """A seed whose four draws give both a 'generate' (< 0.45) and a 'sample' ([0.45, 0.9)) group."""
    for seed in range(1000):
        draws = np.random.RandomState(seed).uniform(0, 1, 4)
        if (draws < 0.45).any() and ((draws >= 0.45) & (draws < ADAPTIVE)).any():
            return seed
    raise AssertionError("no seed found")


def filled_buffer(args, ds, sampler, engine, rows):
    buffer = ReplayBuffer(args)
    _, nmb, gen, _, _ = collate(ds, rows, dynamic=False)
    g_model, g_meta, g_gen = ced_losses.generate_replay_rows(args, sampler, engine, gen, nmb,
                                                             torch.arange(len(rows)))
    buffer.move_to_memory(g_model, g_meta, g_gen)
    return buffer


def test_distillm_generation_matches_the_reference(tokenizer, student, teacher):
    from ced_step import ced_step_loss, distillm_replace_groups, group_slices
    args, ds = load(tokenizer, FEWREL, type="adaptive-srkl", student_gen=True, capacity=CAPACITY,
                    replay_ratio="constant")
    engine = EngineShim(student)
    engine.generate = scripted_generate(ds)
    sampler = SampleGenerator(args, tokenizer)
    replay = [i for i, f in enumerate(ds.replay_flags) if f]
    new = [i for i, f in enumerate(ds.replay_flags) if not f]
    rows = [replay[0], new[0], replay[1], new[1], replay[2], new[2], replay[3], new[3]]
    seed = mixed_seed()

    def reference():
        buffer = filled_buffer(args, ds, sampler, engine, replay[4:8] + new[4:6])
        np.random.seed(seed)
        random.seed(seed)
        captured, capture = [], {"on": True}
        handles = add_hooks(student, captured, capture)
        losses = []
        for gs in group_slices(len(rows), args.loss_group_size):
            mb, nmb, gen, _, _ = collate(ds, rows[gs], dynamic=False)
            mb, nmb = ref.reference_distillm(args, engine, sampler, buffer, mb, nmb, gen, ADAPTIVE, 1, "cpu")
            captured.clear()
            captured.append(None)
            losses.append(ref.reference_micro_step(args, tokenizer, engine, teacher, mb, nmb, gen, "cpu", 1,
                                                   None, {}, captured, capture)["loss"])
        for h in handles:
            h.remove()
        return torch.stack(losses).mean(), buffer

    def grouped():
        buffer = filled_buffer(args, ds, sampler, engine, replay[4:8] + new[4:6])
        np.random.seed(seed)
        random.seed(seed)
        captured, capture = [None], {"on": True}
        handles = add_hooks(student, captured, capture)
        mb, nmb, gen, _, _ = collate(ds, rows, dynamic=False)
        mb, nmb = distillm_replace_groups(args, group_slices(len(rows), args.loss_group_size), engine, sampler,
                                          buffer, mb, nmb, gen, ADAPTIVE * 0.5, ADAPTIVE, "cpu")
        out = ced_step_loss(args, engine, teacher, mb, nmb, captured, capture)
        for h in handles:
            h.remove()
        return out.loss, buffer

    student.train()
    student.zero_grad(set_to_none=True)
    expected, expected_buffer = reference()
    expected.backward()
    expected_grads = lora_grads(student)
    student.zero_grad(set_to_none=True)
    got, got_buffer = grouped()
    got.backward()
    got_grads = lora_grads(student)

    torch.testing.assert_close(got.detach(), expected.detach(), rtol=1e-5, atol=1e-7)
    for name, grad in expected_grads.items():
        torch.testing.assert_close(got_grads[name], grad, rtol=1e-4, atol=1e-7, msg=name)
    assert len(got_buffer) == len(expected_buffer)
    for a, b in zip(expected_buffer.replay_memory, got_buffer.replay_memory):
        assert torch.equal(a["model_data"]["input_ids"], b["model_data"]["input_ids"])
