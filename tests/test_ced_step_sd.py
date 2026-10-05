"""Self-distillation in the grouped step: one sampling call for the physical batch and the
SD loss per group, against the 4257d86 per-micro-batch code, with scripted samples."""
import os
import random
import shutil
import subprocess
import sys
from argparse import Namespace

import pytest
import torch

import ced_losses
import reference_ced_loss as ref
from conftest import EngineShim, MODEL_DIR, REPO, ced_args, need
from data_utils.lm_datasets import LMTrainDataset
from test_ced_step import add_hooks, collate, lora_grads, pick_rows, with_args

SD_STATS = ("steps", "active", "rows", "truncated", "unparsed", "kept", "masked_rec", "masked_tok")
STREAMS = "data/ace_b10_perm0/streams.json"


@pytest.fixture(scope="module")
def sd_data(tmp_path_factory, tokenizer):
    """80 new-task rows and the 16 memory rows at the end of ACE task1, with SD teacher
    prompts, tokenized the way run_ced_v2.sh does it."""
    raw = need("data/ace_b10_perm0/1")
    work = tmp_path_factory.mktemp("sd")
    (work / "raw").mkdir()
    lines = open(f"{raw}/train.jsonl").read().splitlines()
    (work / "raw" / "train.jsonl").write_text("\n".join(lines[:80] + lines[-16:]) + "\n")
    for split in ("dev", "test"):
        shutil.copy(f"{raw}/{split}.jsonl", work / "raw" / f"{split}.jsonl")
    env = {**os.environ, "PYTHONPATH": REPO}
    subprocess.run([sys.executable, "tools/ced_sd_prompts.py", "--data-dir", str(work / "raw"),
                    "--out", str(work / "sd")], check=True, env=env)
    subprocess.run([sys.executable, "tools/process_data.py", "--data-dir", f"{work / 'sd'}/",
                    "--processed-data-dir", str(work / "proc"), "--model-path", MODEL_DIR,
                    "--data-process-workers", "2", "--max-prompt-length", "460", "--t-max-prompt-length", "1000",
                    "--dev-num", "1000", "--model-type", "qwen"], check=True, env=env)
    args = ced_args(ced_streams_file=need(STREAMS), ced_task_id=1, w_span_loss=2.0, type="sfkl", ced_sd=True)
    ds = LMTrainDataset(args, tokenizer, str(work / "proc" / "qwen") + "/", "train", -1, 1, random.Random(0))
    return args, ds


def scripted_generate(ds, truncated=()):
    """Stand-in for model.generate. Each prompt is answered with its own gold response; for
    rows in `truncated`, with that response cut before <|im_end|> and no stop token."""
    answers = {}
    for i in range(len(ds)):
        ids = ds[i]["input_ids"].astype(int)
        cut = int((ids == 4294967295).nonzero()[0][0])
        answers[tuple(ids[:cut].tolist())] = (i, ids[cut + 1:].tolist())

    def generate(input_ids=None, attention_mask=None, max_new_tokens=None, **kwargs):
        rows = []
        for prompt, mask in zip(input_ids, attention_mask):
            i, response = answers[tuple(prompt[mask.bool()].tolist())]
            tail = torch.zeros(max_new_tokens, dtype=torch.long)            # token 0 is no stop token
            if i in truncated:
                tail[:len(response) - 1] = torch.tensor(response[:-1])
            else:
                tail.fill_(151645)
                tail[:len(response)] = torch.tensor(response)
            rows.append(torch.cat([prompt, tail]))
        return Namespace(sequences=torch.stack(rows))
    return generate


def run_reference(args, ds, rows, engine, teacher, tokenizer, ema):
    """Backpropagates each group's loss (scaled by 1/groups) before the next group's
    sd_prepare, as the 4257d86 loop did per micro-step: sd_prepare swaps the LoRA weights in
    place, which would invalidate the saved tensors of a loss not yet backpropagated.
    Returns the mean loss (detached); the gradients are left in the parameters."""
    from ced_step import group_slices
    captured, capture = [], {"on": True}
    handles = add_hooks(engine.module, captured, capture)
    try:
        groups = group_slices(len(rows), args.loss_group_size)
        n_groups = args.batch_size // args.loss_group_size       # a full physical batch's groups
        losses = []
        for gs in groups:
            mb, nmb, gen, _, _ = collate(ds, rows[gs], dynamic=False)
            captured.clear()
            captured.append(None)
            out = ref.reference_micro_step(args, tokenizer, engine, teacher, mb, nmb, gen, "cpu", 1, ema,
                                           dict.fromkeys(SD_STATS, 0), captured, capture)
            (out["loss"] / n_groups).backward()
            losses.append(out["loss"].detach())
        return torch.stack(losses).sum() / n_groups
    finally:
        for h in handles:
            h.remove()


def run_new(args, ds, rows, engine, teacher, tokenizer, ema):
    from ced_step import ced_step_loss
    captured, capture = [None], {"on": True}
    handles = add_hooks(engine.module, captured, capture)
    try:
        mb, nmb, gen, _, _ = collate(ds, rows, dynamic=False)
        sd_batch, _ = ced_losses.sd_prepare(args, tokenizer, engine, ema, gen, nmb, "cpu")
        return ced_step_loss(args, engine, teacher, mb, nmb, captured, capture, sd_batch=sd_batch)
    finally:
        for h in handles:
            h.remove()


def check_sd(args, ds, rows, student, teacher, tokenizer, truncated=(), batch_size=None):
    """batch_size: the physical --batch-size the rows came from (default: all of them)."""
    args = with_args(args, batch_size=batch_size or len(rows))
    engine = EngineShim(student)
    engine.generate = scripted_generate(ds, truncated)
    ema = ced_losses.sd_ema_init(engine)
    student.train()
    student.zero_grad(set_to_none=True)
    expected = run_reference(args, ds, rows, engine, teacher, tokenizer, ema)      # backpropagated inside
    expected_grads = lora_grads(student)
    student.zero_grad(set_to_none=True)
    got = run_new(args, ds, rows, engine, teacher, tokenizer, ema)
    got.loss.backward()
    got_grads = lora_grads(student)
    torch.testing.assert_close(got.loss.detach(), expected.detach(), rtol=1e-5, atol=1e-7)
    for name, grad in expected_grads.items():
        torch.testing.assert_close(got_grads[name], grad, rtol=1e-4, atol=1e-7, msg=name)
    return got


def test_self_distillation_matches_the_reference(sd_data, student, teacher, tokenizer):
    args, ds = sd_data
    got = check_sd(args, ds, pick_rows(ds, 3, 5), student, teacher, tokenizer)
    assert got.sd_groups == 4


def test_a_group_whose_samples_are_all_truncated_gets_no_sd_term(sd_data, student, teacher, tokenizer):
    args, ds = sd_data
    rows = pick_rows(ds, 3, 5)
    got = check_sd(args, ds, rows, student, teacher, tokenizer, truncated=set(rows[0:2]))
    assert got.sd_groups == 3 and torch.isfinite(got.loss)


def test_self_distillation_on_a_short_last_batch_matches_the_reference(sd_data, student, teacher, tokenizer):
    args, ds = sd_data
    got = check_sd(args, ds, pick_rows(ds, 2, 3), student, teacher, tokenizer, batch_size=8)   # groups 2+2+1 of 4
    assert got.sd_groups == 3


def test_omission_mask_path_matches_the_reference(sd_data, student, teacher, tokenizer):
    args, ds = sd_data
    check_sd(with_args(args, ced_sd_omission_mask=True), ds, pick_rows(ds, 3, 5), student, teacher, tokenizer)


def test_sd_group_loss_is_none_when_every_token_is_masked():
    sd = {"label": torch.full((2, 3), -100), "t_logits": torch.zeros(2, 3, 5)}
    assert ced_losses.sd_group_loss(ced_args(), torch.zeros(2, 3, 5), sd, torch.tensor([True, False])) is None
