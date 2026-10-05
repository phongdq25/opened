import json
import random

import pytest
import torch
from torch.utils.data import DataLoader

from conftest import ced_args, need, tiny_qwen
from data_utils.lm_datasets import LMTrainDataset

ACE_TEST = "processed_data/ace_b10_perm0/0/qwen/"


def test_every_mode_is_the_historical_schedule():
    from ced_eval import EvalPlan, eval_plan
    for last in (False, True):
        for adaptive in (False, True):
            assert eval_plan("every", last, adaptive) == EvalPlan(dev=True, dev_generate=True, test=True)


def test_final_mode_generates_only_the_test_set_at_the_end():
    from ced_eval import EvalPlan, eval_plan
    assert eval_plan("final", False, False) == EvalPlan(dev=False, dev_generate=False, test=False)
    assert eval_plan("final", True, False) == EvalPlan(dev=False, dev_generate=False, test=True)


def test_final_mode_keeps_a_loss_only_dev_pass_for_adaptive_methods():
    from ced_eval import EvalPlan, eval_plan
    assert eval_plan("final", False, True) == EvalPlan(dev=True, dev_generate=False, test=False)


def test_final_test_runs_once_when_the_last_evaluation_point_never_fired():
    from ced_eval import final_test_missing
    assert final_test_missing(ced_args(eval_gen_mode="final"), done=False)
    assert not final_test_missing(ced_args(eval_gen_mode="final"), done=True)
    assert not final_test_missing(ced_args(eval_gen_mode="every"), done=False)


@pytest.fixture(scope="module")
def ace_test(tokenizer):
    args = ced_args(max_length=470)                # 10 new tokens: fast generation on CPU
    return LMTrainDataset(args, tokenizer, need(ACE_TEST), "test", 40, 1, random.Random(0))


def old_style_loss(model, ds, batch):
    """The loss pass of evaluate() at 4257d86: full logits, batches of `batch` rows."""
    loss_func = torch.nn.CrossEntropyLoss()
    total, steps = 0.0, 0
    with torch.no_grad():
        for mb, nmb, _, _, _ in DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=ds.collate):
            logits = model(**mb).logits
            total += loss_func(logits.view(-1, logits.shape[-1]), nmb["label"].view(-1)).item()
            steps += 1
    return total / steps


def test_loss_pass_matches_the_old_loop(tmp_path, tokenizer, ace_test):
    from ced_eval import evaluate
    model = tiny_qwen(0).eval()
    args = ced_args(max_length=470, save=str(tmp_path), eval_loss_batch_size=8, eval_batch_size=16)
    got = evaluate(args, tokenizer, model, ace_test, "dev", 0, "cpu", generate=False)
    assert abs(got - old_style_loss(model, ace_test, 8)) < 1e-6
    assert not (tmp_path / "eval").exists()
    assert open(tmp_path / "log.txt").read().strip().endswith("| {}")


def test_generation_writes_one_answer_per_row(tmp_path, tokenizer, ace_test):
    from ced_eval import evaluate
    model = tiny_qwen(0).eval()
    args = ced_args(max_length=470, save=str(tmp_path), eval_batch_size=16, strict_generation=True)
    evaluate(args, tokenizer, model, ace_test, "test", 3, "cpu")
    answers = open(tmp_path / "eval" / "3" / "answers.jsonl").read().splitlines()
    assert len(answers) == len(ace_test)
    assert all("text" in json.loads(a) for a in answers)
    assert "'trigger'" in open(tmp_path / "log.txt").read()
