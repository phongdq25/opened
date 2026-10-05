"""cl_lora/engine.py's grouped step against the verbatim 4257d86 micro-step."""
import copy
from argparse import Namespace

import pytest
import torch

from conftest import need, tiny_qwen
from reference_engine_step import reference_engine_micro_step

DATA = "data/fewrel_perm0/1/train.jsonl"


def engine_args(method, group, phys, accum):
    return Namespace(cl_method=method, rank=4, alpha=16, dropout=0.0, cl_reg=0.5, batch_size=phys,
                     grad_accum=accum, loss_group_size=group, max_length=768, max_prompt_length=460)


def build(method, tokenizer):
    from peft import get_peft_model
    from cl_lora import treelora as tree_mod
    from cl_lora.engine import JsonlED, lora_config
    from cl_lora.multi_adapter import CLLoRAManager
    a = engine_args(method, 2, 8, 1)
    model = get_peft_model(tiny_qwen(0), lora_config(a), adapter_name="task0")
    mgr = CLLoRAManager(model, orth_lambda=0.5 if method in ("olora", "gainlora_o") else 0.0)
    mgr.register_task0("task0")
    tree = None
    if method == "tree":
        tree = tree_mod.KDLoRATree(3, reg=0.5)
        tree.new_epoch_init(4)
        for _ in range(4):
            tree.step()
            tree.insert_grad(tree_mod.signature_from_model(model, "task0"))
        tree.end_task(0)
    mgr.start_new_task(1, lora_config(a))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.02)
    ds = JsonlED(need(DATA), tokenizer, 768, 460, "train", limit=16)
    return model, mgr, tree, ds


@pytest.mark.parametrize("method", ["inclora", "olora", "tree"])
def test_grouped_step_matches_the_reference(method, tokenizer):
    from ced_step import group_slices
    from cl_lora.engine import grouped_step_loss
    model, mgr, tree, ds = build(method, tokenizer)
    rows = [ds[i] for i in range(8)]
    model.train()

    tree_ref = copy.deepcopy(tree)
    if tree_ref is not None:
        tree_ref.new_epoch_init(4)
    torch.manual_seed(7)
    model.zero_grad(set_to_none=True)
    a_ref = engine_args(method, 2, 2, 4)              # 4 micro-steps of 2 rows: one update
    for gs in group_slices(8, 2):
        mb, nmb = ds.collate_train(rows[gs])
        reference_engine_micro_step(a_ref, model, mb, nmb, "cpu", 1, mgr, tree_ref, None, "task1")
    expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    tree_new = copy.deepcopy(tree)
    if tree_new is not None:
        tree_new.new_epoch_init(4)
    torch.manual_seed(7)
    model.zero_grad(set_to_none=True)
    a_new = engine_args(method, 2, 8, 1)              # one physical step of 8 rows, loss groups of 2
    mb, nmb = ds.collate_train(rows)
    loss = grouped_step_loss(a_new, model, mb, nmb["label"], 1, mgr, tree_new, "task1", "cpu")
    (loss / a_new.grad_accum).backward()
    got = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    assert got.keys() == expected.keys()
    for name, grad in expected.items():
        torch.testing.assert_close(got[name], grad, rtol=1e-4, atol=1e-7, msg=name)
    if tree is not None:
        torch.testing.assert_close(tree_new.sim, tree_ref.sim)
        torch.testing.assert_close(tree_new.num_of_selected, tree_ref.num_of_selected)


@pytest.mark.parametrize("method", ["inclora", "tree"])
def test_grouped_step_on_a_short_last_batch_matches_the_reference(method, tokenizer):
    """5 leftover rows of an 8-row physical batch: 4257d86 ran micro-steps of 2, 2 and 1 rows,
    each loss divided by the accumulation; the grouped step must weight them the same."""
    from ced_step import group_slices
    from cl_lora.engine import grouped_step_loss
    model, mgr, tree, ds = build(method, tokenizer)
    rows = [ds[i] for i in range(5)]
    model.train()

    tree_ref = copy.deepcopy(tree)
    if tree_ref is not None:
        tree_ref.new_epoch_init(4)
    torch.manual_seed(7)
    model.zero_grad(set_to_none=True)
    a_ref = engine_args(method, 2, 2, 4)
    for gs in group_slices(5, 2):
        mb, nmb = ds.collate_train(rows[gs])
        reference_engine_micro_step(a_ref, model, mb, nmb, "cpu", 1, mgr, tree_ref, None, "task1")
    expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    tree_new = copy.deepcopy(tree)
    if tree_new is not None:
        tree_new.new_epoch_init(4)
    torch.manual_seed(7)
    model.zero_grad(set_to_none=True)
    a_new = engine_args(method, 2, 8, 1)
    mb, nmb = ds.collate_train(rows)
    (grouped_step_loss(a_new, model, mb, nmb["label"], 1, mgr, tree_new, "task1", "cpu") / a_new.grad_accum).backward()
    got = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    assert got.keys() == expected.keys()
    for name, grad in expected.items():
        torch.testing.assert_close(got[name], grad, rtol=1e-4, atol=1e-7, msg=name)
    if tree is not None:
        torch.testing.assert_close(tree_new.sim, tree_ref.sim)
        torch.testing.assert_close(tree_new.num_of_selected, tree_ref.num_of_selected)


def test_migu_refuses_a_physical_batch_above_its_micro_batch(monkeypatch):
    from cl_lora import engine
    monkeypatch.setattr("sys.argv", ["engine.py", "--cl-method", "migu", "--data-root", "unused",
                                     "--save", "unused", "--batch-size", "16", "--loss-group-size", "2"])
    with pytest.raises(ValueError, match="MIGU"):
        engine.main()
