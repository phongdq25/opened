"""The grouped training step (ced_step.py) against the verbatim 4257d86 loss
(tests/reference_ced_loss.py): same loss, same LoRA gradients."""
import random

import pytest
import torch

import reference_ced_loss as ref
from conftest import EngineShim, ced_args, need
from data_utils.lm_datasets import LMTrainDataset

FEWREL = ("processed_data/fewrel_perm0/1/qwen/", "data/fewrel_perm0/streams.json")
ACE = ("processed_data/ace_b10_perm0/1/qwen/", "data/ace_b10_perm0/streams.json")
ACE_ORACLE = ("processed_data/ace_oracle_b10_perm0/1/qwen/", "data/ace_oracle_b10_perm0/streams.json")
KD_TYPES = ["kd", "rkl", "sfkl", "srkl", "csd", "amid", "no"]


def load(tokenizer, source, **overrides):
    path, streams = source
    args = ced_args(ced_streams_file=need(streams), ced_task_id=1, **overrides)
    return args, LMTrainDataset(args, tokenizer, need(path), "train", -1, 1, random.Random(0))


def with_args(args, **overrides):
    return ced_args(**{**vars(args), **overrides})


def pick_rows(ds, n_replay, n_new, n_pl=0, seed=0):
    rnd = random.Random(seed)
    replay = [i for i, f in enumerate(ds.replay_flags) if f]
    pl = [i for i, f in enumerate(ds.replay_flags) if not f and ds.old_span_offsets[i]]
    new = [i for i, f in enumerate(ds.replay_flags) if not f and not ds.old_span_offsets[i]]
    rows = rnd.sample(replay, n_replay) + rnd.sample(pl, n_pl) + rnd.sample(new, n_new)
    rnd.shuffle(rows)
    return rows


def collate(ds, rows, dynamic):
    saved = ds.args.dynamic_pad_effective
    ds.args.dynamic_pad_effective = dynamic
    try:
        return ds.collate([ds[i] for i in rows])
    finally:
        ds.args.dynamic_pad_effective = saved


def add_hooks(model, captured, capture):
    """The span-loss hooks ced_finetune.py registers on the decoder layers."""
    def hook(module, inputs, output):
        if module.training and capture["on"]:
            captured.append(output[0] if isinstance(output, tuple) else output)
    return [layer.register_forward_hook(hook) for layer in model.base_model.model.model.layers]


def lora_grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None}


def reference_loss(args, ds, rows, student, teacher, tokenizer, mutate=None, n_groups=None):
    """4257d86: each group collated on its own at full width and scored as one micro-batch.
    Returns the sum of the micro-batch losses over n_groups (default: their mean), the weight
    each micro-batch had in 4257d86's accumulation relative to one physical step."""
    from ced_step import group_slices
    engine, captured, capture = EngineShim(student), [], {"on": True}
    handles = add_hooks(student, captured, capture)
    try:
        losses = []
        for gs in group_slices(len(rows), args.loss_group_size):
            mb, nmb, gen, _, _ = collate(ds, rows[gs], dynamic=False)
            if mutate:
                mutate(nmb, rows[gs])
            captured.clear()
            captured.append(None)
            out = ref.reference_micro_step(args, tokenizer, engine, teacher, mb, nmb, gen, "cpu", 1,
                                           None, {}, captured, capture)
            losses.append(out["loss"])
        losses = torch.stack(losses)
        return losses.mean() if n_groups is None else losses.sum() / n_groups
    finally:
        for h in handles:
            h.remove()


def new_loss(args, ds, rows, student, teacher, dynamic=False, mutate=None):
    from ced_step import ced_step_loss
    engine, captured, capture = EngineShim(student), [None], {"on": True}
    handles = add_hooks(student, captured, capture)
    try:
        mb, nmb, _, _, _ = collate(ds, rows, dynamic)
        if mutate:
            mutate(nmb, rows)
        return ced_step_loss(args, engine, teacher, mb, nmb, captured, capture)
    finally:
        for h in handles:
            h.remove()


def check_equivalent(args, ds, rows, student, teacher, tokenizer, dynamic=False, mutate=None, batch_size=None):
    """batch_size: the physical --batch-size the rows came from (default: all of them, a full batch)."""
    args = with_args(args, batch_size=batch_size or len(rows))
    student.train()
    student.zero_grad(set_to_none=True)
    expected = reference_loss(args, ds, rows, student, teacher, tokenizer, mutate,
                              n_groups=args.batch_size // args.loss_group_size)
    expected.backward()
    expected_grads = lora_grads(student)
    student.zero_grad(set_to_none=True)
    got = new_loss(args, ds, rows, student, teacher, dynamic, mutate)
    got.loss.backward()
    got_grads = lora_grads(student)
    torch.testing.assert_close(got.loss.detach(), expected.detach(), rtol=1e-5, atol=1e-7)
    assert got_grads.keys() == expected_grads.keys()
    for name, grad in expected_grads.items():
        torch.testing.assert_close(got_grads[name], grad, rtol=1e-4, atol=1e-7, msg=name)
    return got


@pytest.fixture(scope="module")
def fewrel(tokenizer):
    return load(tokenizer, FEWREL)


@pytest.fixture(scope="module")
def ace(tokenizer):
    return load(tokenizer, ACE, w_span_loss=2.0, type="sfkl")


def test_group_slices_cover_the_batch_in_order():
    from ced_step import group_slices
    assert group_slices(8, 2) == [slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 8)]
    # the last batch of an epoch: like 4257d86's last micro-batch, its last group can be short
    assert group_slices(7, 2) == [slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 7)]
    assert group_slices(1, 2) == [slice(0, 1)]
    with pytest.raises(ValueError):
        group_slices(8, 0)


def test_a_short_last_batch_keeps_each_micro_batch_weight(fewrel, student, teacher, tokenizer):
    """The last batch of an epoch holds the leftover rows, maybe an odd number. 4257d86 trained
    them as short micro-batches weighted like full ones (1/accumulation each), so the grouped
    step divides its group losses by the full batch's group count, not by its own."""
    args, ds = fewrel
    check_equivalent(args, ds, pick_rows(ds, 2, 3), student, teacher, tokenizer, batch_size=8)  # groups 2+2+1 of 4


def test_a_split_smaller_than_one_update_is_refused_up_front():
    from ced_step import updates_per_epoch
    assert updates_per_epoch(1280, 16, 1, 2) == 40
    assert updates_per_epoch(891, 8, 1, 4) == 27          # int(27.84), as 4257d86 computed it
    with pytest.raises(ValueError, match="fewer rows than one update"):
        updates_per_epoch(20, 8, 1, 4)


def test_both_trainers_count_updates_through_the_guard():
    import inspect

    import ced_finetune
    import finetune
    for trainer in (ced_finetune, finetune):
        assert "updates_per_epoch(" in inspect.getsource(trainer.main)


def trainer_counts(physical_accumulation, logical_accumulation, total_iters):
    """The trainers' step bookkeeping (test_both_trainers_count_steps_this_way pins it to their
    source): the updates they apply, and the global_step their log/save/eval checks see after each."""
    from ced_step import first_global_step
    step, global_step = 1, first_global_step(physical_accumulation, logical_accumulation)
    micro_steps, seen = 0, []
    while True:
        micro_steps += 1                       # DeepSpeed applies an update every accumulation micro-steps
        if step % physical_accumulation == 0:
            seen.append(global_step)
        step += 1
        if step % physical_accumulation == 0:
            global_step += 1
        if global_step > total_iters:
            break
    return micro_steps // physical_accumulation, seen


@pytest.mark.parametrize("total_iters", [3, 10])
def test_every_physical_batch_trains_logs_and_stops_like_the_run_it_splits(total_iters):
    # --bs 2 --acc 16 had 16 micro-steps per update; physical batches of 8, 16 and 32 rows have
    # 4, 2 and 1. The trainers never apply a run's last update, and neither may 32 rows, or Ours
    # at PHYS_BS=32 would train one update more than the same run at any other batch
    original = trainer_counts(16, 16, total_iters)
    assert original == (total_iters - 1, list(range(2, total_iters + 1)))
    for physical in (4, 2, 1):
        assert trainer_counts(physical, 16, total_iters) == original


def test_a_run_with_one_micro_step_per_update_still_applies_every_update():
    assert trainer_counts(1, 1, 10) == (10, list(range(1, 11)))       # --acc 1, as it always was


def test_both_trainers_count_steps_this_way():
    import inspect

    import ced_finetune
    import finetune
    for trainer in (ced_finetune, finetune):
        source = " ".join(inspect.getsource(trainer.finetune).split())
        assert "step, global_step = 1, first_global_step(args.gradient_accumulation_steps," in source
        assert ("step += 1 if step % args.gradient_accumulation_steps == 0: global_step += 1 "
                "if global_step > args.total_iters: break") in source


@pytest.mark.parametrize("kd_type", KD_TYPES)
def test_token_kd_matches_the_reference(kd_type, fewrel, student, teacher, tokenizer):
    args, ds = fewrel
    check_equivalent(with_args(args, type=kd_type, amid_div_name="ab"), ds, pick_rows(ds, 3, 5),
                     student, teacher, tokenizer)


@pytest.mark.parametrize("group", [1, 4, 8])
def test_any_group_size_matches_the_reference(group, fewrel, student, teacher, tokenizer):
    args, ds = fewrel
    check_equivalent(with_args(args, loss_group_size=group), ds, pick_rows(ds, 2, 6), student, teacher, tokenizer)


def test_dynamic_padding_matches_the_reference(fewrel, student, teacher, tokenizer):
    args, ds = fewrel
    check_equivalent(args, ds, pick_rows(ds, 3, 5), student, teacher, tokenizer, dynamic=True)


def test_kd_only_replay_mode_matches_the_reference(fewrel, student, teacher, tokenizer):
    args, ds = fewrel
    check_equivalent(with_args(args, ced_replay_mode="kd_only"), ds, pick_rows(ds, 3, 5), student, teacher,
                     tokenizer)


def test_lwf_matches_the_reference(fewrel, student, teacher, tokenizer):
    args, ds = fewrel
    check_equivalent(with_args(args, ced_kd_ratio_new=0.2), ds, pick_rows(ds, 3, 5), student, teacher, tokenizer)


def test_a_group_without_labels_matches_the_reference(fewrel, student, teacher, tokenizer):
    args, ds = fewrel
    rows = pick_rows(ds, 3, 5)
    blank = set(rows[2:4])                        # every row of the second group

    def mutate(no_model_batch, batch_rows):
        for k, row in enumerate(batch_rows):
            if row in blank:
                no_model_batch["label"][k] = -100

    got = check_equivalent(args, ds, rows, student, teacher, tokenizer, mutate=mutate)
    assert torch.isfinite(got.loss)


@pytest.mark.parametrize("metric", ["cosine", "cka"])
def test_span_loss_matches_the_reference(metric, ace, student, teacher, tokenizer):
    args, ds = ace
    check_equivalent(with_args(args, span_metric=metric), ds, pick_rows(ds, 3, 5), student, teacher, tokenizer)


def test_gradient_checkpointing_keeps_the_span_loss_and_its_gradients(ace, student, teacher):
    # --gradient-checkpointing recomputes each decoder layer in the backward pass, and the span-loss
    # hooks capture those layers' outputs: a checkpointed Ours step must give the same loss and grads
    from utils import enable_gradient_checkpointing
    args, ds = ace
    rows = pick_rows(ds, 3, 5)
    args = with_args(args, batch_size=len(rows))
    student.enable_input_require_grads()            # what utils.get_model does for every LoRA model
    student.train()
    student.zero_grad(set_to_none=True)
    plain = new_loss(args, ds, rows, student, teacher)
    plain.loss.backward()
    plain_grads = lora_grads(student)
    enable_gradient_checkpointing(student)
    student.zero_grad(set_to_none=True)
    calls = []
    counter = student.base_model.model.model.layers[0].register_forward_hook(lambda *_: calls.append(1))
    ckpt = new_loss(args, ds, rows, student, teacher)
    ckpt.loss.backward()
    counter.remove()
    ckpt_grads = lora_grads(student)
    assert len(calls) == 2                          # the forward, then its recomputation in the backward
    torch.testing.assert_close(ckpt.loss.detach(), plain.loss.detach())
    assert ckpt_grads.keys() == plain_grads.keys()
    for name, grad in plain_grads.items():
        torch.testing.assert_close(ckpt_grads[name], grad, msg=name)


def test_pseudo_label_kd_scope_matches_the_reference(tokenizer, student, teacher):
    # the oracle split keeps earlier types' events in new-task rows: rows shaped like pseudo-labelled ones
    args, ds = load(tokenizer, ACE_ORACLE, w_span_loss=2.0, type="sfkl", ced_kd_scope="pl")
    check_equivalent(args, ds, pick_rows(ds, 2, 3, n_pl=3), student, teacher, tokenizer)


def test_grouped_ce_matches_finetune_cross_entropy(fewrel, student):
    from ced_step import group_slices, grouped_ce_loss
    args, ds = fewrel
    rows = pick_rows(ds, 2, 6)
    mb, nmb, _, _, _ = collate(ds, rows, dynamic=True)
    got = grouped_ce_loss(student, mb, nmb["label"], 2, n_groups=4)
    loss_func, expected = torch.nn.CrossEntropyLoss(), []
    for gs in group_slices(len(rows), 2):
        gmb, gnmb, _, _, _ = collate(ds, rows[gs], dynamic=False)
        logits = student(**gmb, use_cache=False).logits
        expected.append(loss_func(logits.float().view(-1, logits.shape[-1]), gnmb["label"].view(-1)))
    torch.testing.assert_close(got, torch.stack(expected).mean(), rtol=1e-5, atol=1e-7)


def test_grouped_ce_on_a_short_last_batch_keeps_each_micro_batch_weight(fewrel, student):
    from ced_step import group_slices, grouped_ce_loss
    args, ds = fewrel
    rows = pick_rows(ds, 2, 3)                      # the last 5 rows of an epoch, physical batch 8
    mb, nmb, _, _, _ = collate(ds, rows, dynamic=True)
    got = grouped_ce_loss(student, mb, nmb["label"], 2, n_groups=4)
    loss_func, expected = torch.nn.CrossEntropyLoss(), []
    for gs in group_slices(len(rows), 2):
        gmb, gnmb, _, _, _ = collate(ds, rows[gs], dynamic=False)
        logits = student(**gmb, use_cache=False).logits
        expected.append(loss_func(logits.float().view(-1, logits.shape[-1]), gnmb["label"].view(-1)))
    torch.testing.assert_close(got, torch.stack(expected).sum() / 4, rtol=1e-5, atol=1e-7)
