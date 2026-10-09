"""The loss of one physical batch in ced_finetune.py, computed as today's loss per logical
micro-batch.

ced_finetune.py defines its objective per micro-batch:
  - CE and KD are token means over the micro-batch;
  - the (1 - kd_ratio) * CE + kd_ratio * KD mix applies only to micro-batches that hold a
    memory (or pseudo-labelled) row; --ced-ce-mix rows narrows the (1 - kd_ratio) on CE to the
    memory rows' tokens of such a micro-batch (default batch: every row's, as above);
  - the span loss normalizes over the micro-batch's span pairs.

--loss-group-size keeps that definition while one forward covers --batch-size rows. Rows
[0, g), [g, 2g), ... are each scored as one micro-batch of the code at 4257d86, and the step
loss is their sum over the number of groups in a full physical batch. With the gradient
accumulation divided by the same factor, every micro-batch keeps the weight it had before,
1/accumulation (spec section 1). That includes the last batch of an epoch, which holds the
leftover rows: its groups, the last one possibly short like 4257d86's last micro-batch, are
divided by the full batch's group count too.

Inside the step, logits are computed only over the label window, and the teacher runs only
on the rows whose loss reads it.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ced_losses import (IGNORE, compute_overall_span_loss, generate_replay_rows, get_distil_loss, label_window,
                        replace_batch_rows, sd_group_loss, sd_student_logits, select_batch_rows)


def group_slices(n_rows, group_size):
    """Rows [0, g), [g, 2g), ...: one slice per logical micro-batch. In the last batch of an
    epoch the last slice can be short, as 4257d86's last micro-batch was."""
    if group_size < 1:
        raise ValueError(f"--loss-group-size must be positive, got {group_size}")
    return [slice(start, min(start + group_size, n_rows)) for start in range(0, n_rows, group_size)]


def updates_per_epoch(n_rows, batch_size, world_size, accumulation):
    """Optimizer updates per epoch, counted as the trainers always have. A training split
    smaller than one update is refused here: 4257d86 crashed on a division by zero in its
    first step instead."""
    rows_per_update = batch_size * world_size * accumulation
    updates = int(n_rows / rows_per_update)
    if updates < 1:
        raise ValueError(f"the training split has {n_rows} rows, fewer rows than one update "
                         f"({batch_size} rows x {world_size} GPUs x {accumulation} accumulation = "
                         f"{rows_per_update}); lower --batch-size or --gradient-accumulation-steps")
    return updates


def first_global_step(physical_accumulation, logical_accumulation):
    """The global_step a trainer's loop starts at. The trainers start step and global_step at 1
    and advance global_step after micro-step accumulation - 1 of each update. With several
    micro-steps per update, that puts the log, save and eval checks after update m at m + 1, and
    ends the loop one micro-step short of the last update, which is never applied. A physical
    batch holding a whole update (PHYS_BS >= --bs x --acc) has no micro-step before the update
    ends, so it starts one ahead: it then logs, saves, evaluates and stops as the run it splits.
    A run with one micro-step per update to begin with (--acc 1) keeps 1 and every update."""
    return 2 if physical_accumulation == 1 and logical_accumulation > 1 else 1


@dataclass
class StepLoss:
    loss: torch.Tensor          # sum over groups / groups per full batch: what the engine backpropagates
    lm_loss: torch.Tensor       # CE, summed the same way (logging)
    distil_loss: torch.Tensor   # KD (+ span) term, summed the same way, 0 where a group has none (logging)
    sd_loss_sum: torch.Tensor   # sum of the groups' SD losses (logging)
    sd_groups: int              # groups that got an SD term


def _all_valid(mask):
    """A label tensor marking every gathered position as distilled."""
    return {"label": torch.zeros(int(mask.sum()), dtype=torch.long, device=mask.device)}


def _gather(logits, t_logits, t_index, offset, mask):
    """Student and teacher logits at the positions of `mask`, the rows of one group starting
    at row `offset` of the physical batch."""
    rows, cols = mask.nonzero(as_tuple=True)
    return logits[offset + rows, cols], t_logits[t_index[offset + rows], cols]


def _group_span_loss(args, gs, kd_rows, is_replay, use_pl_kd, model_batch, no_model_batch,
                     captured, t_hidden, t_index):
    """4257d86's span loss of micro-batch `gs`, evaluated on its KD rows. The other rows carry
    no spans, so they never enter the span pairs or their normalization."""
    rows = [r for r in range(gs.start, gs.stop) if bool(kd_rows[r])]
    old_spans = no_model_batch.get("old_span_offsets")
    spans = []
    for r in rows:
        if bool(is_replay[r]):
            spans.append(no_model_batch["span_offsets"][r])
        elif use_pl_kd and old_spans is not None:
            spans.append(old_spans[r])
        else:
            spans.append([])
    idx = torch.tensor(rows, device=is_replay.device)
    s_hidden = [None] + [h.index_select(0, idx) for h in captured[1:]]
    t_hid = [h.index_select(0, t_index[idx]) for h in t_hidden]
    offsets = no_model_batch["offset_mapping"]
    return compute_overall_span_loss(model_batch["attention_mask"].index_select(0, idx), s_hidden, t_hid,
                                     offsets.index_select(0, idx.to(offsets.device)), spans, args)


def ced_step_loss(args, model, teacher_model, model_batch, no_model_batch, captured, capture,
                  sd_batch=None, use_token_kd=True):
    """Forward the physical batch once, and the teacher only on the rows it is needed for.
    Return 4257d86's per-micro-batch loss for groups of args.loss_group_size rows, summed and
    divided by the groups of a full --batch-size batch (the mean, except in a short last batch).

    captured: the list the span-loss hooks fill during the student forward ([None] first).
    capture: the dict whose "on" flag gates those hooks.
    sd_batch: sd_prepare's result for this physical batch, or None.
    use_token_kd: False on the updates where --ced-sd-mix random picked SD.
    """
    label = no_model_batch["label"]
    device = label.device
    n = label.size(0)
    groups = group_slices(n, args.loss_group_size)
    is_replay = no_model_batch["is_replay"].to(device)
    old_token_mask = no_model_batch.get("old_token_mask")
    new_token_mask = no_model_batch.get("new_token_mask")
    use_pl_kd = getattr(args, "ced_kd_scope", "replay") == "pl" and old_token_mask is not None
    if use_pl_kd:
        old_token_mask = old_token_mask.to(device)
        pl_rows = old_token_mask.any(dim=1) & ~is_replay
    else:
        pl_rows = torch.zeros_like(is_replay)
    kd_rows = is_replay | pl_rows
    kd_ratio_new = getattr(args, "ced_kd_ratio_new", 0.0)
    if new_token_mask is not None:
        new_token_mask = new_token_mask.to(device)
    has_kd = [bool(kd_rows[gs].any()) for gs in groups]
    lwf = [teacher_model is not None and kd_ratio_new > 0.0 and new_token_mask is not None
           and bool((~is_replay[gs]).any()) for gs in groups]

    start, end = label_window(label)
    window = torch.arange(start, end, device=device)
    logits = model(**model_batch, use_cache=False, logits_to_keep=window).logits
    lab = label[:, start:end]

    # the teacher reads the KD rows of the groups that distill, and with LwF the new-task rows
    need_teacher = torch.zeros_like(is_replay)
    if teacher_model is not None:
        for gs, kd, lw in zip(groups, has_kd, lwf):
            if kd:
                need_teacher[gs] |= kd_rows[gs]
            if lw:
                need_teacher[gs] |= ~is_replay[gs]
    t_rows = need_teacher.nonzero().flatten()
    t_index = torch.full((n,), -1, dtype=torch.long, device=device)
    t_logits = t_hidden = None
    if t_rows.numel():
        t_index[t_rows] = torch.arange(t_rows.numel(), device=device)
        t_batch = {k: v.index_select(0, t_rows) for k, v in model_batch.items()}
        with torch.no_grad():
            teacher_model.eval()
            t_out = teacher_model(**t_batch, output_hidden_states=(args.w_span_loss != 0),
                                  use_cache=False, logits_to_keep=window)
        t_logits, t_hidden = t_out.logits, t_out.hidden_states

    sd_logits = sd_rows = None
    if sd_batch is not None:
        capture["on"] = False           # the span loss must not see the SD forward's hidden states
        sd_logits = sd_student_logits(model, sd_batch)
        capture["on"] = True
        sd_rows = torch.as_tensor(sd_batch["rows"], device=device)

    losses, lms, distils, sd_terms = [], [], [], []
    for j, gs in enumerate(groups):
        lab_g, rep_g = lab[gs], is_replay[gs]
        ce_label = lab_g
        if args.ced_replay_mode == "kd_only" and bool(rep_g.any()):
            ce_label = lab_g.clone()
            ce_label[rep_g] = IGNORE
        ce_pos = ce_label != IGNORE
        ce_rows = getattr(args, "ced_ce_mix", "batch") == "rows" and has_kd[j]
        if ce_pos.any() and ce_rows:
            # --ced-ce-mix rows: the same token mean over the micro-batch, but (1 - kd_ratio) only
            # on the memory rows' tokens; the mix below then leaves CE unscaled
            tok = F.cross_entropy(logits[gs][ce_pos].float(), ce_label[ce_pos], reduction="none")
            w = torch.where(rep_g.unsqueeze(1).expand_as(ce_pos)[ce_pos], 1 - args.kd_ratio, 1.0)
            lm = (w * tok).mean()
        elif ce_pos.any():
            lm = F.cross_entropy(logits[gs][ce_pos].float(), ce_label[ce_pos])
        else:
            lm = torch.tensor(0.0, device=device)

        distil = torch.tensor(0.0, device=device)
        distil_new = torch.tensor(0.0, device=device)
        if teacher_model is not None and (has_kd[j] or lwf[j]):
            if has_kd[j]:
                keep = torch.zeros_like(lab_g, dtype=torch.bool)
                keep[rep_g] = True
                if use_pl_kd:
                    keep |= old_token_mask[gs][:, start:end] & pl_rows[gs].unsqueeze(1)
                kd_pos = keep & (lab_g != IGNORE)
                if use_token_kd and kd_pos.any():
                    s_k, t_k = _gather(logits, t_logits, t_index, gs.start, kd_pos)
                    distil = get_distil_loss(args, t_k, _all_valid(kd_pos), s_k)
                if args.w_span_loss != 0:
                    span = _group_span_loss(args, gs, kd_rows, is_replay, use_pl_kd, model_batch,
                                            no_model_batch, captured, t_hidden, t_index)
                    distil = distil + args.w_span_loss * span
            if lwf[j]:
                keep_new = (~rep_g).unsqueeze(1) & (lab_g != IGNORE) & ~new_token_mask[gs][:, start:end]
                if keep_new.any():
                    s_n, t_n = _gather(logits, t_logits, t_index, gs.start, keep_new)
                    distil_new = get_distil_loss(args, t_n, _all_valid(keep_new), s_n)

        if has_kd[j]:
            loss = (lm if ce_rows else (1 - args.kd_ratio) * lm) + args.kd_ratio * distil
        else:
            loss = lm
        if lwf[j]:
            loss = loss + min(kd_ratio_new, getattr(args, "ced_kd_new_cap", 0.3)) * distil_new
        if sd_logits is not None:
            sd_g = sd_group_loss(args, sd_logits, sd_batch, (sd_rows >= gs.start) & (sd_rows < gs.stop))
            if sd_g is not None:
                loss = loss + args.ced_sd_weight * sd_g
                sd_terms.append(sd_g.detach())
        losses.append(loss)
        lms.append(lm.detach())
        distils.append(distil.detach())

    zero = torch.zeros((), device=device)
    n_full = args.batch_size // args.loss_group_size
    return StepLoss(torch.stack(losses).sum() / n_full, torch.stack(lms).sum() / n_full,
                    torch.stack(distils).sum() / n_full,
                    torch.stack(sd_terms).sum() if sd_terms else zero, len(sd_terms))


def grouped_ce_loss(model, model_batch, label, group_size, n_groups):
    """finetune.py's CE (a token mean over each micro-batch) per group of `group_size` rows,
    summed and divided by n_groups, the groups of a full --batch-size batch (the mean, except
    in a short last batch). The LM head runs only over the label window."""
    start, end = label_window(label)
    logits = model(**model_batch, use_cache=False,
                   logits_to_keep=torch.arange(start, end, device=label.device)).logits
    lab = label[:, start:end]
    terms = []
    for gs in group_slices(label.size(0), group_size):
        pos = lab[gs] != IGNORE
        terms.append(F.cross_entropy(logits[gs][pos].float(), lab[gs][pos]) if pos.any()
                     else torch.tensor(0.0, device=label.device))
    return torch.stack(terms).sum() / n_groups


def distillm_replace_groups(args, groups, model, student_generator, replay_buffer, model_batch, no_model_batch,
                            gen_data, samp_threshold, adaptive_threshold, device, log=print):
    """DistiLLM/AMiD student generation as 4257d86 does it per micro-batch. Each group gets
    one draw and one generate / sample-from-buffer decision, in group order, against the
    buffer size the earlier groups leave. The rows to generate are generated in a single
    call. Buffer pushes and samples then replay in group order, so every sample sees the
    same buffer as before."""
    is_replay = no_model_batch["is_replay"]
    buffered = len(replay_buffer)
    plan = []
    for gs in groups:
        r = np.random.uniform(0, 1)
        idx = is_replay[gs].nonzero(as_tuple=False).flatten() + gs.start
        count = len(idx)
        should_generate = (("mixed" in args.type and r < args.mixed_alpha)
                           or ("adaptive" in args.type and (
                               r < samp_threshold or (r < adaptive_threshold and buffered < args.capacity))))
        should_sample = "adaptive" in args.type and r < adaptive_threshold and buffered >= count
        if count and should_generate:
            plan.append(("generate", idx))
            buffered = min(buffered + count, args.capacity)
        elif count and should_sample:
            plan.append(("sample", idx))
        else:
            plan.append(None)

    to_generate = [i for p in plan if p and p[0] == "generate" for i in p[1].tolist()]
    if to_generate:
        position = {row: k for k, row in enumerate(to_generate)}
        gen_model, gen_meta, gen_inputs = generate_replay_rows(
            args, student_generator, model, gen_data, no_model_batch,
            torch.tensor(to_generate, device=is_replay.device))
    for step in plan:
        if step is None:
            continue
        kind, idx = step
        if kind == "generate":
            sel = torch.tensor([position[i] for i in idx.tolist()], device=is_replay.device)
            rows_model = select_batch_rows(gen_model, sel)
            rows_meta = select_batch_rows(gen_meta, sel)
            replay_buffer.move_to_memory(rows_model, rows_meta, select_batch_rows(gen_inputs, sel))
            if "mixed" in args.type:
                rows_model, rows_meta, _ = replay_buffer.sample(len(idx))
                rows_model, rows_meta, _ = replay_buffer.move_to_device(rows_model, rows_meta, None, device)
            log(f"student-gen replay insert: {len(idx)}, buffer={len(replay_buffer)}")
        else:
            rows_model, rows_meta, rows_gen = replay_buffer.sample(len(idx))
            rows_model, rows_meta, rows_gen = replay_buffer.move_to_device(rows_model, rows_meta, rows_gen, device)
            log(f"student-gen replay sample: {len(idx)}, buffer={len(replay_buffer)}")
        model_batch = replace_batch_rows(model_batch, rows_model, idx)
        no_model_batch = replace_batch_rows(no_model_batch, rows_meta, idx)
    model.train()
    return model_batch, no_model_batch
