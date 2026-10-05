"""The loss of one physical batch in ced_finetune.py, computed as today's loss per logical
micro-batch.

ced_finetune.py defines its objective per micro-batch:
  - CE and KD are token means over the micro-batch;
  - the (1 - kd_ratio) * CE + kd_ratio * KD mix applies only to micro-batches that hold a
    memory (or pseudo-labelled) row;
  - the span loss normalizes over the micro-batch's span pairs.

--loss-group-size keeps that definition while one forward covers --batch-size rows. Rows
[0, g), [g, 2g), ... are each scored as one micro-batch of the code at 4257d86, and the step
loss is their mean. With the gradient accumulation divided by the same factor, every update
averages the same micro-batch losses as before (spec section 1).

Inside the step, logits are computed only over the label window, and the teacher runs only
on the rows whose loss reads it.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ced_losses import IGNORE, compute_overall_span_loss, get_distil_loss, label_window


def group_slices(n_rows, group_size):
    if group_size < 1 or n_rows % group_size:
        raise ValueError(f"a batch of {n_rows} rows is not a whole number of loss groups of {group_size}")
    return [slice(start, start + group_size) for start in range(0, n_rows, group_size)]


@dataclass
class StepLoss:
    loss: torch.Tensor          # mean over groups: what the engine backpropagates
    lm_loss: torch.Tensor       # mean CE over groups (logging)
    distil_loss: torch.Tensor   # mean over groups of the KD (+ span) term, 0 where a group has none (logging)
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
    Return 4257d86's per-micro-batch loss, averaged over groups of args.loss_group_size rows.

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

    losses, lms, distils, sd_terms = [], [], [], []
    for j, gs in enumerate(groups):
        lab_g, rep_g = lab[gs], is_replay[gs]
        ce_label = lab_g
        if args.ced_replay_mode == "kd_only" and bool(rep_g.any()):
            ce_label = lab_g.clone()
            ce_label[rep_g] = IGNORE
        ce_pos = ce_label != IGNORE
        if ce_pos.any():
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

        loss = (1 - args.kd_ratio) * lm + args.kd_ratio * distil if has_kd[j] else lm
        if lwf[j]:
            loss = loss + min(kd_ratio_new, getattr(args, "ced_kd_new_cap", 0.3)) * distil_new
        losses.append(loss)
        lms.append(lm.detach())
        distils.append(distil.detach())

    zero = torch.zeros((), device=device)
    return StepLoss(torch.stack(losses).mean(), torch.stack(lms).mean(), torch.stack(distils).mean(),
                    torch.stack(sd_terms).sum() if sd_terms else zero, len(sd_terms))


def grouped_ce_loss(model, model_batch, label, group_size):
    """finetune.py's CE (a token mean over each micro-batch) per group of `group_size` rows,
    averaged. The LM head runs only over the label window."""
    start, end = label_window(label)
    logits = model(**model_batch, use_cache=False,
                   logits_to_keep=torch.arange(start, end, device=label.device)).logits
    lab = label[:, start:end]
    terms = []
    for gs in group_slices(label.size(0), group_size):
        pos = lab[gs] != IGNORE
        terms.append(F.cross_entropy(logits[gs][pos].float(), lab[gs][pos]) if pos.any()
                     else torch.tensor(0.0, device=label.device))
    return torch.stack(terms).mean()
