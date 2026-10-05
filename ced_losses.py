"""Loss and self-distillation helpers of the CED trainer.

Moved unchanged out of ced_finetune.py, so that the training step (ced_step.py) and its
tests can import them without DeepSpeed. ced_finetune.py imports them back."""
import copy
import json
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationConfig

import ced_omask
from gen_config import generation_kwargs
from distillm import forward_kl, reverse_kl, js_distance, tv_distance
from distillm import skewed_forward_kl, skewed_reverse_kl, csd, amid

IGNORE = -100


def label_window(label):
    """[start, end) over every column of `label` (rows x columns) that holds a label in some
    row. CE, KD and LwF only read logits at labelled positions, so the forward can skip the
    LM head outside this window (logits_to_keep). A batch with no label at all gets a
    1-column window: the shapes stay valid and the losses see no labels, as before."""
    cols = (label != IGNORE).any(dim=0).nonzero().flatten()
    if cols.numel() == 0:
        return 0, 1
    return int(cols[0]), int(cols[-1]) + 1

def get_distil_loss(args, teacher_logits, no_model_batch, logits):
    if args.model_parallel:
        raise NotImplementedError
    else:
        if "sfkl" in args.type:
            distil_loss = skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
        elif "srkl" in args.type:
            distil_loss = skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
        elif "jsd" in args.type:
            distil_loss = js_distance(logits, teacher_logits, no_model_batch)
        elif "tvd" in args.type:
            distil_loss = tv_distance(logits, teacher_logits, no_model_batch)
        elif "fkl" in args.type or args.type == "kd":
            distil_loss = forward_kl(logits, teacher_logits, no_model_batch)
        elif "rkl" in args.type:
            distil_loss = reverse_kl(logits, teacher_logits, no_model_batch)
        elif "csd" in args.type:
            distil_loss = csd(logits, teacher_logits, no_model_batch)
        elif "amid" in args.type:
            distil_loss = amid(logits, teacher_logits, no_model_batch, args)
        elif "no" in args.type:
            # a tensor, not 0.0: the caller all_reduces this, and with --w-span 0 nothing
            # downstream would turn a float back into one
            distil_loss = torch.tensor(0.0, device=logits.device)
        else:
            raise NotImplementedError
    return distil_loss


def select_batch_rows(batch, indices):
    if batch is None:
        return None
    index_list = indices.detach().cpu().tolist()
    selected = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            selected[key] = value.index_select(0, indices.to(value.device))
        elif isinstance(value, (list, tuple)):
            selected[key] = [copy.deepcopy(value[index]) for index in index_list]
        else:
            selected[key] = copy.deepcopy(value)
    return selected


def replace_batch_rows(batch, replacement, indices):
    index_list = indices.detach().cpu().tolist()
    merged = {}
    for key, value in batch.items():
        if key not in replacement:
            merged[key] = value
        elif isinstance(value, torch.Tensor):
            merged[key] = value.clone()
            merged[key].index_copy_(0, indices.to(value.device), replacement[key].to(value.device))
        elif isinstance(value, (list, tuple)):
            merged[key] = copy.deepcopy(value)
            for source_index, target_index in enumerate(index_list):
                merged[key][target_index] = copy.deepcopy(replacement[key][source_index])
        else:
            merged[key] = copy.deepcopy(replacement[key])
    return merged


def generate_replay_rows(args, student_generator, model, gen_data, no_model_batch, indices):
    replay_gen_data = select_batch_rows(gen_data, indices)
    generated_model = student_generator.run_sample(model, replay_gen_data)
    generated_label = generated_model.pop("no_model_batch")
    if args.model_type == "opt":
        generated_model.pop("position_ids", None)
    replay_metadata = select_batch_rows(no_model_batch, indices)
    replay_metadata["label"] = generated_label
    replay_metadata["loss_mask"] = (generated_label != -100).float()
    replay_metadata["is_replay"] = torch.ones(
        len(indices), dtype=torch.bool, device=generated_label.device
    )
    return generated_model, replay_metadata, replay_gen_data
# ---------------- on-policy self-distillation (SDFT) ----------------
# The student's frozen base is the merged previous-task model and only the LoRA
# tensors train, so the EMA teacher is that same base plus an EMA copy of the
# LoRA tensors, swapped in for the teacher forward. The teacher reads the
# augmented answer (gold + pseudo-labels) in its prompt; both models score the
# same sampled response tokens.

SD_EOS_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|> (same stop set as evaluate())


def sd_lora_params(model):
    return {n: p for n, p in model.module.named_parameters() if p.requires_grad}


def sd_ema_init(model):
    return {n: p.detach().float().clone() for n, p in sd_lora_params(model).items()}


def sd_ema_update(model, ema, mu):
    with torch.no_grad():
        for n, p in sd_lora_params(model).items():
            ema[n].mul_(mu).add_(p.detach().float(), alpha=1.0 - mu)


@contextmanager
def sd_ema_weights(model, ema):
    # in-place swap: must run before any grad-tracking forward of the same step,
    # or autograd sees the LoRA tensors modified after being saved for backward
    params = sd_lora_params(model)
    saved = {n: p.detach().clone() for n, p in params.items()}
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(ema[n].to(p.dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(saved[n])


def sd_left_pad(seqs, pad_id, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(seqs), L), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = s
        mask[i, L - len(s):] = 1
    return {"input_ids": ids, "attention_mask": mask}


def sd_pack(prefixes, responses, pad_id, device):
    """Right-padded [prefix, response] rows, plus the logit positions that predict
    each response token (pos[i, j] predicts responses[i][j]) and a label tensor
    holding the response tokens (-100 past each row's end)."""
    bs = len(prefixes)
    L = max(len(p) + len(r) for p, r in zip(prefixes, responses))
    R = max(len(r) for r in responses)
    ids = torch.full((bs, L), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((bs, L), dtype=torch.long, device=device)
    pos = torch.zeros((bs, R), dtype=torch.long, device=device)
    label = torch.full((bs, R), -100, dtype=torch.long, device=device)
    for i, (p, r) in enumerate(zip(prefixes, responses)):
        n = len(p) + len(r)
        ids[i, :n] = torch.cat([p, r])
        mask[i, :n] = 1
        pos[i, :len(r)] = torch.arange(len(p) - 1, n - 1, device=device)
        label[i, :len(r)] = r
    return ids, mask, pos, label


def sd_gather(logits, pos):
    return logits.gather(1, pos.unsqueeze(-1).expand(-1, -1, logits.size(-1)))


def sd_teacher_entropy(t_logits, label):
    """Mean H(teacher) over the distilled positions. `forward_kl` in distillm is the
    soft cross-entropy, which is KL(teacher||student) + H(teacher); only the KL part
    carries a gradient, so the logged value moves with the teacher's confidence unless
    this is subtracted. Measured on GENEVA: H was 1.35 of a 1.67 soft-CE, i.e. 81%."""
    lt = F.log_softmax(t_logits, dim=-1, dtype=torch.float32)
    return (-(lt.exp() * lt).sum(-1)[label != -100]).mean().item()


def sd_parses(text):
    """Strict JSON, the same bar ed_eval.py scores a prediction against."""
    try:
        json.loads(text.strip())
        return True
    except Exception:
        return False


def sd_prepare(args, tokenizer, model, ema, gen_data, no_model_batch, device):
    """Sample one response per prompt from the current student, then score it
    with the EMA teacher under the teacher prompt. Runs before the step's
    grad-tracking forwards (see sd_ema_weights).

    Returns (sd_batch, stats). sd_batch is None when no row survives, and the
    caller must then skip the SD loss: forward_kl/reverse_kl divide by the
    number of unmasked tokens, so an empty batch would give NaN."""
    gen_config = GenerationConfig(
        do_sample=True, top_p=args.ced_sd_top_p, top_k=0, temperature=args.ced_sd_temperature,
        eos_token_id=list(SD_EOS_IDS), pad_token_id=SD_EOS_IDS[0],
        return_dict_in_generate=True, output_scores=False)
    P = gen_data["input_ids"].size(1)
    model.eval()
    with torch.no_grad():
        seqs = model.generate(**gen_data, generation_config=gen_config,
                              max_new_tokens=args.max_length - P, **generation_kwargs(args)).sequences

    # real prompt tokens come from the attention mask: pad == eos == <|im_end|>,
    # which also appears inside the chat prompt, so filtering by pad id is wrong
    t_prompt_ids = no_model_batch["t_prompt_ids"]
    stats = {"rows": seqs.size(0), "truncated": 0, "unparsed": 0, "kept": 0, "masked_rec": 0, "masked_tok": 0}
    prompts, responses, t_prompts, kept_rows = [], [], [], []
    for i in range(seqs.size(0)):
        r = seqs[i, P:]
        stop = (r == SD_EOS_IDS[0]) | (r == SD_EOS_IDS[1])
        if not stop.any():
            # Cut off by the token budget, so it ends mid-object. Distilling it
            # could only teach the student to leave its JSON unterminated.
            stats["truncated"] += 1
            continue
        r = r[:stop.nonzero()[0].item() + 1]
        if args.ced_sd_skip_unparsed and not sd_parses(tokenizer.decode(r, skip_special_tokens=True)):
            stats["unparsed"] += 1
            continue
        prompts.append(gen_data["input_ids"][i][gen_data["attention_mask"][i].bool()])
        responses.append(r)
        t_prompts.append(t_prompt_ids[i].to(device))
        kept_rows.append(i)
    stats["kept"] = len(responses)
    if not responses:
        model.train()
        return None, stats

    t_ids, t_mask, t_pos, label = sd_pack(t_prompts, responses, SD_EOS_IDS[0], device)
    with torch.no_grad(), sd_ema_weights(model, ema):
        t_logits = sd_window_logits(model, t_ids, t_mask, t_pos, label)
    model.train()

    s_ids, s_mask, s_pos, _ = sd_pack(prompts, responses, SD_EOS_IDS[0], device)
    # both gathers must point at the logit that predicts the same response token
    valid = label != -100
    assert torch.equal(t_ids.gather(1, t_pos + 1)[valid], label[valid])
    assert torch.equal(s_ids.gather(1, s_pos + 1)[valid], label[valid])
    if args.ced_sd_omission_mask:
        sd_omission_mask(args, tokenizer, prompts, responses, label, kept_rows, no_model_batch["label"], stats)
        if not (label != -100).any():
            return None, stats
    return {"ids": s_ids, "mask": s_mask, "pos": s_pos, "label": label, "t_logits": t_logits,
            "rows": kept_rows, "t_entropy": sd_teacher_entropy(t_logits, label),
            "resp_len": sum(len(r) for r in responses) / len(responses)}, stats


_OLD_TYPES = {}


def sd_omission_mask(args, tokenizer, prompts, responses, label, kept_rows, ref_label, stats):
    """Drop from the SD loss (label -> -100) the tokens of sampled old-type records that are
    grounded in the sentence but absent from the reference y~. See ced_omask.py."""
    key = (args.ced_streams_file, args.ced_task_id)
    if key not in _OLD_TYPES:
        with open(args.ced_streams_file) as f:
            streams = json.load(f)
        _OLD_TYPES[key] = {t for s in streams[:args.ced_task_id] for t in s}
    old_types = _OLD_TYPES[key]
    for k, (p, r, row) in enumerate(zip(prompts, responses, kept_rows)):
        ids = r.tolist()
        ref = ref_label[row]
        reference = tokenizer.decode(ref[ref != -100].tolist(), skip_special_tokens=True)
        sentence = ced_omask.input_sentence(tokenizer.decode(p.tolist(), skip_special_tokens=True))
        spans = ced_omask.masked_spans(tokenizer.decode(ids, skip_special_tokens=True), reference,
                                       sentence, old_types)
        if not spans:
            continue
        drop = torch.tensor(ced_omask.token_mask(tokenizer, ids, spans), device=label.device)
        label[k, :len(ids)][drop] = -100
        stats["masked_rec"] += len(spans)
        stats["masked_tok"] += int(drop.sum())


def sd_loss_fn(args, model, sd):
    logits = model(input_ids=sd["ids"], attention_mask=sd["mask"], use_cache=False).logits
    logits = sd_gather(logits, sd["pos"])
    batch = {"label": sd["label"]}
    if args.ced_sd_div == "rkl":
        return reverse_kl(logits, sd["t_logits"], batch)
    return forward_kl(logits, sd["t_logits"], batch)


def sd_window_logits(model, ids, mask, pos, label):
    """sd_gather(model(ids).logits, pos), with the LM head run only over the columns that the
    valid entries of pos (label != -100) cover. Entries past a row's response are clamped
    into that window; nothing reads them."""
    valid = label != IGNORE
    lo = int(pos[valid].min()) if valid.any() else 0
    hi = int(pos[valid].max()) + 1 if valid.any() else 1
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False,
                   logits_to_keep=torch.arange(lo, hi, device=ids.device)).logits
    return sd_gather(logits, (pos - lo).clamp(0, hi - lo - 1))


def sd_student_logits(model, sd):
    """The student's logits on the sampled responses of every kept row (the SD forward)."""
    return sd_window_logits(model, sd["ids"], sd["mask"], sd["pos"], sd["label"])


def sd_group_loss(args, s_logits, sd, in_group):
    """The SD loss 4257d86 computes for a micro-batch whose kept rows are `in_group`. Returns
    None when that micro-batch has nothing left to distill (no kept row, or every token
    masked): 4257d86's sd_prepare returns None there, and the micro-batch gets no SD term."""
    label = sd["label"][in_group]
    if not (label != IGNORE).any():
        return None
    batch = {"label": label}
    if args.ced_sd_div == "rkl":
        return reverse_kl(s_logits[in_group], sd["t_logits"][in_group], batch)
    return forward_kl(s_logits[in_group], sd["t_logits"][in_group], batch)
def compute_token_weights(hidden_state, attention_mask):
    std = hidden_state.std(dim=-1, keepdim=True) + 1e-5
    Q = hidden_state / std
    K = hidden_state / std
    scores = torch.matmul(Q, K.transpose(-1, -2)) / (hidden_state.size(-1) ** 0.5)

    mask = attention_mask.unsqueeze(1).expand(-1, scores.size(-2), -1)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    diag_mask = torch.eye(scores.size(-1), device=scores.device, dtype=torch.bool)
    scores = scores.masked_fill(diag_mask.unsqueeze(0), float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)  # [1, L, L]
    attn_weights = attn_weights * mask
    attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

    token_weights = attn_weights.mean(dim=1).squeeze(0)  # [L]
    return token_weights.detach()

def prepare_span_indices_and_weights(t_layer_weights, s_layer_weights, 
                                     attention_mask, offsets_mapping, spans_offsets):
    device = attention_mask.device
    B_size, SeqLen = attention_mask.shape

    max_spans = max(len(s) for s in spans_offsets)
    if max_spans == 0:
        print(f"No spans found in the batch.")
        return None, None, None, None, None, None

    # (B_size, max_spans)
    padded_span_starts = torch.zeros(B_size, max_spans, dtype=torch.long, device=device)
    padded_span_ends = torch.zeros(B_size, max_spans, dtype=torch.long, device=device)
    padded_span_mask = torch.zeros(B_size, max_spans, dtype=torch.bool, device=device)

    for i in range(B_size):
        num_spans_i = len(spans_offsets[i])
        if num_spans_i > 0:
            spans_i = torch.tensor(spans_offsets[i], device=device, dtype=torch.long)
            padded_span_starts[i, :num_spans_i] = spans_i[:, 0]
            padded_span_ends[i, :num_spans_i] = spans_i[:, 1]
            padded_span_mask[i, :num_spans_i] = True
    
    if offsets_mapping.shape[1] != SeqLen:
        current_offsets_mapping = offsets_mapping[:, :SeqLen, :]
    else:
        current_offsets_mapping = offsets_mapping

    # (B_size, SeqLen, 1)
    offsets_start_expanded = current_offsets_mapping[..., 0].unsqueeze(2).to(device)
    offsets_end_expanded = current_offsets_mapping[..., 1].unsqueeze(2).to(device)
    
    # (B_size, 1, max_spans)
    span_starts_expanded = padded_span_starts.unsqueeze(1)
    span_ends_expanded = padded_span_ends.unsqueeze(1)

    token_in_span_map = (offsets_start_expanded + 1 >= span_starts_expanded) & \
                        (offsets_end_expanded <= span_ends_expanded)

    attention_mask_expanded = attention_mask.unsqueeze(2).bool()
    span_mask_expanded = padded_span_mask.unsqueeze(1) 

    final_token_to_span_map = token_in_span_map & attention_mask_expanded & span_mask_expanded

    if not final_token_to_span_map.any():
        print(f"No valid tokens found for any spans in the batch.")
        return torch.tensor(0.0, device=device)

    nonzero_indices = final_token_to_span_map.nonzero(as_tuple=False)
    
    batch_indices = nonzero_indices[:, 0] # (T_total)
    token_indices = nonzero_indices[:, 1] # (T_total)
    local_span_indices = nonzero_indices[:, 2] # (T_total)

    All_Indices = batch_indices * SeqLen + token_indices

    global_span_ids_flat = batch_indices * max_spans + local_span_indices
    _, Span_IDs = torch.unique(global_span_ids_flat, return_inverse=True) # (T_total)
    Max_Spans = Span_IDs.max().item() + 1 # Tổng số span duy nhất

    Batch_ID_for_Spans = torch.empty(Max_Spans, device=device, dtype=torch.long)
    Batch_ID_for_Spans.scatter_(0, Span_IDs, batch_indices)

    def gather_layer_weights(layer_weights):
        B_size, SeqLen = attention_mask.shape
        num_layers = layer_weights.shape[0]
        layer_weights_flat = layer_weights.view(num_layers, B_size * SeqLen)
        token_weights_unnorm = layer_weights_flat[:, All_Indices].float()
        batch_indices_expanded = batch_indices.unsqueeze(0).expand(num_layers, -1)
        sample_weight_sums = torch.zeros(num_layers, B_size, device=device, dtype=torch.float)
        sample_weight_sums.scatter_add_(1, batch_indices_expanded, token_weights_unnorm)
        sample_weight_sums = sample_weight_sums.clamp(min=1e-5)
        sample_weight_sums_gathered = torch.gather(sample_weight_sums, 1, batch_indices_expanded)
        Token_Weights_all = token_weights_unnorm / sample_weight_sums_gathered

        return Token_Weights_all

    T_Token_Weights_all = gather_layer_weights(t_layer_weights)
    S_Token_Weights_all = gather_layer_weights(s_layer_weights)

    return All_Indices, T_Token_Weights_all, S_Token_Weights_all, Span_IDs, Max_Spans, Batch_ID_for_Spans

def get_span_loss(attention_mask, s_hidden_states, t_hidden_states, 
                  offsets_mapping, spans_offsets, teacher_layer_mapping, student_layer_mapping, metric="cosine"):
    
    t_layer_weights = []
    s_layer_weights = []
    for i in teacher_layer_mapping:
        weights = compute_token_weights(t_hidden_states[i], attention_mask)  # (B, SeqLen)
        t_layer_weights.append(weights)
    for i in student_layer_mapping:
        weights = compute_token_weights(s_hidden_states[i], attention_mask)  # (B, SeqLen)
        s_layer_weights.append(weights)

    t_layer_weights = torch.stack(t_layer_weights)  # (num_layers, B, SeqLen)
    s_layer_weights = torch.stack(s_layer_weights)  # (num_layers, B, SeqLen)

    (All_Indices, T_Token_Weights_all, S_Token_Weights_all, 
     Span_IDs, Max_Spans, Batch_ID_for_Spans) =  prepare_span_indices_and_weights(t_layer_weights, s_layer_weights, 
                                                                                  attention_mask, offsets_mapping, spans_offsets)
    if All_Indices is None:
        return torch.tensor(0.0, device=attention_mask.device)
    
    final_loss = 0.0
    for i, (s_idx, t_idx) in enumerate(zip(student_layer_mapping, teacher_layer_mapping)):
        s_hidden = s_hidden_states[s_idx]
        t_hidden = t_hidden_states[t_idx]
        span_loss = compute_hidden_span_loss(s_hidden, t_hidden, All_Indices,
                                             S_Token_Weights_all[i], T_Token_Weights_all[i], 
                                             Span_IDs, Max_Spans, Batch_ID_for_Spans, metric=metric)
        final_loss += span_loss

    return final_loss

def compute_overall_span_loss(attention_mask, s_hidden_states, t_hidden_states, 
                              offsets_mapping, spans_offsets, args):
    
    s_span_mapping = args.student_layer_mapping
    t_span_mapping = args.teacher_layer_mapping
    span_loss = get_span_loss(attention_mask, s_hidden_states, t_hidden_states, 
                              offsets_mapping, spans_offsets, t_span_mapping, s_span_mapping, metric=args.span_metric)
    
    overall_loss = span_loss / len(args.student_layer_mapping)
    return overall_loss

CKA_MIN_SPANS = 4  # centering kills one rank; below this the Gram matrix carries no structure


def cka_span_loss(S_span, T_span, Batch_ID_for_Spans):
    """1 - linear CKA between the student's and the teacher's span representations,
    per batch item, averaged. Unlike the pairwise-distance metrics this compares the
    two Gram matrices as a whole, so it only asks the two similarity structures to be
    linearly related instead of equal — a looser constraint, and one that cannot carry
    the per-pair importance weights omega_ik. Items with fewer than CKA_MIN_SPANS spans
    are skipped: after centering they leave a rank-1 Gram and CKA is degenerate."""
    losses = []
    for b in Batch_ID_for_Spans.unique():
        idx = (Batch_ID_for_Spans == b).nonzero(as_tuple=True)[0]
        if idx.numel() < CKA_MIN_SPANS:
            continue
        X = S_span[idx].float()
        Y = T_span[idx].float()
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        num = (Y.T @ X).pow(2).sum()
        den = (X.T @ X).norm() * (Y.T @ Y).norm()
        losses.append(1.0 - num / den.clamp(min=1e-6))
    if not losses:
        return torch.zeros((), device=S_span.device)
    return torch.stack(losses).mean()


def compute_hidden_span_loss(s_hidden_state, t_hidden_state, All_Indices,
                             S_Token_Weights_all, T_Token_Weights_all, Span_IDs, Max_Spans, Batch_ID_for_Spans, metric="cosine"):
    D_hidden_s = s_hidden_state.size(-1)
    D_hidden_t = t_hidden_state.size(-1)
    device = t_hidden_state.device

    T_Hidden_Flat = t_hidden_state.flatten(0, 1) # (B*SeqLen, D_hidden_t)
    S_Hidden_Flat = s_hidden_state.flatten(0, 1) # (B*SeqLen, D_hidden_s)

    # 1. Trích xuất và Áp dụng Trọng số
    T_span_all = T_Hidden_Flat[All_Indices] # (T_total, D_hidden_t)
    S_span_all = S_Hidden_Flat[All_Indices] # (T_total, D_hidden_s)
    
    T_Token_Weights_expanded = T_Token_Weights_all.unsqueeze(-1) 
    S_Token_Weights_expanded = S_Token_Weights_all.unsqueeze(-1)
    
    T_span_weighted = T_span_all * T_Token_Weights_expanded # (T_total, D_hidden_t)
    S_span_weighted = S_span_all * S_Token_Weights_expanded # (T_total, D_hidden_s)

    Span_IDs_expanded_t = Span_IDs.unsqueeze(-1).expand(-1, D_hidden_t) 
    Span_IDs_expanded_s = Span_IDs.unsqueeze(-1).expand(-1, D_hidden_s) 

    T_span_sum = torch.zeros(Max_Spans, D_hidden_t, device=device)
    S_span_sum = torch.zeros(Max_Spans, D_hidden_s, device=device)
    T_Weight_sum_1d = torch.zeros(Max_Spans, device=device)
    S_Weight_sum_1d = torch.zeros(Max_Spans, device=device)

    T_span_sum.scatter_add_(0, Span_IDs_expanded_t, T_span_weighted)
    S_span_sum.scatter_add_(0, Span_IDs_expanded_s, S_span_weighted)

    T_Weight_sum_1d.scatter_add_(0, Span_IDs, T_Token_Weights_all) 
    T_Weight_sum = T_Weight_sum_1d.clamp(min=1e-5).unsqueeze(-1) # (Max_Spans, 1)
    S_Weight_sum_1d.scatter_add_(0, Span_IDs, S_Token_Weights_all)
    S_Weight_sum = S_Weight_sum_1d.clamp(min=1e-5).unsqueeze(-1) # (Max_Spans, 1)

    # Tính Trung bình (Mean)
    T_span_hidden_mean = T_span_sum / T_Weight_sum 
    S_span_hidden_mean = S_span_sum / S_Weight_sum

    if metric == "cka":
        return cka_span_loss(S_span_hidden_mean, T_span_hidden_mean, Batch_ID_for_Spans)

    if metric == "cosine":
        S_normalized = F.normalize(S_span_hidden_mean, p=2, dim=-1)
        T_normalized = F.normalize(T_span_hidden_mean, p=2, dim=-1)
        S_Full_Sim_Matrix = S_normalized @ S_normalized.T
        T_Full_Sim_Matrix = T_normalized @ T_normalized.T
    elif metric == "dot":
        S_Full_Sim_Matrix = S_span_hidden_mean @ S_span_hidden_mean.T
        T_Full_Sim_Matrix = T_span_hidden_mean @ T_span_hidden_mean.T
    elif metric == "l2":
        S_Full_Sim_Matrix = torch.cdist(S_span_hidden_mean, S_span_hidden_mean, p=2)
        T_Full_Sim_Matrix = torch.cdist(T_span_hidden_mean, T_span_hidden_mean, p=2)

    else:
        raise ValueError(f"unknown span_metric: {metric}")

    Batch_IDs_col = Batch_ID_for_Spans.unsqueeze(1)
    Batch_IDs_row = Batch_ID_for_Spans.unsqueeze(0)
    Same_Batch_Mask = (Batch_IDs_col == Batch_IDs_row)
    Not_Self_Mask = ~torch.eye(Max_Spans, dtype=torch.bool, device=device)
    Final_Mask = Same_Batch_Mask & Not_Self_Mask

    S_intra_batch_similarities_flat = torch.masked_select(S_Full_Sim_Matrix, Final_Mask)
    T_intra_batch_similarities_flat = torch.masked_select(T_Full_Sim_Matrix, Final_Mask)

    Pair_Weights_Matrix = T_Weight_sum_1d.unsqueeze(1) * T_Weight_sum_1d.unsqueeze(0)
    Valid_Pair_Weights = torch.masked_select(Pair_Weights_Matrix, Final_Mask)

    span_loss = F.mse_loss(S_intra_batch_similarities_flat, T_intra_batch_similarities_flat, reduction='none')
    span_loss = (span_loss * Valid_Pair_Weights).sum() / Valid_Pair_Weights.sum().clamp(min=1e-5)

    return span_loss
