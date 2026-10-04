import copy
import time
import os
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed

import random
import json
from tqdm import tqdm
import math
import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig)

from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from arguments import get_args

from data_utils.lm_datasets import LMTrainDataset
from data_utils.data_utils import LLMDataset
from utils import get_optimizer_params, get_optimizer_params_peft, print_args, initialize
from utils import print_rank, get_rank
from utils import save_rank
from utils import all_gather
from utils import load_parallel, save_parallel
from utils import get_tokenizer, get_model

from distillm import forward_kl, reverse_kl, js_distance, tv_distance
from distillm import skewed_forward_kl, skewed_reverse_kl, csd, amid
from distillm import SampleGenerator, ReplayBuffer

from rouge_metric import compute_metrics

from peft import PeftModel
from ed_eval import ed_evaluate
import ced_omask

torch.set_num_threads(4)


def get_teacher_model(args, device):
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    if args.model_parallel:
        raise NotImplementedError
    else:
        config.is_model_parallel = False
        try: model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.bfloat16)
        except:
            model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.float32)
            model = model.half()
        
        if args.teacher_peft_path is not None:
            model = PeftModel.from_pretrained(model, args.teacher_peft_path)
            model = model.merge_and_unload()
            print("merge_and_unload")

        if dist.get_rank() == 0:
            print(' > number of parameters: {}'.format(
                sum([p.nelement() for p in model.parameters()])), flush=True)

    model.eval()
    
    return model


def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    if args.peft is not None:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)

    # Use AdamW.
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.total_iters,
            eta_min=args.lr_min)
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5)
    elif args.lr_decay_style == "wrmup_cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_ratio * args.total_iters,
            num_training_steps=args.total_iters)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler


def setup_model_and_optimizer(args, ds_config, device, set_optim=True):
    # get the model
    model = get_model(args, device)
    # get the optimizer and lr_scheduler
    if set_optim:
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
        optimizer, lr_scheduler = None, None
        
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config
    )
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model, optimizer, lr_scheduler


def prepare_dataset(args, tokenizer):
    data = {}
    rng_sample = random.Random(args.seed)
    if args.do_train:
        data["train"] = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["train"]))
        data["dev"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)

    data["test"] = LMTrainDataset(args, tokenizer, args.data_dir, "test", args.dev_num, args.dev_ratio, rng_sample)

        
    # pre-trained dataset
    if args.do_train and args.lm_data_dir is not None:
        data["pt_train"] = LMTrainDataset(args, tokenizer, args.lm_data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["pt_train"]))
    return data


def pt_loss(args, model, model_batch, no_model_batch):
    loss_mask = (no_model_batch["label"] != -100).int()
    outputs = model(**model_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), no_model_batch["label"].view(-1))
    return lm_loss


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


def get_teacher_lm_loss(args, tokenizer, model, teacher_model, model_batch):
    with torch.no_grad():
        t_gen_out = teacher_model.generate(
            **model_batch,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args.max_length,
            top_k=0,
            top_p=1,
            temperature=1.0,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=False)
    
    full_ids = t_gen_out.sequences
    
    input_ids = full_ids[:, :-1]
    mask = (input_ids != tokenizer.pad_token_id).long()
    labels = full_ids[:, 1:]    
    labels = torch.masked_fill(labels, mask==0, -100)
    labels[:, :model_batch["input_ids"].size(1)-1] = -100
    loss_mask = (labels != -100).float()
    
    new_batch = {
        "input_ids": input_ids,
        "attention_mask": mask,
    }
    
    if args.model_type in ["gpt2"]:
        position_ids = torch.cumsum(mask, dim=-1) - 1
        position_ids = torch.masked_fill(position_ids, mask==0, 0)    
        new_batch["position_ids"] = position_ids    
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    outputs = model(**new_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    return lm_loss


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
                              max_new_tokens=args.max_length - P).sequences

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
        t_logits = model(input_ids=t_ids, attention_mask=t_mask, use_cache=False).logits
        t_logits = sd_gather(t_logits, t_pos)
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
            "t_entropy": sd_teacher_entropy(t_logits, label),
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


def sd_probe(args, tokenizer, model, ema, dataset, device):
    """Before training: does the teacher copy the reference answer from its
    prompt? Greedy-decode the first rows with and without the reference and
    score both against the augmented answer. A teacher that ignores the
    reference gives SD no signal (SDFT reports weak in-context use below 7B)."""
    n = min(args.ced_sd_probe, len(dataset))
    _, no_model_batch, gen_data, _, _ = dataset.collate([dataset[i] for i in range(n)])
    rows = [dataset.sample_indices[i] if dataset.sample_indices is not None else i for i in range(n)]
    refs = [dataset.answers[j] for j in rows]
    gen_config = GenerationConfig(
        do_sample=False, eos_token_id=list(SD_EOS_IDS), pad_token_id=SD_EOS_IDS[0],
        return_dict_in_generate=True, output_scores=False)

    def decode(prompt_seqs):
        batch = sd_left_pad([s.to(device) for s in prompt_seqs], SD_EOS_IDS[0], device)
        with torch.no_grad(), sd_ema_weights(model, ema):
            seqs = model.generate(**batch, generation_config=gen_config,
                                  max_new_tokens=args.max_length - args.max_prompt_length).sequences
        return tokenizer.batch_decode(seqs[:, batch["input_ids"].size(1):], skip_special_tokens=True)

    s_prompts = [gen_data["input_ids"][i][gen_data["attention_mask"][i].bool()] for i in range(n)]
    model.eval()
    with_ref = ed_evaluate(decode(no_model_batch["t_prompt_ids"]), refs)
    no_ref = ed_evaluate(decode(s_prompts), refs)
    model.train()

    # a teacher prompt cut by --t-max-prompt-length loses its generation-prompt suffix
    mismatch = 0
    for j in range(len(dataset.lm_ctx)):
        s = dataset.lm_ctx[j].astype(int)
        t = dataset.t_lm_ctx[j].astype(int)
        sp = s[:np.where(s == 4294967295)[0][0]]
        tp = t[:np.where(t == 4294967295)[0][0]]
        mismatch += int(not np.array_equal(sp[-4:], tp[-4:]))

    log_str = (f"sd probe | rows {n} | teacher+ref trigger F1 {with_ref['trigger']['f1']:.4f} "
               f"arg F1 {with_ref['argument']['f1']:.4f} | no ref trigger F1 {no_ref['trigger']['f1']:.4f} "
               f"arg F1 {no_ref['argument']['f1']:.4f} | prompt suffix mismatch {mismatch}/{len(dataset.lm_ctx)}")
    print_rank(log_str)
    save_rank(log_str, os.path.join(args.save, "log.txt"))

    # Warn rather than stop, so one weak probe does not take the rest of a queue down.
    gap = with_ref["trigger"]["f1"] - no_ref["trigger"]["f1"]
    problems = []
    if gap < 0.03:
        problems.append(f"the reference in the teacher prompt adds only {gap:+.4f} trigger F1, "
                        f"so the teacher knows little the student does not")
    if mismatch:
        # The two prompts should end in the same generation suffix. [:max] truncation in
        # tools/process_data.py drops the tail, so a mismatch means one side lost
        # <|im_start|>assistant: the teacher (over --t-max-prompt-length) or the student
        # (over --max-prompt-length 460, which already happens for a few MAVEN/RAMS rows).
        problems.append(f"{mismatch} rows end in different prompt suffixes, one side was "
                        f"truncated past its generation prompt")
    for p in problems:
        warn = f"WARNING sd probe: {p}"
        print_rank(warn)
        save_rank(warn, os.path.join(args.save, "log.txt"))


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


def evaluate_loss(args, model, dataset, device):
    sampler = DistributedSampler(
        dataset, shuffle=False, drop_last=False, rank=dist.get_rank(),
        num_replicas=dist.get_world_size()
    )
    # Use the training batch size, not eval_batch_size: the float32 cast below
    # materialises batch x seq x vocab (32 x 768 x 151936 x 4B = 14.9 GB at
    # eval_batch_size 32), which OOMs a 46 GB card before the first step.
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.batch_size,
        num_workers=args.num_workers, collate_fn=dataset.collate
    )
    loss_func = nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()
    all_loss = 0.0
    steps = 0
    with torch.no_grad():
        for model_batch, no_model_batch, gen_data, _, _ in dataloader:
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            logits = model(**model_batch, use_cache=False).logits
            loss = loss_func(
                logits.float().reshape(-1, logits.shape[-1]),
                no_model_batch["label"].view(-1),
            )
            dist.all_reduce(loss, dist.ReduceOp.SUM)
            all_loss += (loss / dist.get_world_size()).item()
            steps += 1
    if was_training:
        model.train()
    return all_loss / max(steps, 1)

def finetune(args, tokenizer: AutoTokenizer, model: deepspeed.DeepSpeedEngine, optimizer: AdamW, lr_scheduler, dataset, device, teacher_model=None):
    print_rank("Start Fine-tuning")

    # print_inspect(model, '*')
    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    train_dataloader = DataLoader(
        dataset['train'], sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["train"].collate)
    
    if "pt_train" in dataset:
        pt_sampler = DistributedSampler(dataset["pt_train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
        pt_train_dataloader = DataLoader(
        dataset['pt_train'], sampler=pt_sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["pt_train"].collate)
        pt_train_iter = iter(pt_train_dataloader)
        
    student_generator = SampleGenerator(args, tokenizer)

    step, global_step = 1, 1
    total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0
    
    adaptive_threshold = args.init_threshold if "adaptive" in args.type else -1.0
    prev_avg_loss = (
        evaluate_loss(args, model, dataset["dev"], device)
        if "adaptive" in args.type else 0.0
    )
    replay_buffer = ReplayBuffer(args)

    student_captured_hidden = []
    hook_handles = []
    capture = {"on": True}  # off during the SD student forward, whose states the span loss must not see
    def capture_hook_fn(module, input, output):
        if module.training and capture["on"]:
            if isinstance(output, tuple):
                student_captured_hidden.append(output[0])
            else:
                student_captured_hidden.append(output)

    if args.w_span_loss != 0:
        for layer in model.base_model.model.model.layers:
            h_layer = layer.register_forward_hook(capture_hook_fn)
            hook_handles.append(h_layer)

    sd_ema = None
    total_sd_loss, total_sd_len, total_sd_ent = 0.0, 0.0, 0.0
    # per log window: how many micro-steps actually distilled, and what happened to the samples
    SD_STAT_KEYS = ("steps", "active", "rows", "truncated", "unparsed", "kept", "masked_rec", "masked_tok")
    sd_win = dict.fromkeys(SD_STAT_KEYS, 0)
    assert args.ced_sd or args.ced_sd_mix == "sum", "--ced-sd-mix random needs --ced-sd"
    if args.ced_sd:
        assert not args.student_gen, "--ced-sd and --student-gen both replace generation; use one"
        assert dataset["train"].t_lm_ctx is not None, \
            "--ced-sd needs teacher prompts: run tools/ced_sd_prompts.py before tokenizing"
        sd_ema = sd_ema_init(model)
        if args.ced_sd_probe > 0:
            sd_probe(args, tokenizer, model, sd_ema, dataset["train"], device)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        alloc_sum = alloc_count = 0

        model.train()
        for it, (model_batch, no_model_batch, gen_data, _, _) in enumerate(train_dataloader):
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)
            student_captured_hidden.clear()
            student_captured_hidden.append(None)
            
            if args.lm_data_dir is not None:
                try:
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                    # pt_model_batch, pt_no_model_batch, pt_gen_data = pt_train_iter.next()
                except:
                    pt_train_iter = iter(pt_train_dataloader)
                    # pt_model_batch, pt_no_model_batch, pt_gen_data = pt_train_iter.next()
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                    
                dataset["pt_train"].move_to_device(pt_model_batch, pt_no_model_batch, pt_gen_data, device)
            
            torch.cuda.synchronize()
            st_time = time.time()
            
            # # sampling ratio:
            samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)
            if "adaptive" in args.type:
                if args.replay_ratio == "constant":
                    samp_threshold = adaptive_threshold * 0.5
                elif args.replay_ratio == "increasing":
                    samp_threshold = adaptive_threshold * global_step / args.total_iters
                else:
                    samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)
            
            # DistiLLM generation is restricted to replay rows. New-task rows keep
            # their gold CE targets and remain in the same mixed training batch.
            if args.student_gen:
                r = np.random.uniform(0, 1)
                replay_indices = no_model_batch["is_replay"].nonzero(as_tuple=False).flatten()
                replay_count = len(replay_indices)
                should_generate = (
                    ("mixed" in args.type and r < args.mixed_alpha)
                    or ("adaptive" in args.type and (
                        r < samp_threshold
                        or (r < adaptive_threshold and len(replay_buffer) < args.capacity)
                    ))
                )
                should_sample = (
                    "adaptive" in args.type
                    and r < adaptive_threshold
                    and len(replay_buffer) >= replay_count
                )
                if replay_count and should_generate:
                    generated_model, generated_metadata, replay_gen_data = generate_replay_rows(
                        args, student_generator, model, gen_data, no_model_batch, replay_indices
                    )
                    replay_buffer.move_to_memory(
                        generated_model, generated_metadata, replay_gen_data
                    )
                    if "mixed" in args.type:
                        generated_model, generated_metadata, _ = replay_buffer.sample(replay_count)
                        generated_model, generated_metadata, _ = replay_buffer.move_to_device(
                            generated_model, generated_metadata, None, device
                        )
                    model_batch = replace_batch_rows(model_batch, generated_model, replay_indices)
                    no_model_batch = replace_batch_rows(
                        no_model_batch, generated_metadata, replay_indices
                    )
                    print_rank(
                        f"student-gen replay insert: {replay_count}, buffer={len(replay_buffer)}"
                    )
                elif replay_count and should_sample:
                    sampled_model, sampled_metadata, sampled_gen = replay_buffer.sample(replay_count)
                    sampled_model, sampled_metadata, sampled_gen = replay_buffer.move_to_device(
                        sampled_model, sampled_metadata, sampled_gen, device
                    )
                    model_batch = replace_batch_rows(model_batch, sampled_model, replay_indices)
                    no_model_batch = replace_batch_rows(
                        no_model_batch, sampled_metadata, replay_indices
                    )
                    print_rank(
                        f"student-gen replay sample: {replay_count}, buffer={len(replay_buffer)}"
                    )
                model.train()

            # --ced-sd-mix random: one KL term per update, token KD against the old-task model
            # or SD against the EMA teacher. Seeded by global_step so all micro-steps of an
            # update, and all ranks, draw the same side.
            use_token_kd, use_sd = True, True
            if args.ced_sd_mix == "random":
                use_sd = random.Random(args.seed * 1000003 + global_step).random() < 0.5
                use_token_kd = not use_sd

            # SD sampling + EMA-teacher scoring come before the grad-tracking forwards
            sd_batch = None
            if args.ced_sd:
                sd_win["steps"] += 1
                # warmup is an ablation knob (default 0); EMA keeps tracking the student during it
                if use_sd and global_step > args.ced_sd_warmup * args.total_iters:
                    sd_batch, st = sd_prepare(args, tokenizer, model, sd_ema, gen_data, no_model_batch, device)
                    for k in ("rows", "truncated", "unparsed", "kept", "masked_rec", "masked_tok"):
                        sd_win[k] += st[k]
                    sd_win["active"] += int(sd_batch is not None)

            outputs = model(**model_batch, use_cache=False)

            logits = outputs.logits

            # CED: route losses by replay flag. Replay samples (exemplars of old
            # tasks) get KD + span loss against the previous-task teacher; new-task
            # samples get plain CE. --ced-replay-mode kd_only drops CE on replay.
            # --ced-kd-scope pl additionally applies KD on old-event token positions
            # of pseudo-labeled rows (the only region where the old teacher is
            # reliable on new-task data) — LwF adapted to generation.
            is_replay = no_model_batch["is_replay"]  # (bs,) bool
            has_replay = bool(is_replay.any().item())

            old_token_mask = no_model_batch.get("old_token_mask")
            use_pl_kd = (getattr(args, "ced_kd_scope", "replay") == "pl"
                         and old_token_mask is not None)
            pl_rows = None
            has_kd = has_replay
            if use_pl_kd:
                pl_rows = old_token_mask.any(dim=1) & ~is_replay
                has_kd = has_replay or bool(pl_rows.any().item())

            # LwF: distill the old teacher on NEW-task rows' non-new-type tokens,
            # with a small capped weight added ON TOP of the replay KD (never stealing
            # from CE). Off by default (--ced-kd-ratio-new 0).
            new_token_mask = no_model_batch.get("new_token_mask")
            kd_ratio_new = getattr(args, "ced_kd_ratio_new", 0.0)
            new_rows = ~is_replay
            lwf_active = (teacher_model is not None and kd_ratio_new > 0.0
                          and new_token_mask is not None and bool(new_rows.any().item()))

            if args.model_parallel:
                raise NotImplementedError

            ce_label = no_model_batch["label"]
            if args.ced_replay_mode == "kd_only" and has_replay:
                ce_label = ce_label.clone()
                ce_label[is_replay] = -100
            if (ce_label != -100).any():
                lm_loss = loss_func(logits.float().reshape(-1, logits.shape[-1]), ce_label.view(-1))
            else:
                lm_loss = torch.tensor(0.0, device=logits.device)

            distil_loss = torch.tensor(0.0, device=logits.device)
            distil_loss_new = torch.tensor(0.0, device=logits.device)
            if teacher_model is not None and (has_kd or lwf_active):
                with torch.no_grad():
                    teacher_model.eval()
                    teacher_outputs = teacher_model(
                        **model_batch,
                        output_hidden_states=(args.w_span_loss != 0),
                        use_cache=False,
                    )
                    teacher_logits = teacher_outputs.logits

                if has_kd:
                    kd_no_model_batch = dict(no_model_batch)
                    kd_label = no_model_batch["label"].clone()
                    keep = torch.zeros_like(kd_label, dtype=torch.bool)
                    keep[is_replay] = True
                    if use_pl_kd:
                        keep |= old_token_mask[:, :kd_label.size(1)].to(keep.device) & pl_rows.unsqueeze(1)
                    kd_label[~keep] = -100
                    kd_no_model_batch["label"] = kd_label
                    if use_token_kd and (kd_label != -100).any():
                        distil_loss = get_distil_loss(args, teacher_logits, kd_no_model_batch, logits)

                    if args.w_span_loss != 0:
                        old_spans = no_model_batch.get("old_span_offsets")
                        spans_offsets = []
                        for i, (spans, flag) in enumerate(zip(
                            no_model_batch["span_offsets"], is_replay.tolist()
                        )):
                            if flag:
                                spans_offsets.append(spans)
                            elif use_pl_kd and old_spans is not None:
                                spans_offsets.append(old_spans[i])
                            else:
                                spans_offsets.append([])
                        span_loss = compute_overall_span_loss(
                            model_batch["attention_mask"],
                            student_captured_hidden,
                            teacher_outputs.hidden_states,
                            no_model_batch["offset_mapping"],
                            spans_offsets,
                            args,
                        )
                        distil_loss = distil_loss + args.w_span_loss * span_loss

                if lwf_active:
                    # KD on new-task rows' answer tokens EXCEPT new-type-event tokens
                    # (the teacher is unreliable exactly where the new types appear).
                    lwf_no_model_batch = dict(no_model_batch)
                    lwf_label = no_model_batch["label"].clone()
                    keep_new = (new_rows.unsqueeze(1)
                                & (lwf_label != -100)
                                & ~new_token_mask[:, :lwf_label.size(1)].to(lwf_label.device))
                    lwf_label[~keep_new] = -100
                    lwf_no_model_batch["label"] = lwf_label
                    if (lwf_label != -100).any():
                        distil_loss_new = get_distil_loss(args, teacher_logits, lwf_no_model_batch, logits)

            if has_kd:
                loss = (1 - args.kd_ratio) * lm_loss + args.kd_ratio * distil_loss
            else:
                loss = lm_loss
            if lwf_active:
                w_new = min(kd_ratio_new, getattr(args, "ced_kd_new_cap", 0.3))
                loss = loss + w_new * distil_loss_new

            if sd_batch is not None:
                capture["on"] = False
                sd_loss = sd_loss_fn(args, model, sd_batch)
                capture["on"] = True
                loss = loss + args.ced_sd_weight * sd_loss
            elif args.ced_sd:
                # nothing to distill this step (warmup, or every sample dropped); keep a
                # tensor so the all_reduce below runs on every rank and cannot hang
                sd_loss = torch.zeros((), device=loss.device)

            if args.lm_data_dir is not None:
                assert args.lm_coef is not None
                loss += args.lm_coef * pt_loss(args, model, pt_model_batch, pt_no_model_batch)

            boundary = model.is_gradient_accumulation_boundary()
            model.backward(loss)
            model.step()
            if sd_ema is not None and boundary:
                sd_ema_update(model, sd_ema, args.ced_sd_ema_mu)

            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            global_loss = loss.item() / dp_world_size

            if args.ced_sd:
                dist.all_reduce(sd_loss, dist.ReduceOp.SUM, group=dp_group)
                total_sd_loss += sd_loss.item() / dp_world_size
                if sd_batch is not None:
                    total_sd_len += sd_batch["resp_len"]
                    total_sd_ent += sd_batch["t_entropy"]

            global_distil_loss = 0
            if teacher_model is not None:
                dist.all_reduce(distil_loss, dist.ReduceOp.SUM, group=dp_group)
                global_distil_loss = distil_loss.item() / dp_world_size
                total_distil_loss += global_distil_loss
    
            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

            total_loss += global_loss
            total_time += elapsed_time

            # Logging
            def get_log(log_loss, log_distil_loss, log_time):
                return "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | loss: {:.4f} | ds_loss: {:.4f} | lr: {:.4e} | scale: {:10.4f} | micro time: {:.3f} | step time: {:.3f}".format(
                    epoch,
                    step,
                    args.total_iters * args.gradient_accumulation_steps,
                    global_step,
                    args.total_iters,
                    log_loss,
                    log_distil_loss,
                    lr_scheduler.get_last_lr()[0],
                    optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    elapsed_time,
                    log_time,
                )

            if args.mid_log_num > 0:
                mid_log_step = args.gradient_accumulation_steps // args.mid_log_num
                mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                if step % mid_log_step == 0:
                    print_rank(get_log(global_loss, global_distil_loss, 0))

            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                log_str = get_log(
                    total_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_distil_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_time / (args.log_interval))
                print_rank("*" * 100)
                print_rank(log_str)
                print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0
                if sd_ema is not None:
                    # averages over the steps that distilled; "active" says how many that was.
                    # An arm whose active rate sits near 0 is plain training under another name.
                    act, rows = max(sd_win["active"], 1), max(sd_win["rows"], 1)
                    # kl is the part of the loss that actually has a gradient: fkl logs
                    # KL + H(teacher), rkl is already the plain KL. Watch kl, not sd_loss.
                    ent = total_sd_ent / act
                    kl = total_sd_loss / act - (ent if args.ced_sd_div == "fkl" else 0.0)
                    sd_str = ("sd | global iter: {:6d} | sd_loss: {:.4f} | kl: {:.4f} | teacher_H: {:.4f} | "
                              "resp_len: {:.1f} | active: {}/{} steps | kept: {:.3f} | truncated: {:.3f} | "
                              "unparsed: {:.3f} | omask: {} records / {} tokens").format(
                        global_step, total_sd_loss / act, kl, ent, total_sd_len / act,
                        sd_win["active"], sd_win["steps"], sd_win["kept"] / rows,
                        sd_win["truncated"] / rows, sd_win["unparsed"] / rows,
                        sd_win["masked_rec"], sd_win["masked_tok"])
                    print_rank(sd_str)
                    save_rank(sd_str, os.path.join(args.save, "log.txt"))
                    total_sd_loss, total_sd_len, total_sd_ent = 0.0, 0.0, 0.0
                    sd_win = dict.fromkeys(SD_STAT_KEYS, 0)

                # --- MEMORY MEASUREMENT BLOCK ---
                allocated   = torch.cuda.memory_allocated() / 1e9
                peak_alloc  = torch.cuda.max_memory_allocated() / 1e9

                alloc_sum   += allocated
                alloc_count += 1
                avg_alloc   = alloc_sum / alloc_count

                print_rank("train | avg_alloc {:.4f} GB | peak_alloc {:.4f} GB".format(avg_alloc, peak_alloc))

                # --- END MEMORY MEASUREMENT BLOCK ---
            
            # Checkpointing
            if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
                save_dir_path = os.path.join(args.save, str(global_step))
                if args.model_parallel:
                    raise NotImplementedError
                else:
                    if dist.get_rank() == 0:
                        os.makedirs(save_dir_path, exist_ok=True)
                        print_rank(f"Model save to {save_dir_path}")
                        tokenizer.save_pretrained(save_dir_path)
                        model.module.save_pretrained(save_dir_path, safe_serialization=False)
                dist.barrier()

            # Evaluation
            if args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0:
                curr_avg_loss = evaluate(args, tokenizer, model, dataset["dev"], "dev", epoch, device, adaptive_threshold)
                if "adaptive" in args.type:
                    if curr_avg_loss >= prev_avg_loss + args.loss_eps:
                        adaptive_threshold += 0.1
                        adaptive_threshold = min(adaptive_threshold, 1.0)
                        prev_avg_loss = curr_avg_loss

                evaluate(args, tokenizer, model, dataset["test"], "test", epoch, device)
                    
                model.train()
                
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            
            if global_step > args.total_iters:
                break

    for h in hook_handles:
        h.remove()
            
    return model


def evaluate(args, tokenizer, model, dataset: LMTrainDataset, split, epoch, device, adaptive_threshold=None):
    
    collate_fn = dataset.collate

    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss()

    print_rank("dp size", dp_world_size)

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=[tokenizer.eos_token_id, 151643],
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    model.eval()
    all_loss = 0.0
    step = 0
    
    all_response_ids = []
    
    with torch.no_grad():
        for it, (model_batch, no_model_batch, gen_data, _, _) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):
            print_rank(f"{it}/{len(dataloader)}")
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            logits = model(**model_batch).logits
            if args.model_parallel:
                raise NotImplementedError
            else:
                loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            
            max_new_tokens = args.max_length - gen_data["input_ids"].size(1)
            
            if args.eval_gen:            
                gen_out = model.generate(
                    **gen_data,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens)
                
                full_ids = gen_out.sequences
                
                full_ids = F.pad(
                    full_ids,
                    (0, args.max_length - full_ids.shape[1]),
                    value=tokenizer.pad_token_id,
                )
                
                response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                all_response_ids.append(response_ids)
                    
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            loss = loss / dp_world_size
            all_loss += loss.item()
            step += 1
    
    if args.eval_gen:
        all_response_ids = torch.cat(all_response_ids, dim=0)
        all_response_ids = all_gather(all_response_ids, dim=1, world_size=dp_world_size, group=dp_group, op="stack")
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
        
        responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)
    
    if get_rank() == 0:
        if args.eval_gen:
            references = dataset.answers
            responses = responses[:len(references)]
            
            res = compute_metrics(responses, references)

            ed_metrics = ed_evaluate(responses, references)
            res.update(ed_metrics)
        
            eval_dir = os.path.join(args.save, "eval", str(epoch))
            print_rank(eval_dir)
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
                for resp in responses:
                    f.write(json.dumps({"text": resp}) + "\n")
        else:
            res = {}
    
        avg_loss = all_loss / step
        
        if "adaptive" in args.type:
            log_str = f"{split} | avg_loss: {avg_loss} | {res} | threshold: {adaptive_threshold}"
        else:
            log_str = f"{split} | avg_loss: {avg_loss} | {res}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
        
    return all_loss / step


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    args.fp32 = not ds_config["fp16"]["enabled"]  
    args.bf16 = "bf16" in ds_config and ds_config["bf16"]["enabled"]  
    args.deepspeed_config = None
    
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    print(type(tokenizer))

    dataset = prepare_dataset(
        args,
        tokenizer,
    )
    
    dp_world_size = dist.get_world_size()
    
    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)
    
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type
    
    if args.teacher_model_path is not None:
        teacher_model = get_teacher_model(args, device)
    else:
        teacher_model = None
    
    if args.do_train:
        model = finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device, teacher_model=teacher_model)
   
    if args.do_eval:
        evaluate(args, tokenizer, model, dataset["test"], "test", 0, device)
        
    
if __name__ == "__main__":
    main()
