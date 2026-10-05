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
from ced_step import ced_step_loss, distillm_replace_groups, group_slices
from ced_losses import (
    get_distil_loss, select_batch_rows, replace_batch_rows, generate_replay_rows,
    SD_EOS_IDS, sd_lora_params, sd_ema_init, sd_ema_update, sd_ema_weights, sd_left_pad,
    sd_pack, sd_gather, sd_teacher_entropy, sd_parses, sd_prepare, sd_omission_mask,
    sd_loss_fn, compute_token_weights, prepare_span_indices_and_weights, get_span_loss,
    compute_overall_span_loss, CKA_MIN_SPANS, cka_span_loss, compute_hidden_span_loss)
from gen_config import generation_kwargs
from ced_eval import evaluate, eval_plan, final_test_missing

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
                                  max_new_tokens=args.max_length - args.max_prompt_length,
                                  **generation_kwargs(args)).sequences
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




def evaluate_loss(args, model, dataset, device):
    sampler = DistributedSampler(
        dataset, shuffle=False, drop_last=False, rank=dist.get_rank(),
        num_replicas=dist.get_world_size()
    )
    # Use the training batch size, not eval_batch_size: the float32 cast below
    # materialises batch x seq x vocab (32 x 768 x 151936 x 4B = 14.9 GB at
    # eval_batch_size 32), which OOMs a 46 GB card before the first step.
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.loss_group_size,
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
    final_test_done = False

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
            
            # DistiLLM/AMiD: 4257d86's per-micro-batch generation decisions, one per loss group
            if args.student_gen:
                model_batch, no_model_batch = distillm_replace_groups(
                    args, group_slices(model_batch["input_ids"].size(0), args.loss_group_size), model,
                    student_generator, replay_buffer, model_batch, no_model_batch, gen_data,
                    samp_threshold, adaptive_threshold, device, log=print_rank)

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
                sd_win["steps"] += model_batch["input_ids"].size(0) // args.loss_group_size
                # warmup is an ablation knob (default 0); EMA keeps tracking the student during it
                if use_sd and global_step > args.ced_sd_warmup * args.total_iters:
                    sd_batch, st = sd_prepare(args, tokenizer, model, sd_ema, gen_data, no_model_batch, device)
                    for k in ("rows", "truncated", "unparsed", "kept", "masked_rec", "masked_tok"):
                        sd_win[k] += st[k]

            if args.model_parallel:
                raise NotImplementedError

            # CED loss per logical micro-batch (ced_step.py): replay rows get KD + span loss
            # against the previous-task teacher, new-task rows plain CE, as before
            step_loss = ced_step_loss(args, model, teacher_model, model_batch, no_model_batch,
                                      student_captured_hidden, capture, sd_batch, use_token_kd)
            loss = step_loss.loss
            distil_loss = step_loss.distil_loss
            if args.ced_sd:
                sd_win["active"] += step_loss.sd_groups
                sd_loss = step_loss.sd_loss_sum

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
                if sd_batch is not None and step_loss.sd_groups:
                    total_sd_len += sd_batch["resp_len"] * step_loss.sd_groups
                    total_sd_ent += sd_batch["t_entropy"] * step_loss.sd_groups

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
                plan = eval_plan(args.eval_gen_mode, is_last=(global_step >= args.total_iters),
                                 adaptive=("adaptive" in args.type))
                if plan.dev:
                    curr_avg_loss = evaluate(args, tokenizer, model, dataset["dev"], "dev", epoch, device,
                                             adaptive_threshold, generate=plan.dev_generate)
                    if "adaptive" in args.type:
                        if curr_avg_loss >= prev_avg_loss + args.loss_eps:
                            adaptive_threshold += 0.1
                            adaptive_threshold = min(adaptive_threshold, 1.0)
                            prev_avg_loss = curr_avg_loss
                if plan.test:
                    evaluate(args, tokenizer, model, dataset["test"], "test", epoch, device)
                    final_test_done = True
                model.train()
                
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            
            if global_step > args.total_iters:
                break

    for h in hook_handles:
        h.remove()

    if final_test_missing(args, final_test_done):
        evaluate(args, tokenizer, model, dataset["test"], "test", max(args.epochs - 1, 0), device)
    return model


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    if args.loss_group_size is None:
        args.loss_group_size = args.batch_size
    if args.batch_size % args.loss_group_size:
        raise ValueError(f"--batch-size {args.batch_size} is not a multiple of --loss-group-size {args.loss_group_size}")
    if args.loss_group_size != args.batch_size and args.lm_data_dir is not None:
        raise ValueError("--lm-data-dir needs --loss-group-size equal to --batch-size")
    # padding length changes the span loss (its token weights average over padded queries), and the
    # DistiLLM replay buffer stores --max-length-wide rows: both keep fixed padding
    args.dynamic_pad_effective = args.dynamic_pad and args.w_span_loss == 0 and not args.student_gen
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
