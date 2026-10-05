"""Evaluation shared by finetune.py and ced_finetune.py.

evaluate() makes up to two passes over a split:
  - a loss pass in chunks of --eval-loss-batch-size rows (32, the historical evaluation
    batch, so the logged loss and the adaptive threshold that reads it average the same
    chunks as before), with logits only over the label window;
  - when asked, a generation pass in batches of --eval-batch-size rows.

eval_plan() decides which passes run at an evaluation point. --eval-gen-mode every is the
historical schedule; final generates answers only for the test set after the last update.
"""
import json
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import GenerationConfig

from ced_losses import label_window
from ed_eval import ed_evaluate
from gen_config import generation_kwargs
from rouge_metric import compute_metrics
from utils import all_gather, get_rank, print_rank, save_rank


@dataclass(frozen=True)
class EvalPlan:
    dev: bool            # run the dev pass
    dev_generate: bool   # with generation
    test: bool           # run the test pass (always with generation)


def eval_plan(mode, is_last, adaptive):
    if mode == "every":
        return EvalPlan(dev=True, dev_generate=True, test=True)
    if mode == "final":
        # nothing reads dev except DistiLLM/AMiD's adaptive threshold, and only its loss
        return EvalPlan(dev=adaptive, dev_generate=False, test=is_last)
    raise ValueError(f"unknown --eval-gen-mode {mode!r}")


def final_test_missing(args, done):
    """--eval-gen-mode final takes the test answers at the last update. When that evaluation
    point never fired (eval_interval does not divide total_iters, or the task had no full
    update), the trainer evaluates the test set once after training."""
    return args.eval_gen_mode == "final" and not done


def _loader(args, dataset, batch_size):
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False,
                                 rank=dist.get_rank(), num_replicas=dist.get_world_size())
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size,
                      num_workers=args.num_workers, collate_fn=dataset.collate)


def evaluate(args, tokenizer, model, dataset, split, epoch, device, adaptive_threshold=None, generate=True):
    dp_world_size = dist.get_world_size()
    loss_func = nn.CrossEntropyLoss()
    print_rank("dp size", dp_world_size)
    model.eval()

    all_loss, step = 0.0, 0
    with torch.no_grad():
        for model_batch, no_model_batch, gen_data, _, _ in _loader(args, dataset, args.eval_loss_batch_size):
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            label = no_model_batch["label"]
            start, end = label_window(label)
            logits = model(**model_batch, use_cache=False,
                           logits_to_keep=torch.arange(start, end, device=label.device)).logits
            loss = loss_func(logits.reshape(-1, logits.shape[-1]), label[:, start:end].reshape(-1))
            dist.all_reduce(loss, dist.ReduceOp.SUM)
            all_loss += (loss / dp_world_size).item()
            step += 1

    responses = None
    if generate and args.eval_gen:
        generation_config = GenerationConfig(
            do_sample=args.do_sample, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature,
            repetition_penalty=args.repetition_penalty, max_length=args.max_length, min_length=None,
            eos_token_id=[tokenizer.eos_token_id, 151643], pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True, output_scores=False)
        loader = _loader(args, dataset, args.eval_batch_size)
        all_response_ids = []
        with torch.no_grad():
            for it, (_, _, gen_data, _, _) in enumerate(tqdm(loader, desc="Evaluating",
                                                             disable=(dist.get_rank() != 0))):
                print_rank(f"{it}/{len(loader)}")
                gen_data = {k: v.to(device) for k, v in gen_data.items()}
                width = gen_data["input_ids"].size(1)
                sequences = model.generate(**gen_data, generation_config=generation_config,
                                           max_new_tokens=args.max_length - width,
                                           **generation_kwargs(args)).sequences
                sequences = F.pad(sequences, (0, args.max_length - sequences.shape[1]),
                                  value=tokenizer.pad_token_id)
                all_response_ids.append(sequences[:, width:])
        all_response_ids = torch.cat(all_response_ids, dim=0)
        all_response_ids = all_gather(all_response_ids, dim=1, world_size=dp_world_size, op="stack")
        all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
        responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)

    if get_rank() == 0:
        res = {}
        if responses is not None:
            references = dataset.answers
            responses = responses[:len(references)]
            res = compute_metrics(responses, references)
            res.update(ed_evaluate(responses, references))
            eval_dir = os.path.join(args.save, "eval", str(epoch))
            print_rank(eval_dir)
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
                for resp in responses:
                    f.write(json.dumps({"text": resp}) + "\n")
        avg_loss = all_loss / step
        if "adaptive" in args.type:
            log_str = f"{split} | avg_loss: {avg_loss} | {res} | threshold: {adaptive_threshold}"
        else:
            log_str = f"{split} | avg_loss: {avg_loss} | {res}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))

    return all_loss / step
