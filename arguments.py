# coding=utf-8
# Copyright 2020 The OpenBMB team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import deepspeed
import numpy as np


def add_model_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--model-path', type=str, help='model path')
    group.add_argument("--ckpt-name", type=str)
    group.add_argument("--model-type", type=str, default="gpt2")
    group.add_argument("--teacher-model-type", type=str, default=None)
    group.add_argument("--n-gpu", type=int, default=1)
    group.add_argument("--n-nodes", type=int, default=1)
    group.add_argument("--teacher-model-path", type=str)
    group.add_argument("--teacher-ckpt-name", type=str)
    group.add_argument("--teacher-model-fp16", action="store_true")
    group.add_argument("--model-parallel", action="store_true")
    group.add_argument("--model-parallel-size", type=int, default=None)
    group.add_argument("--no-value", action="store_true")
    group.add_argument("--dropout-path-rate", type=float, default=None)
    group.add_argument("--fp32", action="store_true")
    group.add_argument("--bf16", action="store_true")
    return parser


def add_runtime_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('runtime', 'runtime configurations')

    group.add_argument("--type", type=str, default=None)
    group.add_argument("--do-train", action="store_true")
    group.add_argument("--do-valid", action="store_true")
    group.add_argument("--do-eval", action="store_true")
    group.add_argument('--base-path', type=str, default=None, help='Path to the project base directory.')
    group.add_argument('--load', type=str, default=None,
                       help='Path to a directory containing a model checkpoint.')
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument("--log-interval", type=int, default=10)
    group.add_argument("--mid-log-num", type=int, default=4)
    group.add_argument('--save-interval', type=int, default=1000,
                       help='number of iterations between saves')
    group.add_argument("--eval-interval", type=int, default=1000)
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')
    group.add_argument("--save-additional-suffix", type=str, default="")
    group.add_argument("--save-rollout", action="store_true")
    group.add_argument("--eb-sample-times", type=int, default=3)
    return parser


def add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('data', 'data configurations')
    group.add_argument("--data-dir", type=str, default=None)
    group.add_argument("--processed-data-dir", type=str, default=None)
    group.add_argument("--force-process", action="store_true")
    group.add_argument("--force-process-demo", action="store_true")
    group.add_argument("--data-process-workers", type=int, default=-1)
    group.add_argument("--train-num", type=int, default=-1)
    group.add_argument("--train-ratio", type=float, default=1)
    group.add_argument("--dev-num", type=int, default=-1)
    group.add_argument("--dev-ratio", type=float, default=1)
    group.add_argument("--test-num", type=int, default=None,
                       help="cap test rows separately from dev; defaults to --dev-num. "
                            "Lets a run subsample dev for per-epoch selection while still "
                            "scoring the full test set.")
    group.add_argument("--gen-num", type=int, default=-1)
    group.add_argument("--data-names", type=str, default=None)
    group.add_argument("--prompt-type", type=str, default=None)
    group.add_argument("--num-workers", type=int, default=1)
    group.add_argument("--max-prompt-length", type=int, default=512)
    group.add_argument("--t-max-prompt-length", type=int, default=640)
    group.add_argument("--min-prompt-length", type=int, default=128)
    group.add_argument("--json-data", action="store_true")
    group.add_argument("--bin-data", action="store_true")
    group.add_argument("--txt-data", action="store_true")
    
    group.add_argument("--prompt-data-dir", type=str)
    group.add_argument("--lm-data-dir", type=str)
    group.add_argument("--eval-ppl", action="store_true")
    group.add_argument("--eval-rw", action="store_true")
    group.add_argument("--eval-gen", action="store_true")
    group.add_argument("--eval-gen-mode", type=str, default="every", choices=["every", "final"],
                       help="every: dev and test answers after each epoch (historical); final: test answers "
                            "after the last update only, and a loss-only dev pass for the adaptive methods")
    group.add_argument("--eval-loss-batch-size", type=int, default=32,
                       help="rows per chunk of the evaluation loss pass (32 = the historical eval batch)")
    group.add_argument("--dynamic-pad", action="store_true",
                       help="pad each batch to its longest row (rounded up to 64) instead of --max-length; "
                            "ignored with a span loss or --student-gen")
    
    group.add_argument("--only-prompt", action="store_true")
    return parser


def add_hp_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("hp", "hyper parameter configurations")
    group.add_argument('--batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--eval-batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='gradient clipping')
    group.add_argument('--total-iters', type=int, default=None,
                       help='total number of iterations')
    group.add_argument('--train-iters-per-epoch', type=int, default=-1,
                       help='total number of iterations per epoch')
    group.add_argument('--max-length', type=int, default=1024,
                       help='max length of input')
    group.add_argument('--t-max-length', type=int, default=1024,
                       help='max length of input')
    group.add_argument('--seed', type=int, default=1234,
                       help='random seed for reproducibility')
    group.add_argument("--seed-order", type=int, default=42)
    group.add_argument("--seed-data", type=int, default=42)
    group.add_argument("--seed-ppo", type=int, default=42)
    group.add_argument("--seed-lm", type=int, default=7)
    group.add_argument('--epochs', type=int, default=None,
                       help='total number of epochs to train over all training runs')
    group.add_argument('--training-epochs', type=int, default=10000)
    group.add_argument("--gradient-accumulation-steps", type=int, default=1)
    group.add_argument("--loss-group-size", type=int, default=None,
                       help="rows per logical micro-batch, the unit the CE/KD loss is defined over; "
                            "--batch-size (the physical batch) must be a multiple of it (default: --batch-size)")
    group.add_argument("--gradient-checkpointing", action="store_true")
    group.add_argument("--attn-dtype", default=None)
    
    group.add_argument('--lr', type=float, help='initial learning rate')
    group.add_argument("--lr-min", type=float, default=0.0000001)
    group.add_argument('--weight-decay', type=float, default=1.0e-2,
                       help='weight-decay')
    group.add_argument('--loss-scale', type=float, default=65536,
                       help='loss scale')
    group.add_argument("--kd-ratio", type=float, default=None)

    group.add_argument('--warmup-iters', type=int, default=0,
                       help='percentage of data to warmup on (.01 = 1% of all '
                       'training iters). Default 0.01')
    group.add_argument('--warmup-ratio', type=float, default=0.0),
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay LR over,'
                       ' If None defaults to `--train-iters`*`--epochs`')
    group.add_argument('--lr-decay-style', type=str, default='noam',
                       choices=['constant', 'linear', 'cosine', 'exponential', 'noam', 'wrmup_cosine'],
                       help='learning rate decay function')
    group.add_argument("--scheduler-name", type=str, default="constant_trm")

    group.add_argument("--w-span-loss", type=float, default=1.0)

    return parser


def add_ppo_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('ppo', 'ppo configurations')
    
    group.add_argument("--reward-scaling", type=float, default=None)
    group.add_argument("--cliprange-reward", type=float, default=1)
    group.add_argument("--ppo-epochs", type=int, default=None)
    group.add_argument("--num-rollouts", type=int, default=256)
    group.add_argument("--num-rollouts-per-device", type=int, default=None)
    group.add_argument("--cliprange", type=float, default=0.2)
    group.add_argument("--chunk-size", type=int, default=None)
    group.add_argument("--gamma", type=float, default=0.95)
    
    return parser


def add_minillm_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('minillm', 'minillm configurations')
    
    group.add_argument("--length-norm", action="store_true")
    group.add_argument("--single-step-reg", action="store_true")
    group.add_argument("--teacher-mixed-alpha", type=float, default=None)
    group.add_argument("--lm-coef", type=float, default=1)
    
    return parser


def add_distillm_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('distillm', 'distillm configurations')

    # skew kld
    group.add_argument("--skew-alpha", type=float, default=0.1)

    # AMiD (alpha-mixture assistant distribution KD; aailab-kaist/AMiD defaults)
    group.add_argument("--amid-alpha", type=float, default=0.5)
    group.add_argument("--amid-lam", type=float, default=0.5)
    group.add_argument("--amid-div-name", type=str, default="fkl", choices=["fkl", "ab"])
    group.add_argument(
        "--amid-div-order", type=str, default="pr", choices=["pr", "qr", "rp", "rq"]
    )

    # student generation
    group.add_argument("--student-gen", action="store_true")
    group.add_argument("--gen-do-sample", action="store_true")
    group.add_argument("--gen-top-p", type=float, default=1.0)
    group.add_argument("--gen-num-beams", type=int, default=2)
    group.add_argument("--gen-temperature", type=float, default=1.0)
    
    # adaptive threshold
    group.add_argument("--mixed-alpha", type=float, default=0.5)
    group.add_argument("--loss-eps", type=float, default=0.1)
    group.add_argument("--init-threshold", type=float, default=0.0)
    
    # off-policy
    group.add_argument("--capacity", type=int, default=1000)
    group.add_argument("--replay-ratio", type=str, default="decreasing")
    # group.add_argument("--time", action="store_true")

    group.add_argument("--student_layer_mapping", nargs='+', type=int, default=[-1])
    group.add_argument("--teacher_layer_mapping", nargs='+', type=int, default=[-1])
    group.add_argument("--split_layer_mapping", nargs='+', type=int, default=[0, 0, 0, 0])
    group.add_argument("--span_metric", type=str, default="cosine", choices=["cosine", "dot", "l2", "cka"],
                       help="span relational distance: cosine | dot | l2(euclidean) | cka "
                            "(linear CKA per batch item; drops the omega_ik weighting)")
    group.add_argument("--use_dsa", action="store_true")
    group.add_argument("--use_hs", action="store_true")

    return parser


def add_gen_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('generation', 'generation configurations')
    
    group.add_argument("--top-k", type=int, default=0)
    group.add_argument("--top-p", type=float, default=1.0)
    group.add_argument("--do-sample", action="store_true")
    group.add_argument("--no-repeat-ngram-size", type=int, default=6)
    group.add_argument("--repetition-penalty", type=float, default=None)
    group.add_argument("--num-beams", type=int, default=1)
    group.add_argument("--temperature", type=float, default=1)
    group.add_argument("--strict-generation", action="store_true",
                       help="use every GenerationConfig as written; by default transformers fills fields "
                            "left at their library default from the model's generation_config.json (see "
                            "gen_config.py), which is how all runs so far were made")
    group.add_argument("--gen-backend", choices=["hf", "vllm"], default="hf",
                       help="hf: answers come from model.generate(); vllm: from tools/vllm_generate.py in the "
                            "vLLM environment, with the settings generate() would use (gen_backend.py)")

    return parser


def add_peft_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('generation', 'generation configurations')
    
    group.add_argument("--peft", type=str, default=None)
    group.add_argument("--peft-lora-r", type=int, default=16)
    group.add_argument("--peft-lora-alpha", type=int, default=64)
    group.add_argument("--peft-lora-dropout", type=float, default=0.1)
    group.add_argument("--peft-name", type=str, default=None)
    group.add_argument("--peft-path", type=str, default=None)
    group.add_argument("--teacher-peft-name", type=str, default=None)
    group.add_argument("--teacher-peft-path", type=str, default=None)
    return parser


def add_ced_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('ced', 'continual event detection configurations')

    group.add_argument("--ced-streams-file", type=str, default=None,
                       help="json list of per-task event-type lists, in this run's task order")
    group.add_argument("--ced-task-id", type=int, default=None,
                       help="current task index in the sequence (0-based)")
    group.add_argument("--ced-replay-mode", type=str, default="ce_kd", choices=["ce_kd", "kd_only"],
                       help="replay batches: CE + KD, or KD only")
    group.add_argument("--ced-kd-scope", type=str, default="replay", choices=["replay", "pl"],
                       help="KD coverage: replay rows only, or also old-event tokens of pseudo-labeled rows")
    group.add_argument("--ced-kd-ratio-new", type=float, default=0.0,
                       help="LwF: KD weight on the old teacher's logits over NEW-task rows' non-new-type "
                            "tokens (0 = off). Added on top of the replay KD, not stealing from CE.")
    group.add_argument("--ced-kd-new-cap", type=float, default=0.3,
                       help="hard cap on --ced-kd-ratio-new")
    group.add_argument("--ced-smoke-rows", type=int, default=0,
                       help="test only: balanced number of new/replay train rows (0 = full data)")
    # on-policy self-distillation (SDFT): EMA teacher reads the augmented answer in its prompt
    group.add_argument("--ced-sd", action="store_true",
                       help="add on-policy self-distillation (needs teacher prompts from tools/ced_sd_prompts.py)")
    group.add_argument("--ced-sd-weight", type=float, default=1.0,
                       help="weight of the SD loss, added on top of the existing objective")
    group.add_argument("--ced-sd-ema-mu", type=float, default=0.99,
                       help="EMA teacher decay per optimizer step (SDFT: 0.99)")
    group.add_argument("--ced-sd-temperature", type=float, default=1.0,
                       help="sampling temperature of the on-policy response")
    group.add_argument("--ced-sd-div", type=str, default="fkl", choices=["fkl", "rkl"],
                       help="fkl = KL(teacher||student) (SDFT code default), rkl = KL(student||teacher)")
    group.add_argument("--ced-sd-omission-mask", action="store_true",
                       help="drop from the SD loss the tokens of sampled old-type records that are "
                            "grounded in the sentence but absent from the reference y~ (ced_omask.py)")
    group.add_argument("--ced-sd-mix", type=str, default="sum", choices=["sum", "random"],
                       help="sum = token KD and SD on every step; random = each optimizer step keeps "
                            "one of the two KL terms (p = 0.5). Span loss and CE are unchanged.")
    group.add_argument("--ced-sd-probe", type=int, default=64,
                       help="rows used to check, before training, that the teacher copies the reference (0 = off)")
    # Ablation knobs. The defaults reproduce SDFT; none of these is part of the method.
    # (Samples cut off by the token budget are always dropped from the SD loss, no flag.)
    group.add_argument("--ced-sd-top-p", type=float, default=1.0,
                       help="nucleus p of the on-policy sample (SDFT: 1.0)")
    group.add_argument("--ced-sd-skip-unparsed", action="store_true",
                       help="also drop samples that are not valid JSON from the SD loss")
    group.add_argument("--ced-sd-warmup", type=float, default=0.0,
                       help="fraction of total_iters to train before SD starts (0 = from step 1)")
    # CL-LoRA baselines (cl_lora/): which method + its reg/ratio knobs
    group.add_argument("--cl-method", type=str, default=None,
                       choices=["inclora", "olora", "migu", "tree",
                                "inflora", "gainlora_o", "gainlora_inf", "epi"])
    group.add_argument("--cl-reg", type=float, default=0.5,
                       help="O-LoRA orth lambda / TreeLoRA reg coefficient")
    group.add_argument("--cl-migu-ratio", type=float, default=0.7)
    group.add_argument("--cl-data-root", type=str, default=None,
                       help="dir with per-task subdirs 0/qwen ... N-1/qwen (e.g. processed_data/ace_b10_perm0)")
    group.add_argument("--cl-num-tasks", type=int, default=5)
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    parser = add_runtime_args(parser)
    parser = add_data_args(parser)
    parser = add_hp_args(parser)
    parser = add_ppo_args(parser)
    parser = add_minillm_args(parser)
    parser = add_distillm_args(parser)
    parser = add_gen_args(parser)
    parser = add_peft_args(parser)
    parser = add_ced_args(parser)
    parser = deepspeed.add_config_arguments(parser)
    
    args, unknown = parser.parse_known_args()
    
    assert all(["--" not in x for x in unknown]), unknown
    
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        
    args.n_gpu = args.n_gpu * args.n_nodes
        
    if args.type == "eval_main":
        ckpt_name = None
        if args.ckpt_name is not None:
            ckpt_name = args.ckpt_name
        if args.peft_name is not None:
            ckpt_name = args.peft_name

        if ckpt_name is not None:
            tmp = ckpt_name.split("/")
            if tmp[-1].isdigit():
                ckpt_name = "_".join(tmp[:-1]) + "/" + tmp[-1]
            else:
                ckpt_name = "_".join(tmp)

        save_path = os.path.join(
            args.save,
            f"{args.data_names}-{args.max_length}" + (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else ""),
            ckpt_name,
            f"{args.seed}",
        )
        args.save = save_path
    elif args.type == "lm":
        save_path = os.path.join(
            args.save,
            (f"{args.ckpt_name}" + f"-{args.peft_name}" if args.peft_name is not None else ""),
            (f"e{args.epochs}-bs{args.batch_size}-lr{args.lr}-G{args.gradient_accumulation_steps}-N{args.n_gpu}-NN{args.n_nodes}") + \
            (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else "") + \
            (f"-lora-{args.peft_lora_r}-{args.peft_lora_alpha}-{args.peft_lora_dropout}" if args.peft == "lora" else "") + \
            args.save_additional_suffix
        )
        args.save = save_path
    elif args.type == "kd":
        save_path = os.path.join(
            args.save,
            (f"{args.ckpt_name}" + f"-{args.peft_name}" if args.peft_name is not None else "" + \
             f"-{args.teacher_ckpt_name}" + f"-{args.teacher_peft_name}" if args.teacher_peft_name is not None else ""),
            (f"e{args.epochs}-bs{args.batch_size}-lr{args.lr}-G{args.gradient_accumulation_steps}-N{args.n_gpu}-NN{args.n_nodes}-kd{args.kd_ratio}") + \
            (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else "") + \
            (f"-lora-{args.peft_lora_r}-{args.peft_lora_alpha}-{args.peft_lora_dropout}" if args.peft == "lora" else "") + \
            args.save_additional_suffix
        )
        args.save = save_path
    elif args.type == "gen":
        save_path = os.path.join(
            args.save,
            (f"{args.ckpt_name}"),
            (f"t{args.temperature}-l{args.max_length}"),
        )
        args.save = save_path
    elif args.type == "minillm":
        ppo_prefix = f"pe{args.ppo_epochs}" + \
                     (f"_rs{args.reward_scaling}" if args.ppo_epochs is not None else "") + \
                     (f"_nr{args.num_rollouts}" if args.num_rollouts is not None else "") + \
                     (f"_ln" if args.length_norm else "") + \
                     (f"_sr" if args.single_step_reg else "") + \
                     (f"_tm{args.teacher_mixed_alpha}" if args.teacher_mixed_alpha is not None else "")
        save_path = os.path.join(
            args.save,
            (f"{args.ckpt_name}" + f"-{args.peft_name}" if args.peft_name is not None else "" + \
             f"-{args.teacher_ckpt_name}" + f"-{args.teacher_peft_name}" if args.teacher_peft_name is not None else ""),
            (f"bs{args.batch_size}-lr{args.lr}-G{args.gradient_accumulation_steps}-N{args.n_gpu}-NN{args.n_nodes}-lm{args.lm_coef}-len{args.max_length}" + \
                (f"-mp{args.model_parallel_size}" if args.model_parallel > 0 else "")) + \
            (f"-lora-{args.peft_lora_r}-{args.peft_lora_alpha}-{args.peft_lora_dropout}" if args.peft == "lora" else ""),
            ppo_prefix + args.save_additional_suffix
        )
        args.save = save_path
        args.num_rollouts_per_device = args.num_rollouts // args.n_gpu
        
        if args.warmup_iters > 0:
            assert args.scheduler_name is not None

    return args
