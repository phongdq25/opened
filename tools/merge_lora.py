"""Merge a trained LoRA adapter into its base model and save the full model.

Used by the CED task loop: the merged model of task t becomes the init (and
later the KD teacher) for task t+1.

Usage:
    python tools/merge_lora.py --base-model-path <dir|hf-id> --peft-path <adapter_dir> --out <dir>
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--peft-path", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"loading base: {args.base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16)
    print(f"loading adapter: {args.peft_path}")
    model = PeftModel.from_pretrained(model, args.peft_path)
    model = model.merge_and_unload()
    print(f"saving merged model to: {args.out}")
    model.save_pretrained(args.out)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.save_pretrained(args.out)
    print("done")


if __name__ == "__main__":
    main()
