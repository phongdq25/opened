"""Stands in for tools/vllm_generate.py in the CPU tests: the same command line, no vLLM.

FAKE_VLLM_MODE picks the answers:
  checksum (default)  [100 + sum(prompt) % 1000, the first stop token] per request,
                      with log-probabilities [-0.5, -0.1] when asked
  replay              the records in the JSON file FAKE_VLLM_REPLAY, in order
  short               like checksum, one answer short
  fail                print "boom" and exit 3
FAKE_VLLM_LOG, when set, gets one JSON line per call: the arguments, the environment, the
requests, the params and the files in --lora."""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    for flag in ("--model", "--lora", "--requests", "--params", "--out"):
        ap.add_argument(flag, default=None)
    ap.add_argument("--max-lora-rank", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-memory-gb", type=float, default=None)
    ap.add_argument("--enforce-eager", action="store_true")
    a = ap.parse_args()
    requests = [json.loads(line) for line in open(a.requests)]
    params = json.load(open(a.params))
    if os.environ.get("FAKE_VLLM_LOG"):
        with open(os.environ["FAKE_VLLM_LOG"], "a") as f:
            f.write(json.dumps({"args": vars(a), "env": dict(os.environ), "requests": requests, "params": params,
                                "lora_files": sorted(os.listdir(a.lora)) if a.lora and os.path.isdir(a.lora)
                                else None}) + "\n")
    mode = os.environ.get("FAKE_VLLM_MODE", "checksum")
    if mode == "fail":
        print("boom")
        sys.exit(3)
    if mode == "replay":
        records = json.load(open(os.environ["FAKE_VLLM_REPLAY"]))
    else:
        records = []
        for request in requests:
            record = {"token_ids": [100 + sum(request["prompt_token_ids"]) % 1000, params["stop_token_ids"][0]]}
            if params.get("logprobs"):
                record["logprobs"] = [-0.5, -0.1]
            records.append(record)
        if mode == "short":
            records = records[:-1]
    with open(a.out, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
