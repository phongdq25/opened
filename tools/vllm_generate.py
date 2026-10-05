"""One batch of generation in vLLM, run by gen_backend.run_vllm in the vLLM environment.

Token IDs in, token IDs out: no tokenizer is loaded (skip_tokenizer_init), and the model's own
generation defaults are ignored (generation_config="vllm"). Every setting therefore comes from
--params, which gen_backend.vllm_params built from what transformers' generate() would use.
vllm is imported in main() only, so the helpers import in any environment."""
import argparse
import json

LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)   # vllm.config.lora.MaxLoRARanks in 0.27.1


def allowed_lora_rank(rank):
    """The smallest --max-lora-rank vLLM accepts that holds an adapter of this rank."""
    for allowed in LORA_RANKS:
        if allowed >= rank:
            return allowed
    raise ValueError(f"LoRA rank {rank} is above vLLM's maximum {LORA_RANKS[-1]}")


def sampling_kwargs(params, request):
    """vllm.SamplingParams arguments for one request: the shared settings plus the row's cap and seed."""
    kwargs = dict(temperature=params["temperature"], top_p=params["top_p"], top_k=params["top_k"],
                  min_p=params["min_p"], repetition_penalty=params["repetition_penalty"],
                  stop_token_ids=params["stop_token_ids"], max_tokens=request["max_tokens"],
                  seed=request["seed"], detokenize=False)
    if params["logprobs"]:
        kwargs["logprobs"] = 0          # the chosen token's log-probability, nothing else
    return kwargs


def engine_kwargs(a, total_gb):
    """vllm.LLM arguments. The KV cache gets a fixed size: vLLM would otherwise size it by profiling
    free memory, which the other jobs on a shared card move while it measures (it then finds no room
    at all, or counts their freed memory as its own). --gpu-memory-gb minus about 4 GiB for weights,
    activations and CUDA graphs goes to the cache, 1 GiB at least."""
    return dict(model=a.model, skip_tokenizer_init=True, dtype="bfloat16", seed=0, generation_config="vllm",
                max_model_len=a.max_model_len, gpu_memory_utilization=min(a.gpu_memory_gb / total_gb, 0.9),
                kv_cache_memory_bytes=int(max(1.0, a.gpu_memory_gb - 4.0) * 2 ** 30),
                enforce_eager=a.enforce_eager, enable_lora=a.lora is not None,
                max_lora_rank=allowed_lora_rank(a.max_lora_rank), max_loras=1)


def output_record(completion, logprobs):
    """One line of --out: the generated token IDs, and their log-probabilities when asked."""
    record = {"token_ids": list(completion.token_ids)}
    if logprobs:
        record["logprobs"] = [step[token].logprob for token, step in zip(completion.token_ids, completion.logprobs)]
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="full model directory: the base, or a merged model")
    ap.add_argument("--lora", default=None, help="PEFT adapter directory, applied unmerged")
    ap.add_argument("--max-lora-rank", type=int, default=16)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-model-len", type=int, required=True)
    ap.add_argument("--gpu-memory-gb", type=float, default=10.0)
    ap.add_argument("--enforce-eager", action="store_true")
    a = ap.parse_args()
    requests = [json.loads(line) for line in open(a.requests)]
    params = json.load(open(a.params))

    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    llm = LLM(**engine_kwargs(a, torch.cuda.get_device_properties(0).total_memory / 2 ** 30))
    results = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_token_ids"]) for r in requests],
                           [SamplingParams(**sampling_kwargs(params, r)) for r in requests],
                           lora_request=LoRARequest("eval", 1, a.lora) if a.lora else None, use_tqdm=False)
    with open(a.out, "w") as f:
        for result in results:
            f.write(json.dumps(output_record(result.outputs[0], params["logprobs"])) + "\n")


if __name__ == "__main__":
    main()
