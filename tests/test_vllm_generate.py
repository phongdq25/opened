"""tools/vllm_generate.py's helpers. vllm itself is imported in main() only; Task 2's GPU smoke
test runs the whole program."""
import importlib.util
import os
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location("vllm_generate", os.path.join("tools", "vllm_generate.py"))
vg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vg)
PARAMS = {"temperature": 0.5, "top_p": 0.95, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
          "stop_token_ids": [151645, 151643], "logprobs": False}


@pytest.mark.parametrize("rank,allowed", [(1, 1), (4, 8), (16, 16), (17, 32), (144, 256), (160, 256), (512, 512)])
def test_the_lora_rank_rounds_up_to_one_vllm_accepts(rank, allowed):
    assert vg.allowed_lora_rank(rank) == allowed


def test_a_lora_rank_above_vllms_maximum_is_refused():
    with pytest.raises(ValueError, match="above vLLM's maximum"):
        vg.allowed_lora_rank(600)


def test_each_request_keeps_its_own_cap_and_seed():
    assert vg.sampling_kwargs(PARAMS, {"prompt_token_ids": [1], "max_tokens": 308, "seed": 9}) == dict(
        temperature=0.5, top_p=0.95, top_k=0, min_p=0.0, repetition_penalty=1.0,
        stop_token_ids=[151645, 151643], max_tokens=308, seed=9, detokenize=False)


def test_log_probabilities_ask_for_the_chosen_token_only():
    kwargs = vg.sampling_kwargs({**PARAMS, "logprobs": True}, {"prompt_token_ids": [1], "max_tokens": 4, "seed": 0})
    assert kwargs["logprobs"] == 0


@pytest.mark.parametrize("gpu_gb,kv_gib", [(10.0, 6.0), (6.0, 2.0), (3.0, 1.0)])
def test_the_kv_cache_has_a_fixed_size_on_a_shared_card(gpu_gb, kv_gib):
    """vLLM sizes the cache by profiling free memory, which other jobs on the card move; a fixed size
    leaves --gpu-memory-gb minus about 4 GiB (weights, activations, graphs) for it, 1 GiB at least."""
    kwargs = vg.engine_kwargs(SimpleNamespace(model="m", lora=None, max_lora_rank=16, max_model_len=768,
                                              gpu_memory_gb=gpu_gb, enforce_eager=False), total_gb=140.0)
    assert kwargs["kv_cache_memory_bytes"] == int(kv_gib * 2 ** 30)
    assert kwargs["gpu_memory_utilization"] == pytest.approx(gpu_gb / 140.0)
    assert (kwargs["skip_tokenizer_init"], kwargs["generation_config"], kwargs["enable_lora"]) == (True, "vllm", False)


def test_an_output_line_holds_the_tokens_and_their_log_probabilities():
    completion = SimpleNamespace(token_ids=[7, 151645], logprobs=[{7: SimpleNamespace(logprob=-0.25)},
                                                                  {151645: SimpleNamespace(logprob=-0.01)}])
    assert vg.output_record(completion, False) == {"token_ids": [7, 151645]}
    assert vg.output_record(completion, True) == {"token_ids": [7, 151645], "logprobs": [-0.25, -0.01]}
