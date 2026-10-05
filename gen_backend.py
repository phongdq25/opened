"""Generation in vLLM for evaluation and pseudo-labelling (spec:
docs/superpowers/specs/2026-10-06-vllm-generation-design.md).

The training environment builds the requests (each row's unpadded prompt token IDs, its length
cap and its sampling seed), asks transformers which settings generate() would really decode
with, and decodes the answers with today's code. tools/vllm_generate.py runs in the vLLM
environment and turns token IDs into token IDs. torch and transformers are imported inside the
functions, so `python -c "from gen_backend import find_vllm_python"` works in any environment."""
import copy
import hashlib
import json
import os
import subprocess

REPO = os.path.dirname(os.path.abspath(__file__))
VLLM_SCRIPT = os.path.join(REPO, "tools", "vllm_generate.py")
VLLM_CANDIDATES = (os.path.join(REPO, ".venv-vllm", "bin", "python"), "/venv/main/bin/python")
EAGER_DEFAULT = "0"      # VLLM_EAGER unset: "1" = no CUDA graphs (decided by measurement, plan Task 7)
# torchrun sets these; a vLLM child that inherits them can try to join the trainer's process group
TORCHRUN_VARS = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "GROUP_WORLD_SIZE",
                 "ROLE_NAME", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
# GenerationConfig fields vLLM has no equivalent for, each with the value that means "not used"
UNSUPPORTED = {
    "num_beams": 1, "num_beam_groups": 1, "penalty_alpha": None, "no_repeat_ngram_size": 0,
    "encoder_no_repeat_ngram_size": 0, "bad_words_ids": None, "force_words_ids": None,
    "typical_p": 1.0, "epsilon_cutoff": 0.0, "eta_cutoff": 0.0, "diversity_penalty": 0.0,
    "exponential_decay_length_penalty": None, "suppress_tokens": None, "begin_suppress_tokens": None,
    "sequence_bias": None, "min_new_tokens": None, "forced_bos_token_id": None,
    "forced_eos_token_id": None, "renormalize_logits": False, "guidance_scale": None, "dola_layers": None,
}


class UnsupportedForVLLM(ValueError):
    """A generation setting vLLM cannot reproduce."""


def hf_model(model):
    """The transformers model inside a DeepSpeed/DDP wrapper (.module) and a PEFT model."""
    model = getattr(model, "module", model)
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def resolve_generation_config(model, generation_config=None, **kwargs):
    """The settings model.generate(generation_config=generation_config, **kwargs) decodes with,
    after transformers fills in the model's own defaults (gen_config.py). This is transformers'
    own resolution; 4.57.3 is pinned and tests/test_gen_backend.py checks what it gives."""
    config, _ = hf_model(model)._prepare_generation_config(copy.deepcopy(generation_config), **kwargs)
    return config


def vllm_params(config, logprobs=False):
    """tools/vllm_generate.py's --params for a resolved GenerationConfig."""
    for field, unused in UNSUPPORTED.items():
        value = getattr(config, field, unused)
        if value != unused and value not in ([], ()):
            raise UnsupportedForVLLM(f"generation setting {field}={value!r} has no vLLM equivalent")
    if (config.min_length or 0) > 0:
        raise UnsupportedForVLLM(f"generation setting min_length={config.min_length} has no vLLM equivalent")
    eos = config.eos_token_id
    stop = [eos] if isinstance(eos, int) else list(eos or [])
    if not stop:
        raise UnsupportedForVLLM("no eos_token_id: vLLM could not tell where an answer ends")
    params = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
              "stop_token_ids": stop, "logprobs": logprobs}
    if config.do_sample:
        params.update(temperature=float(config.temperature), top_p=float(config.top_p or 1.0),
                      top_k=int(config.top_k or 0), min_p=float(config.min_p or 0.0))
    if config.repetition_penalty is not None:
        params["repetition_penalty"] = float(config.repetition_penalty)
    return params


def unpadded_prompts(input_ids, attention_mask):
    """Each row's prompt as generate() reads it: the tokens its attention mask keeps. generate()
    gives masked tokens no position and no attention, so dropping them computes the same thing."""
    return [ids[mask.bool()].tolist() for ids, mask in zip(input_ids.cpu(), attention_mask.cpu())]


def row_seed(seed, split, epoch, index):
    """A row's sampling seed: the same evaluation of the same run draws the same answers."""
    digest = hashlib.sha256(f"{seed}/{split}/{epoch}/{index}".encode()).digest()
    return int.from_bytes(digest[:4], "little") % 2 ** 31


def child_env(cuda=True):
    """The environment of a vLLM process: without torchrun's variables and the repo's PYTHONPATH."""
    env = {k: v for k, v in os.environ.items()
           if k not in TORCHRUN_VARS and k != "PYTHONPATH" and not k.startswith("TORCHELASTIC_")}
    env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    env["VLLM_NO_USAGE_STATS"] = "1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"      # engine in-process: one short-lived process per call
    if not cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def find_vllm_python():
    """(interpreter, vLLM version) of the vLLM environment: VLLM_PY when it is set (no fallback),
    else the first of VLLM_CANDIDATES that imports vllm."""
    candidates = [os.environ["VLLM_PY"]] if os.environ.get("VLLM_PY") else list(VLLM_CANDIDATES)
    for python in candidates:
        if not os.path.exists(python):
            continue
        out = subprocess.run([python, "-c", "import vllm; print('VLLM_VERSION', vllm.__version__)"],
                             capture_output=True, text=True, env=child_env(cuda=False), timeout=600)
        versions = [line.split()[1] for line in out.stdout.splitlines() if line.startswith("VLLM_VERSION ")]
        if out.returncode == 0 and versions:
            return python, versions[-1]
    raise RuntimeError(f"no vLLM environment (tried {', '.join(candidates)}): set VLLM_PY, or create one with "
                       "`uv venv .venv-vllm --python 3.12 && uv pip install --python .venv-vllm vllm==0.27.1`")


def run_vllm(work_dir, model_dir, requests, params, lora_dir=None, max_lora_rank=16, max_model_len=None,
             vllm_py=None):
    """Answer `requests` ({"prompt_token_ids", "max_tokens", "seed"} each) with tools/vllm_generate.py.
    Returns one {"token_ids": [...]} per request, in order, with "logprobs" when params asks for them.
    On failure the files stay in work_dir and the error carries the end of the vLLM log."""
    if not requests:
        return []
    os.makedirs(work_dir, exist_ok=True)
    paths = {name: os.path.join(work_dir, name)
             for name in ("requests.jsonl", "params.json", "outputs.jsonl", "vllm.log")}
    with open(paths["requests.jsonl"], "w") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")
    with open(paths["params.json"], "w") as f:
        json.dump(params, f)
    max_model_len = max_model_len or max(len(r["prompt_token_ids"]) + r["max_tokens"] for r in requests)
    cmd = [vllm_py or find_vllm_python()[0], VLLM_SCRIPT, "--model", model_dir,
           "--requests", paths["requests.jsonl"], "--params", paths["params.json"],
           "--out", paths["outputs.jsonl"], "--max-model-len", str(max_model_len),
           "--gpu-memory-gb", os.environ.get("VLLM_GPU_GB", "10")]
    if lora_dir is not None:
        cmd += ["--lora", lora_dir, "--max-lora-rank", str(max_lora_rank)]
    if os.environ.get("VLLM_EAGER", EAGER_DEFAULT) == "1":
        cmd.append("--enforce-eager")
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()          # hand the training step's cached blocks back to the card
    except ImportError:
        pass
    with open(paths["vllm.log"], "w") as log:
        returncode = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=child_env()).returncode
    if returncode != 0:
        tail = open(paths["vllm.log"], errors="replace").read().splitlines()[-50:]
        raise RuntimeError(f"vLLM generation failed (exit {returncode}); last lines of {paths['vllm.log']}:\n"
                           + "\n".join(tail))
    outputs = [json.loads(line) for line in open(paths["outputs.jsonl"])]
    if len(outputs) != len(requests):
        raise RuntimeError(f"vLLM returned {len(outputs)} answers for {len(requests)} requests "
                           f"({paths['outputs.jsonl']})")
    for path in paths.values():
        os.remove(path)
    return outputs


def export_for_vllm(model, base_path, out_dir):
    """What vLLM loads for the model a trainer evaluates: (model_dir, lora_dir, max_lora_rank).
    A PEFT model is saved as its adapter and applied, unmerged, on base_path (the model the
    trainer started from), as PEFT does; any other model is saved whole."""
    inner = getattr(model, "module", model)
    if hasattr(inner, "peft_config"):
        lora_dir = os.path.join(out_dir, "adapter")
        inner.save_pretrained(lora_dir, safe_serialization=True)
        return base_path, lora_dir, max(config.r for config in inner.peft_config.values())
    model_dir = os.path.join(out_dir, "model")
    inner.save_pretrained(model_dir, safe_serialization=True)
    return model_dir, None, 0


def meta_model(model_dir):
    """A model without weights whose generation settings resolve like model_dir's."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig
    config = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_dir)
    except OSError:
        model.generation_config = GenerationConfig.from_model_config(config)
    return model
