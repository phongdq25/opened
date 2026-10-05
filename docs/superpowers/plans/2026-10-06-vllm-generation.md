# vLLM Evaluation and Pseudo-labelling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate evaluation answers and teacher pseudo-labels with vLLM behind an opt-in switch
(`--gen-backend vllm`, runner setting `GEN_BACKEND=vllm`). vLLM gets the same decoding settings,
prompts and text handling as Hugging Face `generate()`. The H200 defaults switch to vLLM once an
end-to-end TACRED check passes.

**Architecture:** The training environment builds the requests: each row's unpadded prompt
token IDs, its length cap and its sampling seed. It resolves the decoding settings with
transformers' own `_prepare_generation_config` and maps them to vLLM sampling parameters. It then
exports the evaluated model in one of three forms:

- a LoRA adapter on the starting model;
- one concatenated adapter, for CL-LoRA's summed adapters;
- the whole model.

`tools/vllm_generate.py` runs in the separate vLLM environment as a subprocess and returns token
IDs. Decoding, parsing, scoring and logging stay in today's code.

**Tech Stack:**

- Training environment: Python 3.11, PyTorch 2.9.1, transformers 4.57.3, peft 0.18.0.
- vLLM 0.27.1 in its own environment (`/venv/main`, Python 3.12, PyTorch 2.13).
- pytest 8.3.3 and the bash runners.

**Spec:** `docs/superpowers/specs/2026-10-06-vllm-generation-design.md`

## Global Constraints

- **Same decoding settings.** vLLM receives exactly the settings Hugging Face would have used,
  after transformers fills in Qwen3's defaults: the same temperature, top-p, top-k, stop tokens
  and per-row length cap. Greedy stays greedy.
- **Same prompts and same text handling.** vLLM receives the token IDs the Hugging Face path
  would have fed `generate()`, and returns token IDs. Decoding to text, parsing and scoring stay
  in the training environment with the same code as today.
- **Same evaluated model.** Base plus unmerged LoRA where the trainer uses LoRA.
- **Statistically equivalent, not bit-identical.**
- **Off by default.**
  - `--gen-backend hf` stays the code default.
  - The H200 defaults switch to vLLM only after the end-to-end check passes (Task 8).
- **Training environment untouched.** vLLM runs in its own Python environment, as a separate
  process.
- **The FewRel run in `/workspace/opened` is not touched.** All work happens in
  `/workspace/opened_dev`, which has its own `.venv`.
- **Git:**
  - Commit only on `perf/fast-generation`, and never push.
  - Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **vLLM 0.27.1 is the tested version.**

## Review Focus

Each of these is pinned by a test in the task named.

1. **A pseudo-labelling task with no candidate rows under `GEN_BACKEND=vllm`:**
   - no vLLM process starts;
   - the outputs are written as on the Hugging Face path.

   Tests: Task 1 `test_no_requests_start_no_vllm`; Task 5
   `test_a_task_without_candidates_starts_no_vllm`.
2. **A vLLM child inherits torchrun's `RANK`/`WORLD_SIZE`/`MASTER_PORT`.** They must be stripped.
   Test: Task 1 `test_run_vllm_round_trip_without_torchruns_variables`.
3. **vLLM crashes or returns fewer answers than rows.**
   - The run fails loudly with the end of the vLLM log.
   - The files stay for inspection.

   Tests: Task 1 `test_a_short_answer_file_fails_loudly` and
   `test_a_vllm_crash_fails_with_its_last_log_lines_and_keeps_them`.
4. **A run resumed under the other backend is refused.** Tests: Task 4
   `test_a_resume_under_the_other_backend_is_refused`; Task 6
   `test_the_ced_runner_records_the_backend_in_its_manifest`.
5. **`GEN_BACKEND=vllm` on a host without vLLM is refused before any run directory exists.**
   Tests: Task 4 `test_a_vllm_run_without_vllm_is_refused_before_its_run_directory_exists`;
   Task 6 `test_runners_refuse_vllm_without_an_environment_before_any_run_directory`.

## Working on the box

All tests and GPU runs happen on the H200 box, in `/workspace/opened_dev`. Code is edited on the
Mac and synced. Define these once per shell:

```bash
REPO=/Users/kienvu/Desktop/research-papers/opened
BOX="ssh -p 40292 -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 root@5.145.175.100"
DEV=/workspace/opened_dev
sync_dev () {
    rsync -az --no-o --no-g --delete \
        --exclude /.venv --exclude /.venv-vllm --exclude /data --exclude /processed_data --exclude /models \
        --exclude /results --exclude /logs --exclude /.mps --exclude /bench --exclude /.superpowers \
        -e "ssh -p 40292 -o BatchMode=yes" "${REPO}/" root@5.145.175.100:${DEV}/
}
dev_pytest () {
    sync_dev && ${BOX} "cd ${DEV} && CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        .venv/bin/python -m pytest -q -p no:cacheprovider $*"
}
DEV_GPU_ENV="cd ${DEV} && export ENV_BIN=${DEV}/.venv/bin PY=${DEV}/.venv/bin/python HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/usr/local/cuda DISK_PATH=. \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 VLLM_PY=/venv/main/bin/python"
```

Rules on the box:

- **Never write under `/workspace/opened`.** That is the FewRel run's tree.
- **Symlinks:**
  - `data/`, `processed_data/` and `models/` in the dev tree are symlinks to the FewRel tree.
    Read them; never write into them.
  - The checks write to `/workspace/vllm_checks/`.
- **Development GPU processes do not set `CUDA_MPS_*`.**
  - They run outside the FewRel run's MPS server, so a crash cannot reach its jobs.
  - A process outside MPS was checked on 2026-10-05 to run fine next to that server.
- **Round-1 artifacts the checks use**, under `/workspace/dev_runs/results_2026-10-05/qwen3/ced/`:
  - `cllora_inclora_perm0_tacred_cre_s42/checkpoint_latest.pt`: IncLoRA, TACRED order 0, all
    10 tasks. Task-9 trigger F1 was 0.6306.
  - `bench_t0_fewrel/task0/e1-bs16-lr0.0002-G2-N1-NN1-lora-16-64-0.1/40/`: a FewRel task-0 LoRA
    adapter on `models/Qwen3-0.6B`.
  - `bench_t0_ace/task0/merged/`: an ACE task-0 merged model, used as a pseudo-label teacher.

---

### Task 1: Dev tree on the box, and `gen_backend.py`

**Files:**
- Create: `gen_backend.py`, `tests/fake_vllm.py`, `tests/test_gen_backend.py`
- Modify: `tests/conftest.py` (add the `fake_vllm` fixture)

**Interfaces:**
- Produces, in `gen_backend.py`:
  - constants `VLLM_SCRIPT: str`, `VLLM_CANDIDATES: tuple[str, ...]`, `TORCHRUN_VARS`,
    `UNSUPPORTED: dict`, `EAGER_DEFAULT = "0"`;
  - `class UnsupportedForVLLM(ValueError)`;
  - `hf_model(model)`: the transformers model inside `.module` and PEFT wrappers;
  - `resolve_generation_config(model, generation_config=None, **kwargs) -> GenerationConfig`;
  - `vllm_params(config, logprobs=False) -> dict`, with keys `temperature`, `top_p`, `top_k`,
    `min_p`, `repetition_penalty`, `stop_token_ids`, `logprobs`;
  - `unpadded_prompts(input_ids, attention_mask) -> list[list[int]]`;
  - `row_seed(seed, split, epoch, index) -> int`;
  - `child_env(cuda=True) -> dict`;
  - `find_vllm_python() -> tuple[str, str]`: the interpreter and the vLLM version. It raises
    `RuntimeError("no vLLM environment ...")`;
  - `run_vllm(work_dir, model_dir, requests, params, lora_dir=None, max_lora_rank=16,
    max_model_len=None, vllm_py=None) -> list[dict]`. A request is
    `{"prompt_token_ids", "max_tokens", "seed"}`; a result is `{"token_ids"[, "logprobs"]}`;
  - `export_for_vllm(model, base_path, out_dir) -> tuple[str, str | None, int]`: the model
    directory, the adapter directory, and the maximum LoRA rank;
  - `meta_model(model_dir)`: a weightless model with `model_dir`'s generation config.
- Produces, in `tests/conftest.py`: the fixture `fake_vllm`, an object with:
  - `.python`: a `VLLM_PY` that reports vLLM 0.27.1 and runs `tests/fake_vllm.py`;
  - `.calls() -> list[dict]`: each call's `args`, `env`, `requests`, `params` and `lora_files`.
- Produces, in `tests/fake_vllm.py`: `tools/vllm_generate.py`'s command line.
  `FAKE_VLLM_MODE` is one of:
  - `checksum` (the default): returns `[100 + sum(prompt) % 1000, stop_token_ids[0]]`, with
    logprobs `[-0.5, -0.1]`;
  - `replay`: returns the records in `FAKE_VLLM_REPLAY`;
  - `short`;
  - `fail`.

- [ ] **Step 1: Create the dev tree (one time)**

```bash
${BOX} 'set -e; mkdir -p /workspace/opened_dev /workspace/vllm_checks; cd /workspace/opened_dev
for d in data processed_data models; do [ -e $d ] || ln -s /workspace/opened/$d $d; done
mkdir -p results logs'
sync_dev
${BOX} 'cd /workspace/opened_dev && uv sync && uv pip install --python .venv/bin/python pytest==8.3.3 \
    && .venv/bin/python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)"'
```
Expected: the last line reads `2.9.1+cu128 4.57.3 0.18.0`.

- [ ] **Step 2: Write `tests/fake_vllm.py`**

```python
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
```

- [ ] **Step 3: Add the `fake_vllm` fixture to `tests/conftest.py`**

Add `import json` to the imports. After the `ced_args` function, add:

```python
FAKE_VLLM = os.path.join(REPO, "tests", "fake_vllm.py")


class FakeVLLM:
    """What the fake_vllm fixture hands a test: the VLLM_PY to pass on, and the recorded calls."""

    def __init__(self, python, log):
        self.python, self.log = python, log

    def calls(self):
        if not os.path.exists(self.log):
            return []
        return [json.loads(line) for line in open(self.log)]


@pytest.fixture
def fake_vllm(monkeypatch, tmp_path):
    """gen_backend runs tests/fake_vllm.py in place of tools/vllm_generate.py, through a VLLM_PY
    that tells find_vllm_python() it holds vLLM 0.27.1."""
    import gen_backend
    python = tmp_path / "vllm_python"
    python.write_text('#!/bin/bash\nif [ "$1" = "-c" ]; then echo "VLLM_VERSION 0.27.1"; exit 0; fi\n'
                      f'exec {sys.executable} "$@"\n')
    python.chmod(0o755)
    monkeypatch.setattr(gen_backend, "VLLM_SCRIPT", FAKE_VLLM)
    monkeypatch.setenv("VLLM_PY", str(python))
    monkeypatch.setenv("FAKE_VLLM_LOG", str(tmp_path / "fake_vllm.log"))
    return FakeVLLM(str(python), str(tmp_path / "fake_vllm.log"))
```

- [ ] **Step 4: Write the failing tests, `tests/test_gen_backend.py`**

```python
"""gen_backend.py: which settings vLLM gets, what it is asked, and how its process runs."""
import os

import pytest
import torch
from transformers import GenerationConfig

from conftest import LORA_TARGETS, EngineShim, tiny_qwen

QWEN_EOS = [151645, 151643]
# what evaluate() builds under --greedy 1 (runner flags --top-k 0 --top-p 0.95 --temperature 0.5)
EVAL = dict(do_sample=False, temperature=0.5, top_p=0.95, top_k=0, repetition_penalty=None,
            eos_token_id=QWEN_EOS, pad_token_id=151645)
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
          "stop_token_ids": QWEN_EOS, "logprobs": False}


def request(prompt, max_tokens=4, seed=0):
    return {"prompt_token_ids": prompt, "max_tokens": max_tokens, "seed": seed}


def test_todays_evaluation_resolves_to_sampling_through_the_wrappers(student):
    from gen_backend import resolve_generation_config, vllm_params
    g = resolve_generation_config(EngineShim(student), GenerationConfig(**EVAL))
    assert (g.do_sample, g.temperature, g.top_p, g.top_k) == (True, 0.5, 0.95, 0)   # the Qwen3 fill-in
    assert vllm_params(g) == {**GREEDY, "temperature": 0.5, "top_p": 0.95}


def test_strict_generation_resolves_to_greedy():
    from gen_backend import resolve_generation_config, vllm_params
    g = resolve_generation_config(tiny_qwen(0), GenerationConfig(**EVAL), use_model_defaults=False)
    assert g.do_sample is False
    assert vllm_params(g) == GREEDY


def test_cl_lora_and_pseudo_label_calls_resolve_to_greedy_with_the_models_stop_tokens():
    from gen_backend import resolve_generation_config, vllm_params
    g = resolve_generation_config(tiny_qwen(0), None, do_sample=False, pad_token_id=151643)
    assert vllm_params(g, logprobs=True) == {**GREEDY, "logprobs": True}


@pytest.mark.parametrize("field,value", [("num_beams", 2), ("no_repeat_ngram_size", 6), ("typical_p", 0.9),
                                         ("min_length", 5), ("bad_words_ids", [[7]])])
def test_settings_vllm_cannot_reproduce_are_refused(field, value):
    from gen_backend import UnsupportedForVLLM, resolve_generation_config, vllm_params
    g = resolve_generation_config(tiny_qwen(0), GenerationConfig(**{**EVAL, field: value}))
    with pytest.raises(UnsupportedForVLLM, match=field):
        vllm_params(g)


def test_prompts_are_the_tokens_the_attention_mask_keeps():
    from gen_backend import unpadded_prompts
    ids = torch.tensor([[0, 0, 5, 6, 7], [0, 8, 9, 0, 10]])
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 1]])
    assert unpadded_prompts(ids, mask) == [[5, 6, 7], [8, 9, 10]]


def test_row_seeds_are_reproducible_and_distinct():
    from gen_backend import row_seed
    seeds = {row_seed(42, split, epoch, i) for split in ("dev", "test") for epoch in (0, 1) for i in range(100)}
    assert len(seeds) == 400 and all(0 <= s < 2 ** 31 for s in seeds)
    assert row_seed(42, "test", 1, 7) == row_seed(42, "test", 1, 7)


def test_run_vllm_round_trip_without_torchruns_variables(fake_vllm, tmp_path, monkeypatch):
    from gen_backend import run_vllm
    for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID"):
        monkeypatch.setenv(var, "1")
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    work = tmp_path / "work"
    requests = [request([5, 6], seed=1), request([7, 8, 9], seed=2)]
    out = run_vllm(str(work), "some/model", requests, {**GREEDY, "logprobs": True}, lora_dir="some/adapter",
                   max_lora_rank=16, vllm_py=fake_vllm.python)
    assert out == [{"token_ids": [111, 151645], "logprobs": [-0.5, -0.1]},
                   {"token_ids": [124, 151645], "logprobs": [-0.5, -0.1]}]
    call = fake_vllm.calls()[0]
    assert call["requests"] == requests and call["params"] == {**GREEDY, "logprobs": True}
    assert (call["args"]["model"], call["args"]["lora"], call["args"]["max_lora_rank"]) == ("some/model", "some/adapter", 16)
    assert call["args"]["max_model_len"] == 7                       # the longest prompt plus its max_tokens
    env = call["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "3" and env["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert not {"RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID",
                "PYTHONPATH"} & set(env)
    assert os.listdir(work) == []                                     # the files go once it worked


def test_a_short_answer_file_fails_loudly(fake_vllm, tmp_path, monkeypatch):
    from gen_backend import run_vllm
    monkeypatch.setenv("FAKE_VLLM_MODE", "short")
    with pytest.raises(RuntimeError, match="1 answers for 2 requests"):
        run_vllm(str(tmp_path / "w"), "m", [request([1]), request([2])], GREEDY, vllm_py=fake_vllm.python)


def test_a_vllm_crash_fails_with_its_last_log_lines_and_keeps_them(fake_vllm, tmp_path, monkeypatch):
    from gen_backend import run_vllm
    monkeypatch.setenv("FAKE_VLLM_MODE", "fail")
    work = tmp_path / "w"
    with pytest.raises(RuntimeError, match="(?s)exit 3.*boom"):
        run_vllm(str(work), "m", [request([1])], GREEDY, vllm_py=fake_vllm.python)
    assert "boom" in (work / "vllm.log").read_text()


def test_no_requests_start_no_vllm(fake_vllm, tmp_path):
    from gen_backend import run_vllm
    assert run_vllm(str(tmp_path / "w"), "m", [], GREEDY, vllm_py=fake_vllm.python) == []
    assert fake_vllm.calls() == []


def test_find_vllm_python_takes_vllm_py_and_its_version(fake_vllm):
    from gen_backend import find_vllm_python
    assert find_vllm_python() == (fake_vllm.python, "0.27.1")


def test_find_vllm_python_does_not_fall_back_when_vllm_py_is_set(fake_vllm, monkeypatch, tmp_path):
    import gen_backend
    monkeypatch.setattr(gen_backend, "VLLM_CANDIDATES", (fake_vllm.python,))
    monkeypatch.setenv("VLLM_PY", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="no vLLM environment"):
        gen_backend.find_vllm_python()
    monkeypatch.delenv("VLLM_PY")
    assert gen_backend.find_vllm_python() == (fake_vllm.python, "0.27.1")


def test_an_exported_adapter_reproduces_the_peft_model(tmp_path):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM
    from gen_backend import export_for_vllm
    base = tiny_qwen(0)
    base.save_pretrained(tmp_path / "base")
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=16, lora_dropout=0.0,
                                            target_modules=LORA_TARGETS))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.02)
    model_dir, lora_dir, rank = export_for_vllm(EngineShim(model), str(tmp_path / "base"), str(tmp_path / "x"))
    assert (model_dir, rank) == (str(tmp_path / "base"), 4)
    reloaded = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_dir), lora_dir)
    ids = torch.tensor([[5, 6, 7, 8]])
    torch.testing.assert_close(reloaded(input_ids=ids).logits, model(input_ids=ids).logits)


def test_a_model_without_lora_is_exported_whole(tmp_path):
    from transformers import AutoModelForCausalLM
    from gen_backend import export_for_vllm
    model = tiny_qwen(0)
    model_dir, lora_dir, rank = export_for_vllm(model, "unused", str(tmp_path / "x"))
    assert lora_dir is None and rank == 0
    ids = torch.tensor([[5, 6, 7, 8]])
    torch.testing.assert_close(AutoModelForCausalLM.from_pretrained(model_dir)(input_ids=ids).logits,
                               model(input_ids=ids).logits)


def test_a_weightless_model_resolves_like_the_saved_one(tmp_path):
    from gen_backend import meta_model, resolve_generation_config
    tiny_qwen(0).save_pretrained(tmp_path / "m")
    g = resolve_generation_config(meta_model(str(tmp_path / "m")), None, do_sample=False)
    assert g.eos_token_id == QWEN_EOS and g.do_sample is False
```

- [ ] **Step 5: Run the tests to see them fail**

Run: `dev_pytest tests/test_gen_backend.py`
Expected: every test fails with `ModuleNotFoundError: No module named 'gen_backend'`.

- [ ] **Step 6: Write `gen_backend.py`**

```python
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
```

- [ ] **Step 7: Run the tests to see them pass**

Run: `dev_pytest tests/test_gen_backend.py`
Expected: `19 passed`.

- [ ] **Step 8: Commit**

```bash
cd ${REPO} && git add gen_backend.py tests/fake_vllm.py tests/test_gen_backend.py tests/conftest.py && \
git commit -q -m "feat: gen_backend, settings and requests for vLLM generation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `tools/vllm_generate.py` and the GPU smoke test

**Files:**
- Create: `tools/vllm_generate.py`, `tests/test_vllm_generate.py`

**Interfaces:**
- Consumes: Task 1's `run_vllm`, `vllm_params`, `resolve_generation_config` and
  `unpadded_prompts` (smoke test only).
- Produces: the command line `run_vllm` builds:
  - `--model --lora --max-lora-rank --requests --params --out --max-model-len --gpu-memory-gb
    --enforce-eager`;
  - helpers `allowed_lora_rank(rank) -> int`, `sampling_kwargs(params, request) -> dict` and
    `output_record(completion, logprobs) -> dict`.

- [ ] **Step 1: Write the failing tests, `tests/test_vllm_generate.py`**

```python
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


def test_an_output_line_holds_the_tokens_and_their_log_probabilities():
    completion = SimpleNamespace(token_ids=[7, 151645], logprobs=[{7: SimpleNamespace(logprob=-0.25)},
                                                                  {151645: SimpleNamespace(logprob=-0.01)}])
    assert vg.output_record(completion, False) == {"token_ids": [7, 151645]}
    assert vg.output_record(completion, True) == {"token_ids": [7, 151645], "logprobs": [-0.25, -0.01]}
```

- [ ] **Step 2: Run them to see them fail**

Run: `dev_pytest tests/test_vllm_generate.py`
Expected: an error collecting the file: `FileNotFoundError` for `tools/vllm_generate.py`.

- [ ] **Step 3: Write `tools/vllm_generate.py`**

```python
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

    total_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    llm = LLM(model=a.model, skip_tokenizer_init=True, dtype="bfloat16", seed=0, generation_config="vllm",
              max_model_len=a.max_model_len, gpu_memory_utilization=min(a.gpu_memory_gb / total_gb, 0.9),
              enforce_eager=a.enforce_eager, enable_lora=a.lora is not None,
              max_lora_rank=allowed_lora_rank(a.max_lora_rank), max_loras=1)
    results = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_token_ids"]) for r in requests],
                           [SamplingParams(**sampling_kwargs(params, r)) for r in requests],
                           lora_request=LoRARequest("eval", 1, a.lora) if a.lora else None, use_tqdm=False)
    with open(a.out, "w") as f:
        for result in results:
            f.write(json.dumps(output_record(result.outputs[0], params["logprobs"])) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `dev_pytest tests/test_vllm_generate.py`
Expected: `11 passed`.

- [ ] **Step 5: GPU smoke test: real vLLM against Hugging Face greedy, with and without a LoRA**

```bash
sync_dev && ${BOX} 'cat > /workspace/vllm_checks/smoke.py' <<'EOF'
"""64 FewRel task-0 test prompts, greedy, 64 new tokens: Hugging Face generate() against vLLM."""
import json
import sys

import torch
sys.path.insert(0, "/workspace/opened_dev")
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from gen_backend import resolve_generation_config, run_vllm, unpadded_prompts, vllm_params

BASE = "models/Qwen3-0.6B"
ADAPTER = ("/workspace/dev_runs/results_2026-10-05/qwen3/ced/bench_t0_fewrel/task0/"
           "e1-bs16-lr0.0002-G2-N1-NN1-lora-16-64-0.1/40")
STOP = [151645, 151643]
tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
rows = [json.loads(line) for line in open("data/fewrel_perm0/0/test.jsonl")][:64]
texts = [tok.apply_chat_template([{"role": "system", "content": r["system_prompt"]},
                                  {"role": "user", "content": r["user_prompt"]}],
                                 add_generation_prompt=True, tokenize=False, enable_thinking=False) for r in rows]


def cut(ids):
    for i, t in enumerate(ids):
        if t in STOP:
            return ids[:i]
    return ids


for mode in ("full", "lora"):
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda().eval()
    if mode == "lora":
        model = PeftModel.from_pretrained(model, ADAPTER).eval()
    hf = []
    for b in range(0, len(texts), 16):
        enc = tok(texts[b:b + 16], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            seq = model.generate(**enc, do_sample=False, max_new_tokens=64, eos_token_id=STOP, pad_token_id=151643)
        hf += [cut(row) for row in seq[:, enc["input_ids"].size(1):].tolist()]
    enc = tok(texts, return_tensors="pt", padding=True)
    params = vllm_params(resolve_generation_config(model, None, do_sample=False, eos_token_id=STOP,
                                                   pad_token_id=151643))
    requests = [{"prompt_token_ids": p, "max_tokens": 64, "seed": 0}
                for p in unpadded_prompts(enc["input_ids"], enc["attention_mask"])]
    out = run_vllm(f"/workspace/vllm_checks/smoke_{mode}", BASE, requests, params,
                   lora_dir=ADAPTER if mode == "lora" else None, max_lora_rank=16)
    vl = [cut(o["token_ids"]) for o in out]
    print(f"SMOKE {mode}: {sum(a == b for a, b in zip(hf, vl)) / len(hf):.3f} identical answers ({len(hf)} prompts)",
          flush=True)
    del model
    torch.cuda.empty_cache()
EOF
${BOX} "${DEV_GPU_ENV} && .venv/bin/python /workspace/vllm_checks/smoke.py 2>&1 | grep -E 'SMOKE|Error|error' | tail -20"
```
Expected:
- `SMOKE full: X identical answers (64 prompts)` with X ≥ 0.95;
- `SMOKE lora: X identical answers (64 prompts)` with X ≥ 0.95.

If vLLM fails to start, read the error tail in the exception. A short reproducer with a known vLLM
setting change (for example `enforce_eager`) tells a configuration problem from a code problem;
use superpowers:systematic-debugging.

- [ ] **Step 6: Commit**

```bash
cd ${REPO} && git add tools/vllm_generate.py tests/test_vllm_generate.py && \
git commit -q -m "feat: tools/vllm_generate.py, token IDs in and out of vLLM

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `evaluate()` through vLLM, `--gen-backend`, and the trainers' startup check

**Files:**
- Modify:
  - `arguments.py`: `add_gen_args` gets `--gen-backend`.
  - `ced_eval.py`: `eval_generation_config`, `check_gen_backend` and the vLLM branch of
    `evaluate`.
  - `ced_finetune.py` and `finetune.py`: `main` calls `check_gen_backend`.
  - `tests/conftest.py`: `ced_args` gets `gen_backend`, `vllm_py` and `model_path`.
- Test: `tests/test_ced_eval_vllm.py`

**Interfaces:**
- Consumes: Task 1's `export_for_vllm`, `find_vllm_python`, `resolve_generation_config`,
  `row_seed`, `run_vllm`, `unpadded_prompts`, `vllm_params`, and the `fake_vllm` fixture.
- Produces:
  - `ced_eval.eval_generation_config(args, tokenizer) -> GenerationConfig`;
  - `ced_eval.check_gen_backend(args, model, tokenizer) -> None`. It sets `args.vllm_py` and
    `args.vllm_version`;
  - the flag `--gen-backend {hf,vllm}` (`args.gen_backend`).

- [ ] **Step 1: Give the test namespace the new fields**

In `tests/conftest.py`, `ced_args`, extend the last line of the `base` dict:

```python
        repetition_penalty=None, num_workers=0, save=None, gen_backend="hf", vllm_py=None, model_path=None,
```

- [ ] **Step 2: Write the failing tests, `tests/test_ced_eval_vllm.py`**

```python
"""evaluate() with --gen-backend vllm, run against tests/fake_vllm.py."""
import json
import os
import random

import pytest

from conftest import EngineShim, ced_args, need, tiny_qwen
from data_utils.lm_datasets import LMTrainDataset

ACE_TEST = "processed_data/ace_b10_perm0/0/qwen/"


@pytest.fixture(scope="module")
def ace_test(tokenizer):
    return LMTrainDataset(ced_args(max_length=470), tokenizer, need(ACE_TEST), "test", 40, 1, random.Random(0))


def vllm_args(tmp_path, fake_vllm, **overrides):
    os.makedirs(tmp_path / "run", exist_ok=True)
    return ced_args(max_length=470, save=str(tmp_path / "run"), eval_batch_size=16, gen_backend="vllm",
                    vllm_py=fake_vllm.python, model_path=str(tmp_path / "base"), **overrides)


def test_vllm_gets_each_rows_unpadded_prompt_in_order_with_todays_cap(tmp_path, tokenizer, ace_test, fake_vllm):
    from ced_eval import evaluate
    evaluate(vllm_args(tmp_path, fake_vllm), tokenizer, tiny_qwen(0).eval(), ace_test, "test", 3, "cpu")
    call = fake_vllm.calls()[0]
    expected = []
    for i in range(len(ace_test)):
        gen_data = ace_test.collate([ace_test[i]])[2]
        expected.append(gen_data["input_ids"][0][gen_data["attention_mask"][0].bool()].tolist())
    assert [r["prompt_token_ids"] for r in call["requests"]] == expected
    assert {r["max_tokens"] for r in call["requests"]} == {470 - 460}        # max_length - prompt width
    assert len({r["seed"] for r in call["requests"]}) == len(ace_test)


def test_vllms_answers_are_decoded_and_scored_like_generates(tmp_path, tokenizer, ace_test, fake_vllm):
    from ced_eval import evaluate
    evaluate(vllm_args(tmp_path, fake_vllm), tokenizer, tiny_qwen(0).eval(), ace_test, "test", 3, "cpu")
    requests = fake_vllm.calls()[0]["requests"]
    # generate()'s path pads every answer with eos and decodes without special tokens
    expected = [tokenizer.decode([100 + sum(r["prompt_token_ids"]) % 1000] + [tokenizer.pad_token_id] * 3,
                                 skip_special_tokens=True) for r in requests]
    run = tmp_path / "run"
    assert [json.loads(line)["text"] for line in open(run / "eval" / "3" / "answers.jsonl")] == expected
    assert "'trigger'" in open(run / "log.txt").read()
    assert not (run / "vllm_tmp").exists()


def test_vllm_gets_the_settings_generate_would_use(tmp_path, tokenizer, ace_test, fake_vllm):
    from ced_eval import evaluate
    evaluate(vllm_args(tmp_path, fake_vllm), tokenizer, tiny_qwen(0).eval(), ace_test, "test", 0, "cpu")
    params = fake_vllm.calls()[0]["params"]
    assert (params["temperature"], params["top_p"], params["top_k"]) == (0.5, 0.95, 0)  # the Qwen3 fill-in
    assert params["stop_token_ids"] == [151645, 151643]


def test_a_lora_model_goes_to_vllm_as_its_adapter_on_the_starting_model(tmp_path, tokenizer, ace_test,
                                                                        fake_vllm, student):
    from ced_eval import evaluate
    evaluate(vllm_args(tmp_path, fake_vllm), tokenizer, EngineShim(student).eval(), ace_test, "test", 0, "cpu")
    call = fake_vllm.calls()[0]
    assert call["args"]["model"] == str(tmp_path / "base")
    assert call["args"]["lora"] == str(tmp_path / "run" / "vllm_tmp" / "adapter")
    assert call["args"]["max_lora_rank"] == 4
    assert {"adapter_config.json", "adapter_model.safetensors"} <= set(call["lora_files"])


def test_the_startup_check_refuses_more_than_one_process(monkeypatch, tokenizer):
    import torch.distributed as dist
    from ced_eval import check_gen_backend
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="world size 1"):
        check_gen_backend(ced_args(gen_backend="vllm"), tiny_qwen(0), tokenizer)


def test_the_startup_check_refuses_without_vllm(monkeypatch, tmp_path, tokenizer):
    from ced_eval import check_gen_backend
    monkeypatch.setenv("VLLM_PY", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="no vLLM environment"):
        check_gen_backend(ced_args(gen_backend="vllm"), tiny_qwen(0), tokenizer)


def test_the_startup_check_refuses_settings_vllm_cannot_reproduce(fake_vllm, tokenizer):
    from ced_eval import check_gen_backend
    from gen_backend import UnsupportedForVLLM
    model = tiny_qwen(0)
    model.generation_config.no_repeat_ngram_size = 6      # filled in, so generate() would block n-grams
    with pytest.raises(UnsupportedForVLLM, match="no_repeat_ngram_size"):
        check_gen_backend(ced_args(gen_backend="vllm"), model, tokenizer)


def test_the_startup_check_records_the_interpreter(fake_vllm, tokenizer):
    from ced_eval import check_gen_backend
    args = ced_args(gen_backend="vllm")
    check_gen_backend(args, tiny_qwen(0), tokenizer)
    assert (args.vllm_py, args.vllm_version) == (fake_vllm.python, "0.27.1")


def test_the_hf_backend_needs_no_check(monkeypatch, tmp_path, tokenizer):
    from ced_eval import check_gen_backend
    monkeypatch.setenv("VLLM_PY", str(tmp_path / "missing"))
    check_gen_backend(ced_args(), tiny_qwen(0), tokenizer)


def test_both_trainers_check_the_backend_before_training():
    import inspect

    import ced_finetune
    import finetune
    for trainer in (ced_finetune, finetune):
        assert "check_gen_backend(args, model, tokenizer)" in inspect.getsource(trainer.main)
```

- [ ] **Step 3: Run them to see them fail**

Run: `dev_pytest tests/test_ced_eval_vllm.py`
Expected: failures with `ImportError: cannot import name 'check_gen_backend' from 'ced_eval'`.
The tests that call `evaluate()` with `gen_backend="vllm"` fail with `FileNotFoundError` reading
the fake's log, because today's `evaluate()` generates with Hugging Face.

- [ ] **Step 4: Add `--gen-backend` to `arguments.py`**

In `add_gen_args`, after the `--strict-generation` argument:

```python
    group.add_argument("--gen-backend", choices=["hf", "vllm"], default="hf",
                       help="hf: answers come from model.generate(); vllm: from tools/vllm_generate.py in the "
                            "vLLM environment, with the settings generate() would use (gen_backend.py)")
```

- [ ] **Step 5: Change `ced_eval.py`**

Add `import shutil` to the imports, and after `from ed_eval import ed_evaluate`:

```python
from gen_backend import (export_for_vllm, find_vllm_python, resolve_generation_config, row_seed, run_vllm,
                         unpadded_prompts, vllm_params)
```

After `_loader`, add:

```python
def eval_generation_config(args, tokenizer):
    """The GenerationConfig evaluate() hands to generate(); transformers then fills in the model's
    defaults (gen_config.py)."""
    return GenerationConfig(
        do_sample=args.do_sample, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature,
        repetition_penalty=args.repetition_penalty, max_length=args.max_length, min_length=None,
        eos_token_id=[tokenizer.eos_token_id, 151643], pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True, output_scores=False)


def check_gen_backend(args, model, tokenizer):
    """With --gen-backend vllm, refuse to start a run whose evaluation vLLM could not do."""
    if getattr(args, "gen_backend", "hf") != "vllm":
        return
    if dist.get_world_size() != 1:
        raise ValueError("--gen-backend vllm evaluates in one process: run with world size 1")
    args.vllm_py, args.vllm_version = find_vllm_python()
    vllm_params(resolve_generation_config(model, eval_generation_config(args, tokenizer), **generation_kwargs(args)))
    print_rank(f"generation backend: vLLM {args.vllm_version} ({args.vllm_py})")


def _vllm_response_ids(args, model, loader, generation_config, split, epoch):
    """Each row's answer token IDs from vLLM, with the settings generate() would decode with."""
    requests = []
    for _, _, gen_data, _, _ in loader:
        width = gen_data["input_ids"].size(1)
        for prompt in unpadded_prompts(gen_data["input_ids"], gen_data["attention_mask"]):
            requests.append({"prompt_token_ids": prompt, "max_tokens": args.max_length - width,
                             "seed": row_seed(args.seed, split, epoch, len(requests))})
    params = vllm_params(resolve_generation_config(model, generation_config, **generation_kwargs(args)))
    work_dir = os.path.join(args.save, "vllm_tmp")
    model_dir, lora_dir, rank = export_for_vllm(model, args.model_path, work_dir)
    outputs = run_vllm(work_dir, model_dir, requests, params, lora_dir=lora_dir, max_lora_rank=rank,
                       max_model_len=args.max_length, vllm_py=getattr(args, "vllm_py", None))
    shutil.rmtree(work_dir, ignore_errors=True)
    return [output["token_ids"] for output in outputs]
```

In `evaluate`, replace the generation block. It runs from `responses = None` to
`responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)`, inclusive:

```python
    responses = None
    if generate and args.eval_gen:
        generation_config = eval_generation_config(args, tokenizer)
        loader = _loader(args, dataset, args.eval_batch_size)
        if getattr(args, "gen_backend", "hf") == "vllm":
            response_ids = _vllm_response_ids(args, model, loader, generation_config, split, epoch)
            responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        else:
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
```

- [ ] **Step 6: Call the check in both trainers**

In `ced_finetune.py`, change the import to
`from ced_eval import check_gen_backend, evaluate, eval_plan, final_test_missing`. In `main`, add
right after
`model, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)`:

```python
    check_gen_backend(args, model, tokenizer)     # --gen-backend vllm: refuse before the first update
```

In `finetune.py`, make the same two changes: the import on line 49 and the call after
`setup_model_and_optimizer` in `main`.

- [ ] **Step 7: Run the new tests, then the evaluation tests, to see them pass**

Run: `dev_pytest tests/test_ced_eval_vllm.py tests/test_ced_eval.py tests/test_gen_config.py`
Expected:
- `10 passed` from `test_ced_eval_vllm.py`;
- every test in `test_ced_eval.py` and `test_gen_config.py` passes as before (the Hugging Face
  path is unchanged).

- [ ] **Step 8: Commit**

```bash
cd ${REPO} && git add arguments.py ced_eval.py ced_finetune.py finetune.py tests/conftest.py \
    tests/test_ced_eval_vllm.py && \
git commit -q -m "feat: --gen-backend vllm for the trainers' evaluation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: CL-LoRA evaluation through vLLM

**Files:**
- Create: `cl_lora/vllm_export.py`, `tests/test_cl_vllm.py`
- Modify `cl_lora/engine.py`:
  - imports;
  - `parse_args`: `--gen-backend`;
  - `runtime_fingerprint`;
  - a new `manifest_for(a)` and `RESUME_KEYS`;
  - `cl_generation_config`, `_vllm_generate`, and the vLLM branch of `eval_task`;
  - in `main`, the startup check and the use of `manifest_for` and `RESUME_KEYS`.

**Interfaces:**
- Consumes: Task 1's `find_vllm_python`, `meta_model`, `resolve_generation_config`, `run_vllm`,
  `unpadded_prompts`, `vllm_params`, and the `fake_vllm` fixture.
- Produces, in `cl_lora/vllm_export.py`:
  - `SUMMED = ("inclora", "olora", "inflora", "tree")` and `WHOLE = ("migu",)`;
  - `effective_backend(method, requested) -> "hf" | "vllm"`;
  - `export_summed_adapters(model, adapters, base_path, out_dir) -> int`, the summed rank.
- Produces, in `cl_lora/engine.py`:
  - `manifest_for(a) -> dict`, which now includes `gen_backend`;
  - `RESUME_KEYS: tuple`, which includes `"gen_backend"`;
  - `cl_generation_config(model, tok)`;
  - the flag `--gen-backend` (`a.gen_backend`);
  - `a.vllm_py`, set in `main`.

- [ ] **Step 1: Write the failing tests, `tests/test_cl_vllm.py`**

```python
"""CL-LoRA evaluation through vLLM: the exported adapter, eval_task's vLLM path, and the manifest."""
import json
import os
import sys
from argparse import Namespace

import pytest
import torch

from conftest import LORA_TARGETS, MODEL_DIR, need, tiny_qwen

TACRED = "data/tacred_perm0"


def summed_model(n_adapters, tmp_path):
    """A tiny PEFT model with n task adapters, all active (as consolidate() leaves them), its base saved."""
    from peft import LoraConfig, get_peft_model
    from cl_lora.multi_adapter import CLLoRAManager

    def config():
        return LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=16, lora_dropout=0.0, target_modules=LORA_TARGETS)

    base = tiny_qwen(0)
    base.save_pretrained(tmp_path / "base")
    model = get_peft_model(base, config(), adapter_name="task0")
    mgr = CLLoRAManager(model, orth_lambda=0.0)
    mgr.register_task0("task0")
    for task_id in range(1, n_adapters):
        mgr.start_new_task(task_id, config())
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.normal_(0, 0.05)
    mgr.consolidate()
    return model.eval(), mgr


def engine_args(tmp_path, method="inclora", vllm_py=None):
    return Namespace(cl_method=method, data_root=need(TACRED), max_length=768, max_prompt_length=460, limit=6,
                     eval_batch_size=4, gen_backend="vllm", vllm_py=vllm_py, model_path=str(tmp_path / "base"),
                     save=str(tmp_path / "run"))


@pytest.mark.parametrize("n_adapters", [1, 2, 3])
def test_the_concatenated_adapter_computes_the_consolidated_sum(n_adapters, tmp_path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from cl_lora.vllm_export import export_summed_adapters
    model, mgr = summed_model(n_adapters, tmp_path)
    rank = export_summed_adapters(model, mgr.task_adapters, str(tmp_path / "base"), str(tmp_path / "adapter"))
    assert rank == 4 * n_adapters
    reloaded = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(tmp_path / "base"), tmp_path / "adapter")
    ids = torch.tensor([[5, 6, 7, 8, 9]])
    torch.testing.assert_close(reloaded(input_ids=ids).logits, model(input_ids=ids).logits, atol=1e-5, rtol=0)


@pytest.mark.parametrize("method,requested,backend", [
    ("inclora", "vllm", "vllm"), ("olora", "vllm", "vllm"), ("inflora", "vllm", "vllm"), ("tree", "vllm", "vllm"),
    ("migu", "vllm", "vllm"), ("gainlora_o", "vllm", "hf"), ("gainlora_inf", "vllm", "hf"), ("epi", "vllm", "hf"),
    ("inclora", "hf", "hf")])
def test_gated_and_routed_methods_keep_hugging_face_generation(method, requested, backend):
    from cl_lora.vllm_export import effective_backend
    assert effective_backend(method, requested) == backend


def test_eval_task_sends_the_unpadded_prompts_and_decodes_vllms_answers(tmp_path, fake_vllm):
    from transformers import AutoTokenizer
    from cl_lora.engine import JsonlED, eval_task
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model, mgr = summed_model(2, tmp_path)
    a = engine_args(tmp_path, vllm_py=fake_vllm.python)
    os.makedirs(a.save)
    metrics, preds, refs = eval_task(a, model, tok, "cpu", 1, mgr)
    call = fake_vllm.calls()[0]
    ds = JsonlED(os.path.join(a.data_root, "1", "test.jsonl"), tok, 768, 460, "test", limit=6)
    assert [r["prompt_token_ids"] for r in call["requests"]] == [ds[i]["prompt"] for i in range(6)]
    assert {r["max_tokens"] for r in call["requests"]} == {768 - 460}
    assert call["params"]["temperature"] == 0.0 and call["params"]["stop_token_ids"] == [tok.eos_token_id, 151643]
    assert call["args"]["model"] == a.model_path and call["args"]["max_lora_rank"] == 8
    assert call["args"]["lora"] == os.path.join(a.save, "vllm_tmp", "adapter")
    assert preds == [tok.decode([100 + sum(r["prompt_token_ids"]) % 1000], skip_special_tokens=True)
                     for r in call["requests"]]
    assert refs == [[ds[i]["answer"]] for i in range(6)]
    assert "trigger" in metrics and not os.path.exists(os.path.join(a.save, "vllm_tmp"))


def test_migu_is_handed_to_vllm_whole(tmp_path, fake_vllm):
    from transformers import AutoTokenizer
    from cl_lora.engine import eval_task
    a = engine_args(tmp_path, method="migu", vllm_py=fake_vllm.python)
    os.makedirs(a.save)
    eval_task(a, tiny_qwen(0).eval(), AutoTokenizer.from_pretrained(MODEL_DIR), "cpu", 1, None)
    call = fake_vllm.calls()[0]
    assert call["args"]["model"] == os.path.join(a.save, "vllm_tmp", "model") and call["args"]["lora"] is None


def test_a_resume_under_the_other_backend_is_refused(tmp_path, fake_vllm, monkeypatch):
    from cl_lora import engine
    save = tmp_path / "run"
    argv = ["engine.py", "--cl-method", "inclora", "--data-root", need(TACRED), "--num-tasks", "10",
            "--model-path", MODEL_DIR, "--save", str(save)]
    monkeypatch.setattr(sys, "argv", argv + ["--gen-backend", "hf"])
    a = engine.parse_args()
    a.loss_group_size = a.batch_size                         # as main() sets it
    save.mkdir()
    (save / "run_manifest.json").write_text(json.dumps(engine.manifest_for(a)))
    monkeypatch.setattr(sys, "argv", argv + ["--gen-backend", "vllm", "--resume"])
    with pytest.raises(ValueError, match="resume manifest mismatch for gen_backend"):
        engine.main()


def test_a_vllm_run_without_vllm_is_refused_before_its_run_directory_exists(tmp_path, monkeypatch):
    from cl_lora import engine
    monkeypatch.setenv("VLLM_PY", str(tmp_path / "missing"))
    monkeypatch.setattr(sys, "argv", ["engine.py", "--cl-method", "inclora", "--data-root", need(TACRED),
                                      "--model-path", MODEL_DIR, "--save", str(tmp_path / "run"),
                                      "--gen-backend", "vllm"])
    with pytest.raises(RuntimeError, match="no vLLM environment"):
        engine.main()
    assert not (tmp_path / "run").exists()
```

- [ ] **Step 2: Run them to see them fail**

Run: `dev_pytest tests/test_cl_vllm.py`
Expected: failures with `ModuleNotFoundError: No module named 'cl_lora.vllm_export'`. The
engine tests fail on `--gen-backend` being an unrecognized argument, or on the missing
`manifest_for`.

- [ ] **Step 3: Write `cl_lora/vllm_export.py`**

```python
"""What vLLM runs when a CL-LoRA model is evaluated (spec §4).

- inclora, olora, inflora and tree evaluate the base plus the sum of every task adapter
  (CLLoRAManager.consolidate). One adapter with A = [A_1; ...; A_n] and B = [s_1 B_1, ..., s_n B_n]
  (rank and lora_alpha both sum(r_i), so its own scaling is 1) computes exactly that sum.
- migu fine-tunes the base weights, so the whole model is saved (engine._vllm_generate).
- gainlora_o, gainlora_inf and epi pick adapters per input (gates, router). They keep Hugging
  Face generation whatever --gen-backend says."""
import os

import torch
from peft import LoraConfig
from peft.tuners.lora import LoraLayer
from safetensors.torch import save_file

SUMMED = ("inclora", "olora", "inflora", "tree")
WHOLE = ("migu",)


def effective_backend(method, requested):
    """The generation backend eval_task uses for `method` under --gen-backend `requested`."""
    return "vllm" if requested == "vllm" and method in SUMMED + WHOLE else "hf"


def export_summed_adapters(model, adapters, base_path, out_dir):
    """Save the sum of the PEFT adapters named in `adapters` as one LoRA adapter in out_dir.
    Returns its rank."""
    state, targets, ranks = {}, set(), set()
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        names = [adapter for adapter in adapters if adapter in module.lora_A]
        if not names:
            continue
        A = torch.cat([module.lora_A[n].weight.detach().float() for n in names], dim=0)
        B = torch.cat([module.lora_B[n].weight.detach().float() * module.scaling[n] for n in names], dim=1)
        state[f"{name}.lora_A.weight"] = A.cpu().contiguous()
        state[f"{name}.lora_B.weight"] = B.cpu().contiguous()
        targets.add(name.rsplit(".", 1)[-1])
        ranks.add(A.size(0))
    if len(ranks) != 1:
        raise ValueError(f"the task adapters cover the layers unevenly (summed ranks {sorted(ranks)})")
    rank = ranks.pop()
    os.makedirs(out_dir, exist_ok=True)
    save_file(state, os.path.join(out_dir, "adapter_model.safetensors"))
    LoraConfig(task_type="CAUSAL_LM", r=rank, lora_alpha=rank, lora_dropout=0.0, target_modules=sorted(targets),
               base_model_name_or_path=base_path).save_pretrained(out_dir)
    return rank
```

- [ ] **Step 4: Change `cl_lora/engine.py`**

Imports: add `import shutil` after `import random`. After `from cl_lora import gainlora as gain_mod`,
add:

```python
from cl_lora.vllm_export import WHOLE, effective_backend, export_summed_adapters
from gen_backend import find_vllm_python, meta_model, resolve_generation_config, run_vllm, unpadded_prompts, vllm_params
```

`parse_args`: after the `--resume` argument, add:

```python
    p.add_argument("--gen-backend", choices=["hf", "vllm"], default="hf",
                   help="vllm: evaluation answers from vLLM (cl_lora/vllm_export.py); "
                        "GainLoRA and EPI keep Hugging Face generation")
```

`runtime_fingerprint`: add the two files that now shape the evaluation to its list:

```python
        os.path.join(module_root, "vllm_export.py"),
        os.path.join(project_root, "gen_backend.py"),
```

After `def _generate(...)`, add:

```python
def cl_generation_config(model, tok):
    """The settings _generate()'s generate() call decodes with."""
    return resolve_generation_config(model, None, do_sample=False, eos_token_id=[tok.eos_token_id, 151643],
                                     pad_token_id=tok.pad_token_id)


def _vllm_generate(a, model, tok, mgr, prompts):
    """Greedy answers from vLLM for the evaluated model (cl_lora/vllm_export.py)."""
    params = vllm_params(cl_generation_config(model, tok))
    requests = [{"prompt_token_ids": prompt, "max_tokens": a.max_length - a.max_prompt_length, "seed": 0}
                for prompt in prompts]
    work_dir = os.path.join(a.save, "vllm_tmp")
    if a.cl_method in WHOLE:
        model_dir, lora_dir, rank = os.path.join(work_dir, "model"), None, 0
        model.save_pretrained(model_dir, safe_serialization=True)
    else:
        model_dir, lora_dir = a.model_path, os.path.join(work_dir, "adapter")
        rank = export_summed_adapters(model, mgr.task_adapters, a.model_path, lora_dir)
    outputs = run_vllm(work_dir, model_dir, requests, params, lora_dir=lora_dir, max_lora_rank=rank,
                       max_model_len=a.max_length, vllm_py=a.vllm_py)
    shutil.rmtree(work_dir, ignore_errors=True)
    return tok.batch_decode([output["token_ids"] for output in outputs], skip_special_tokens=True)
```

`eval_task`: right after `model.eval()`, add:

```python
    if effective_backend(a.cl_method, a.gen_backend) == "vllm":
        prompts = []
        for mb, answers in loader:
            prompts.extend(unpadded_prompts(mb["input_ids"], mb["attention_mask"]))
            refs.extend([[x] for x in answers])
        preds = _vllm_generate(a, model, tok, mgr, prompts)
        return ed_evaluate(preds, refs), preds, refs
```

The manifest moves into a function. Add above `def main():`:

```python
RESUME_KEYS = ("method", "data_root", "seed", "model", "rank", "alpha", "dropout", "data_sha256", "runtime_sha256",
               "micro_batch", "gradient_accumulation", "loss_group_size", "epochs", "num_tasks", "row_limit",
               "scheduler", "prompt_mode", "gen_backend")


def manifest_for(a):
    """The run manifest a new run writes and a resumed run must match on RESUME_KEYS."""
    return {
        "method": a.cl_method,
        "data_root": os.path.abspath(a.data_root),
        "seed": a.seed,
        "model": a.model_path,
        "data_sha256": dataset_fingerprint(a.data_root, a.num_tasks),
        "runtime_sha256": runtime_fingerprint(),
        "rank": None if a.cl_method == "migu" else a.rank,
        "alpha": None if a.cl_method == "migu" else a.alpha,
        "dropout": 0.0 if a.cl_method in GATED else a.dropout,
        "micro_batch": a.batch_size,
        "gradient_accumulation": a.grad_accum,
        "effective_batch": a.batch_size * a.grad_accum,
        "loss_group_size": a.loss_group_size,
        "epochs": a.epochs,
        "num_tasks": a.num_tasks,
        "row_limit": a.limit,
        "scheduler": "warmup_cosine",
        "warmup_ratio": a.warmup_ratio,
        "prompt_mode": "qwen_chat_template_thinking_disabled",
        "decoding": "greedy",
        "gen_backend": effective_backend(a.cl_method, a.gen_backend),
        "status": "running",
        "completed_task": -1,
    }
```

In `main`:
- Replace the `requested_manifest = {...}` literal with `requested_manifest = manifest_for(a)`.
- Replace the inline key tuple of the resume loop with `for key in RESUME_KEYS:`.
- After the MIGU `ValueError` check and before `random.seed(a.seed)`, add:

```python
    a.vllm_py = None
    if effective_backend(a.cl_method, a.gen_backend) == "vllm":
        # before the run directory exists: a refused run leaves nothing to clean up
        a.vllm_py, version = find_vllm_python()
        vllm_params(cl_generation_config(meta_model(a.model_path), AutoTokenizer.from_pretrained(a.model_path)))
        print(f"[cl:{a.cl_method}] generation backend: vLLM {version} ({a.vllm_py})", flush=True)
```

- [ ] **Step 5: Run the new tests and the engine tests to see them pass**

Run: `dev_pytest tests/test_cl_vllm.py tests/test_engine_grouping.py`
Expected:
- `16 passed` from `test_cl_vllm.py`;
- every test in `test_engine_grouping.py` passes as before.

- [ ] **Step 6: Commit**

```bash
cd ${REPO} && git add cl_lora/vllm_export.py cl_lora/engine.py tests/test_cl_vllm.py && \
git commit -q -m "feat: CL-LoRA evaluation through vLLM, one concatenated adapter per evaluation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Pseudo-labelling through vLLM

**Files:**
- Modify `tools/ced_pseudo_label.py`:
  - add `chat_prompts`, `strip_stop`, `generate_hf` and `generate_vllm`;
  - `main` gets `--gen-backend`, and its pass-1 loop reads from either generator.
- Test: `tests/test_pl_vllm.py`

**Interfaces:**
- Consumes: Task 1's `find_vllm_python`, `meta_model`, `resolve_generation_config`, `run_vllm`,
  `vllm_params`, and the `fake_vllm` fixture.
- Produces, in `tools/ced_pseudo_label.py`:
  - `chat_prompts(tokenizer, rows, idxs) -> list[str]`;
  - `strip_stop(ids, tokenizer) -> list[int]`;
  - `generate_hf(model, tokenizer, prompts, max_new_tokens, need_scores, device)` and
    `generate_vllm(teacher, tokenizer, prompts, max_new_tokens, need_scores, work_dir, vllm_py)`.
    Both return a list of `(text, gid | None, lp | None)`.

- [ ] **Step 1: Write the failing tests, `tests/test_pl_vllm.py`**

```python
"""tools/ced_pseudo_label.py with --gen-backend vllm."""
import importlib.util
import json
import os
import sys

import pytest
import torch

from conftest import MODEL_DIR, need, tiny_qwen

_SPEC = importlib.util.spec_from_file_location("ced_pseudo_label", os.path.join("tools", "ced_pseudo_label.py"))
pl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pl)


@pytest.fixture(scope="module")
def teacher_dir(tmp_path_factory):
    """A tiny teacher saved like a merged model, next to the real tokenizer."""
    from transformers import AutoTokenizer
    path = tmp_path_factory.mktemp("teacher")
    tiny_qwen(0).save_pretrained(path)
    AutoTokenizer.from_pretrained(need(MODEL_DIR)).save_pretrained(path)
    return str(path)


def row(sentence, events):
    return {"system_prompt": "You extract events.",
            "user_prompt": f"Given an input text: {sentence}\n\nYour task is to list the events.",
            "response": json.dumps({"events": events})}


def write_tasks(root, task1_rows):
    """Two tasks: task 0 teaches TypeA, task 1 adds TypeB."""
    root.mkdir(parents=True)
    (root / "streams.json").write_text(json.dumps([["TypeA"], ["TypeB"]]))
    for task, rows in ((0, [row("They attack the town.", [["attack", "TypeA", [], ""]])]), (1, task1_rows)):
        (root / str(task)).mkdir()
        for split in ("train", "dev", "test"):
            (root / str(task) / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def run_main(monkeypatch, teacher, data, out):
    monkeypatch.setattr(sys, "argv", ["ced_pseudo_label.py", "--teacher", teacher, "--data-dir", str(data / "1"),
                                      "--streams", str(data / "streams.json"), "--task-id", "1", "--out", str(out),
                                      "--conf-filter", "percentile", "--gen-backend", "vllm"])
    pl.main()


def test_vllms_tokens_rebuild_generates_texts_ids_and_scores(teacher_dir, tmp_path, fake_vllm, monkeypatch):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(teacher_dir, padding_side="left")
    rows = [row(f"Sentence {i}: they meet and attack.", [["meet", "TypeB", [], ""]]) for i in range(4)]
    prompts = pl.chat_prompts(tok, rows, range(4))
    hf = pl.generate_hf(AutoModelForCausalLM.from_pretrained(teacher_dir).eval(), tok, prompts, 8, True, "cpu")
    # vLLM returns what generate() produced: its tokens (without padding) and their log-probabilities
    replay = [{"token_ids": gid, "logprobs": lp[:len(gid)].tolist()} for _, gid, lp in hf]
    (tmp_path / "replay.json").write_text(json.dumps(replay))
    monkeypatch.setenv("FAKE_VLLM_MODE", "replay")
    monkeypatch.setenv("FAKE_VLLM_REPLAY", str(tmp_path / "replay.json"))
    vl = pl.generate_vllm(teacher_dir, tok, prompts, 8, True, str(tmp_path / "work"), fake_vllm.python)
    for (text_h, gid_h, lp_h), (text_v, gid_v, lp_v) in zip(hf, vl):
        assert (text_v, gid_v) == (text_h, gid_h)
        torch.testing.assert_close(lp_v[:len(gid_v)], lp_h[:len(gid_h)])
    call = fake_vllm.calls()[0]
    assert [r["prompt_token_ids"] for r in call["requests"]] == tok(prompts, truncation=True, max_length=1024)["input_ids"]
    assert {r["max_tokens"] for r in call["requests"]} == {8}
    assert call["params"]["temperature"] == 0.0 and call["params"]["logprobs"] is True
    assert call["params"]["stop_token_ids"] == [151645, 151643]          # the teacher's generation_config


def test_main_with_vllm_writes_todays_outputs(teacher_dir, tmp_path, fake_vllm, monkeypatch):
    task1 = [row("They meet.", [["meet", "TypeB", [], ""]]), row("We meet again.", [["meet", "TypeB", [], ""]]),
             row("They attack.", [["attack", "TypeA", [], ""]])]
    data = write_tasks(tmp_path / "data", task1)
    run_main(monkeypatch, teacher_dir, data, tmp_path / "out")
    stats = json.load(open(tmp_path / "out" / "pl_stats.json"))
    assert stats["candidates"] == 2
    calls = fake_vllm.calls()
    assert len(calls) == 1 and len(calls[0]["requests"]) == 2
    # the fake's answers parse to no events: every row is written back unchanged
    assert [json.loads(line) for line in open(tmp_path / "out" / "train.jsonl")] == task1
    assert (tmp_path / "out" / "test.jsonl").exists() and not (tmp_path / "out" / "vllm_tmp").exists()


def test_a_task_without_candidates_starts_no_vllm(teacher_dir, tmp_path, fake_vllm, monkeypatch):
    task1 = [row("They attack.", [["attack", "TypeA", [], ""]]), row("Nothing happens.", [])]
    data = write_tasks(tmp_path / "data", task1)
    run_main(monkeypatch, teacher_dir, data, tmp_path / "out")
    assert fake_vllm.calls() == []
    assert json.load(open(tmp_path / "out" / "pl_stats.json"))["candidates"] == 0
    assert [json.loads(line) for line in open(tmp_path / "out" / "train.jsonl")] == task1
```

- [ ] **Step 2: Run them to see them fail**

Run: `dev_pytest tests/test_pl_vllm.py`
Expected: failures with
`AttributeError: module 'ced_pseudo_label' has no attribute 'chat_prompts'`, and for `main`,
`unrecognized arguments: --gen-backend vllm`.

- [ ] **Step 3: Change `tools/ced_pseudo_label.py`**

After the imports, put the repository root on the path (the runners start this file as
`python tools/ced_pseudo_label.py`, so `sys.path[0]` is `tools/`), then import `gen_backend`:

```python
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gen_backend import find_vllm_python, meta_model, resolve_generation_config, run_vllm, vllm_params  # noqa: E402
```

After `event_conf_score`, add:

```python
def chat_prompts(tokenizer, rows, idxs):
    """The teacher's prompt text for each candidate row."""
    return [tokenizer.apply_chat_template(
        [{"role": "system", "content": rows[i]["system_prompt"]}, {"role": "user", "content": rows[i]["user_prompt"]}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False) for i in idxs]


def strip_stop(ids, tokenizer):
    """Generated ids without eos/pad: the view event_conf_score aligns with the text."""
    return [t for t in ids if t != tokenizer.eos_token_id and t != tokenizer.pad_token_id]


def generate_hf(model, tokenizer, prompts, max_new_tokens, need_scores, device):
    """[(text, gen_ids, token_logprobs)] from model.generate(); the ids and log-probabilities are
    None without the confidence filter."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id,
                             return_dict_in_generate=need_scores, output_scores=need_scores)
    if need_scores:
        sequences = out.sequences
        # [bs, gen_len] logprob of each generated token — no extra forward
        trans = model.compute_transition_scores(sequences, out.scores, normalize_logits=True).float().cpu()
    else:
        sequences = out
    gen_ids_batch = sequences[:, enc["input_ids"].shape[1]:].cpu()
    texts = tokenizer.batch_decode(gen_ids_batch, skip_special_tokens=True)
    if not need_scores:
        return [(text, None, None) for text in texts]
    return [(text, strip_stop(gen_ids_batch[pos].tolist(), tokenizer), trans[pos]) for pos, text in enumerate(texts)]


def generate_vllm(teacher, tokenizer, prompts, max_new_tokens, need_scores, work_dir, vllm_py):
    """generate_hf's results from vLLM, with the settings generate() would use (gen_backend.py)."""
    if not prompts:
        return []
    config = resolve_generation_config(meta_model(teacher), None, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    requests = [{"prompt_token_ids": ids, "max_tokens": max_new_tokens, "seed": 0}
                for ids in tokenizer(prompts, truncation=True, max_length=1024)["input_ids"]]
    outputs = run_vllm(work_dir, teacher, requests, vllm_params(config, logprobs=need_scores), vllm_py=vllm_py)
    shutil.rmtree(work_dir, ignore_errors=True)
    texts = tokenizer.batch_decode([output["token_ids"] for output in outputs], skip_special_tokens=True)
    if not need_scores:
        return [(text, None, None) for text in texts]
    return [(text, strip_stop(output["token_ids"], tokenizer), torch.tensor(output["logprobs"], dtype=torch.float32))
            for output, text in zip(outputs, texts)]
```

`main`, the arguments: after `--conf-thresh`, add:

```python
    ap.add_argument("--gen-backend", choices=["hf", "vllm"], default="hf",
                    help="vllm: the teacher's answers come from vLLM (gen_backend.py), same settings")
```

`main`, the model load. Replace

```python
    device = f"cuda:{args.gpu}"
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16,
                                                 device_map={"": device})
    model.eval()
```

with

```python
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, padding_side="left")
```

`main`, pass 1. Replace everything from `for b in range(0, len(cand_idx), args.batch_size):`
through the line `lp = trans[pos]` with the code below. `pending`, `n_dropped_conflict` and
`n_seen` keep their initialisation above it.

```python
    if args.gen_backend == "vllm":
        vllm_py, version = find_vllm_python()
        print(f"generation backend: vLLM {version} ({vllm_py})", flush=True)
        chunks = [(cand_idx, generate_vllm(args.teacher, tokenizer, chat_prompts(tokenizer, rows, cand_idx),
                                           args.max_new_tokens, need_scores, os.path.join(args.out, "vllm_tmp"),
                                           vllm_py))]
    else:
        device = f"cuda:{args.gpu}"
        model = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16,
                                                     device_map={"": device})
        model.eval()
        chunks = ((cand_idx[b:b + args.batch_size],
                   generate_hf(model, tokenizer, chat_prompts(tokenizer, rows, cand_idx[b:b + args.batch_size]),
                               args.max_new_tokens, need_scores, device))
                  for b in range(0, len(cand_idx), args.batch_size))
    n_done = 0
    for idxs, generated in chunks:
        for i, (text, gid, lp) in zip(idxs, generated):
            r = rows[i]
            sent = input_text_of(r["user_prompt"]) or ""
            gold = json.loads(r["response"]).get("events", [])
            gold_keys = {(e[0], e[1]) for e in gold}
            gold_triggers = {str(e[0]).lower() for e in gold if isinstance(e, list) and e}
```

The per-event body that follows (`for e in parse_events(text):` ...) stays as it is, one level
deeper. Then replace the progress print at the end of the old batch loop with:

```python
        n_done += len(idxs)
        print(f"pseudo-label {n_done}/{len(cand_idx)} "
              f"(candidates so far: {n_seen}, conflict-dropped: {n_dropped_conflict})", flush=True)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `dev_pytest tests/test_pl_vllm.py`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
cd ${REPO} && git add tools/ced_pseudo_label.py tests/test_pl_vllm.py && \
git commit -q -m "feat: --gen-backend vllm for teacher pseudo-labelling

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Runners and scheduler: `GEN_BACKEND`

**Files:**
- Modify:
  - `scripts/qwen/lib.sh`: `gpu_defaults` gets a sixth value, `apply_card_defaults` exports it,
    and a new `vllm_check`.
  - `scripts/qwen/ced/run_ced_v2.sh`: the knob, the flag, the check, the trainer options,
    pseudo-labelling, the manifest config and the echo line.
  - `scripts/qwen/ced/run_cllora.sh`: the check and the engine flag.
  - `project_commands.sh`: the step-3 line and the header's knob list.
  - `README.md`: the knob docs.
  - `.gitignore`: `/.venv-vllm`.
- Test: `tests/test_scripts.py`, `tests/test_scheduler.py`

**Interfaces:**
- Consumes: Task 1's `find_vllm_python`; the trainers' and engine's `--gen-backend` (Tasks 3, 4);
  `ced_pseudo_label.py --gen-backend` (Task 5).
- Produces:
  - `gpu_defaults` prints `"<slots> <phys> <mps> <gpu MB> <lora MB> <gen>"`;
  - `apply_card_defaults` exports `GEN_BACKEND`;
  - `vllm_check <python>` returns 1 when `GEN_BACKEND=vllm` and no vLLM environment is found;
  - the runners read `GEN_BACKEND` (default `hf`).

- [ ] **Step 1: Update and add the script tests (failing first)**

In `tests/test_scripts.py`, the `gpu_defaults` cases become:

```python
@pytest.mark.parametrize("mib,expected", [
    ("143771", "3 8 1 39833 18432 hf"),    # H200 NVL: the values tools/bench_gpu.sh measured
    ("46068", "1 - 0 - - hf"),             # 46 GB cards: one run per GPU, the runners' own settings
    ("0", "1 - 0 - - hf"),                 # no nvidia-smi
])
```

In `test_apply_card_defaults_fills_only_what_the_caller_left_unset`, add
`gen=${GEN_BACKEND:-unset}` to the echoed line. Add `"GEN_BACKEND"` to the stripped variables,
and change the cases to:

```python
@pytest.mark.parametrize("mib,preset,expected", [
    ("143771", "", "slots=3 phys=8 mps=1 gpu=39833 lora=18432 gen=hf"),
    ("46068", "", "slots=1 phys=unset mps=0 gpu=unset lora=unset gen=hf"),
    ("143771", "PHYS_BS=16 SLOTS_PER_GPU=2 USE_MPS=0 GEN_BACKEND=vllm", "slots=2 phys=16 mps=0 gpu=39833 lora=18432 gen=vllm"),
])
```

The echo command in that test becomes:

```python
                          'echo "slots=$SLOTS_PER_GPU phys=${PHYS_BS:-unset} mps=$USE_MPS '
                          'gpu=${NEED_GPU_MB:-unset} lora=${NEED_LORA_MB:-unset} gen=${GEN_BACKEND:-unset}"'],
```

Its environment filter becomes:

```python
                              if k not in ("PHYS_BS", "SLOTS_PER_GPU", "USE_MPS", "NEED_GPU_MB", "NEED_LORA_MB",
                                           "GEN_BACKEND")},
```

Append the new tests:

```python
import sys


def vllm_python(tmp_path):
    """A VLLM_PY that tells gen_backend.find_vllm_python() it holds vLLM 0.27.1."""
    python = tmp_path / "vllm_python"
    python.write_text('#!/bin/bash\nif [ "$1" = "-c" ]; then echo "VLLM_VERSION 0.27.1"; exit 0; fi\n')
    python.chmod(0o755)
    return str(python)


@pytest.mark.parametrize("backend,found,rc", [("hf", False, 0), ("vllm", False, 1), ("vllm", True, 0)])
def test_vllm_check_stops_only_a_vllm_run_without_vllm(tmp_path, backend, found, rc):
    vllm_py = vllm_python(tmp_path) if found else str(tmp_path / "missing")
    out = subprocess.run(["bash", "-c", f"source {LIB}; vllm_check {sys.executable}"],
                         env={**os.environ, "GEN_BACKEND": backend, "VLLM_PY": vllm_py},
                         capture_output=True, text=True)
    assert out.returncode == rc, out.stderr
    if rc:
        assert "no vLLM environment" in out.stderr


@pytest.mark.parametrize("runner,args", [
    ("scripts/qwen/ced/run_cllora.sh", ["--method", "inclora", "--data-root", "data/tacred_perm0", "--num-tasks", "10"]),
    ("scripts/qwen/ced/run_ced_v2.sh", ["--run-name", "vllm_refusal_probe", "--gpus", "0"]),
])
def test_runners_refuse_vllm_without_an_environment_before_any_run_directory(runner, args, tmp_path):
    save = tmp_path / "run"
    extra = ["--py", sys.executable, "--save", str(save)] if "cllora" in runner else []
    env = {**os.environ, "GEN_BACKEND": "vllm", "VLLM_PY": str(tmp_path / "missing"),
           "ENV_BIN": os.path.dirname(sys.executable)}
    out = subprocess.run(["bash", runner] + args + extra, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode != 0
    assert "no vLLM environment" in out.stdout + out.stderr
    assert not save.exists() and not os.path.exists("results/qwen3/ced/vllm_refusal_probe")


def test_the_ced_runner_records_the_backend_in_its_manifest():
    script = open("scripts/qwen/ced/run_ced_v2.sh").read()
    manifest_line = next(line for line in script.splitlines() if line.startswith("MANIFEST_CONFIG="))
    assert ";gen=${GEN_BACKEND};" in manifest_line


def test_the_runners_hand_the_backend_on():
    ced = open("scripts/qwen/ced/run_ced_v2.sh").read()
    assert 'OPTS+=" --gen-backend ${GEN_BACKEND}"' in ced and 'PL_OPTS+=" --gen-backend ${GEN_BACKEND}"' in ced
    assert '--gen-backend "${GEN_BACKEND:-hf}"' in open("scripts/qwen/ced/run_cllora.sh").read()
```

In `tests/test_scheduler.py`, the two expected step-3 lines become:

```python
    assert train_step_line(tmp_path, 143771) == \
        "=== 3. train 16 jobs on gpus 0 1 0 1 0 1 (PHYS_BS=8 USE_MPS=1 GEN_BACKEND=hf) ==="
```

```python
    assert train_step_line(tmp_path, 46068) == \
        "=== 3. train 16 jobs on gpus 0 1 (PHYS_BS=runner default USE_MPS=0 GEN_BACKEND=hf) ==="
```

`train_step_line`'s environment filter also drops `"GEN_BACKEND"`.

- [ ] **Step 2: Run them to see them fail**

Run: `dev_pytest tests/test_scripts.py tests/test_scheduler.py`
Expected failures:
- the `gpu_defaults` and `apply_card_defaults` cases (five values, no `gen=`);
- `test_vllm_check_*` (`vllm_check: command not found`);
- the runner refusal cases (a runner does not know `GEN_BACKEND`, so it either runs on or fails
  on something else);
- the manifest and hand-on text checks;
- the scheduler lines.

- [ ] **Step 3: Change `scripts/qwen/lib.sh`**

The comment and body of `gpu_defaults`:

```bash
# gpu_defaults <MiB of the smallest card>
#   -> "<SLOTS_PER_GPU> <PHYS_BS> <USE_MPS> <NEED_GPU_MB> <NEED_LORA_MB> <GEN_BACKEND>" ("-" = leave unset)
# The values tools/bench_gpu.sh measured on 1x H200 NVL (README) go to H200-class cards only;
# smaller cards keep one run per GPU and the runners' own settings (PHYS_BS = --bs, no MPS).
# GEN_BACKEND stays hf until the vLLM end-to-end check passes (plan 2026-10-06, Task 8).
gpu_defaults () {
    if [ "${1:-0}" -ge 130000 ]; then echo "3 8 1 39833 18432 hf"; else echo "1 - 0 - - hf"; fi
}
```

In `apply_card_defaults`:
- add `d_gen` to the `local` line;
- the `read` line becomes `read -r d_slots d_phys d_mps d_gpu d_lora d_gen <<< "$(gpu_defaults "${mib}")"`;
- add as its last line `export GEN_BACKEND=${GEN_BACKEND:-${d_gen}}`;
- its comment names `GEN_BACKEND` among the filled variables.

Append:

```bash
# vllm_check <python>: with GEN_BACKEND=vllm, stop unless gen_backend.find_vllm_python() finds a vLLM
# environment (VLLM_PY, ./.venv-vllm or /venv/main). Prints the interpreter and the vLLM version.
vllm_check () {
    [ "${GEN_BACKEND:-hf}" = "vllm" ] || return 0
    "$1" -c "from gen_backend import find_vllm_python; print('vLLM', *find_vllm_python())"
}
```

- [ ] **Step 4: Change `scripts/qwen/ced/run_ced_v2.sh`**

- After the `STRICT_GEN=` knob line:
  `GEN_BACKEND=${GEN_BACKEND:-hf}         # vllm = answers and pseudo-labels from vLLM (gen_backend.py), same settings`.
- In the flag loop, after `--strict-gen)`: `--gen-backend) GEN_BACKEND=$2; shift 2;;`.
- After `export PATH=${ENV_BIN}:$PATH`, before anything touches the run directory:
  `vllm_check "${ENV_BIN}/python" || exit 1`.
- In `MANIFEST_CONFIG`, change `strict_gen=${STRICT_GEN};extra=` to
  `strict_gen=${STRICT_GEN};gen=${GEN_BACKEND};extra=`.
- On the `echo "run=..."` line, after `strict_gen=${STRICT_GEN}`, add ` gen=${GEN_BACKEND}`.
- In the `OPTS` function, after the `STRICT_GEN` line: `OPTS+=" --gen-backend ${GEN_BACKEND}"`.
- After `PL_OPTS+=" --lexicon-filter ${PL_LEXICON}"`: `PL_OPTS+=" --gen-backend ${GEN_BACKEND}"`.

- [ ] **Step 5: Change `scripts/qwen/ced/run_cllora.sh`**

- After `[ -x "${PY}" ] || { echo "python not executable: ${PY}"; exit 1; }`:
  `vllm_check "${PY}" || exit 1`.
- In the engine command, after `--limit "${LIMIT}" --model-path "${MODEL_PATH}" \`, add a line:
  `    --gen-backend "${GEN_BACKEND:-hf}" \`.

- [ ] **Step 6: Change `project_commands.sh`, `README.md` and `.gitignore`**

- `project_commands.sh`:
  - the step line becomes
    `step "3. train ${#JOBS[@]} jobs on gpus ${GPUS[*]} (PHYS_BS=${PHYS_BS:-runner default} USE_MPS=${USE_MPS} GEN_BACKEND=${GEN_BACKEND})"`;
  - in the header's knob list, after `USE_MPS`, add
    `#   GEN_BACKEND   hf | vllm: where evaluation answers and pseudo-labels come from (default: by card size)`.
- `README.md`, H200 section: after the **Decoding (read this).** bullet, add:

```markdown
- **Generation backend.** `GEN_BACKEND=vllm` generates the evaluation answers and the teacher's
  pseudo-labels with vLLM (`gen_backend.py`, `tools/vllm_generate.py`).
  - vLLM gets the settings transformers' `generate()` would use, including the fill-in above,
    and the same prompt tokens.
  - Text handling and scoring stay as they are.
  - GainLoRA and EPI keep Hugging Face generation, because they pick adapters per input. The
    CL-LoRA manifest records which backend each run used.
  - vLLM runs in its own environment. `VLLM_PY` points at its Python; otherwise
    `./.venv-vllm` or `/venv/main` is used. To create one:
    `uv venv .venv-vllm --python 3.12 && uv pip install --python .venv-vllm vllm==0.27.1`.
  - `VLLM_GPU_GB` (default 10) caps each vLLM instance. `VLLM_EAGER=1` turns off CUDA graphs.
```

- `.gitignore`: add `/.venv-vllm`.

- [ ] **Step 7: Run the script tests to see them pass**

Run: `dev_pytest tests/test_scripts.py tests/test_scheduler.py`
Expected: all pass. The new ones are:
- `test_vllm_check_*` (3);
- `test_runners_refuse_*` (2);
- the manifest and hand-on checks (2).

- [ ] **Step 8: Commit**

```bash
cd ${REPO} && git add scripts/qwen/lib.sh scripts/qwen/ced/run_ced_v2.sh scripts/qwen/ced/run_cllora.sh \
    project_commands.sh README.md .gitignore tests/test_scripts.py tests/test_scheduler.py && \
git commit -q -m "feat: GEN_BACKEND in the runners and the scheduler

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Parity and speed on the GPU (`tools/gen_parity.py`)

**Files:**
- Create: `tools/gen_parity.py`, `tests/test_gen_parity.py`
- Modify: `gen_backend.py`, `EAGER_DEFAULT` only, if Step 7 rules eager mode the default.

**Interfaces:**
- Consumes:
  - `ced_eval.evaluate` and `check_gen_backend` (Task 3);
  - `cl_lora.engine.eval_task` and `lora_config` (Task 4);
  - `tools/ced_pseudo_label.py --gen-backend` (Task 5).
- Produces the measured numbers Task 8 writes into the README:
  - greedy agreement and F1, for CL-LoRA and for the distillation path in strict mode;
  - sampled F1 means and spreads;
  - pseudo-label agreement;
  - seconds per backend;
  - the engine-mode ruling.

- [ ] **Step 1: Write the failing helper tests, `tests/test_gen_parity.py`**

```python
"""tools/gen_parity.py's helpers (the GPU runs are Task 7's checks)."""
import importlib.util
import os

_SPEC = importlib.util.spec_from_file_location("gen_parity", os.path.join("tools", "gen_parity.py"))
gp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gp)


def test_the_last_test_line_of_a_log_gives_the_metrics(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text("dev | avg_loss: 0.1 | {}\n"
                   "test | avg_loss: 0.2 | {'trigger': {'f1': 0.5}}\n"
                   "test | avg_loss: 0.3 | {'exact_match': 1.0, 'trigger': {'precision': 0.7, 'recall': 0.7, 'f1': 0.7}}\n")
    assert gp.last_test_metrics(str(log))["trigger"]["f1"] == 0.7


def test_the_summary_compares_the_means_in_points():
    rows = [{"backend": "hf", "f1": 0.60}, {"backend": "hf", "f1": 0.62},
            {"backend": "vllm", "f1": 0.605}, {"backend": "vllm", "f1": 0.615}]
    assert gp.summary(rows) == {"hf": {"mean": 61.0, "spread": 2.0}, "vllm": {"mean": 61.0, "spread": 1.0},
                                "gap": 0.0}
```

Run: `dev_pytest tests/test_gen_parity.py`
Expected: an error collecting the file: `FileNotFoundError` for `tools/gen_parity.py`.

- [ ] **Step 2: Write `tools/gen_parity.py`**

```python
"""Hugging Face generate() against vLLM on one model and one test set (spec §8, checks 2, 3, 5).

  python tools/gen_parity.py ced --base DIR [--adapter DIR] --data-dir processed_data/<prefix><perm>/<task>/qwen/ \
      --out DIR [--strict] [--seeds 1 2 3] [--backends hf vllm]
  python tools/gen_parity.py cllora --base DIR --checkpoint <run>/checkpoint_latest.pt --method inclora \
      --data-root data/<ds>_perm<p> --task T --out DIR

Runs in the training environment on one GPU; vLLM comes from VLLM_PY. Prints one JSON line per run
(backend, seed, seconds, trigger F1, and the share of answers identical to the first backend's run
with the same seed), then a SUMMARY line with each backend's mean and spread in F1 points."""
import argparse
import ast
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def last_test_metrics(log_path):
    """The metrics of the last 'test | avg_loss: ... | {...}' line evaluate() wrote."""
    line = [line for line in open(log_path) if line.startswith("test |")][-1]
    return ast.literal_eval(line.split(" | ", 2)[2].strip())


def identical_share(first, second):
    return sum(a == b for a, b in zip(first, second)) / len(first)


def summary(rows):
    """Mean and spread (max - min) of trigger F1 per backend, and the gap between the means, in points."""
    out = {}
    for backend in sorted({row["backend"] for row in rows}):
        f1 = [100 * row["f1"] for row in rows if row["backend"] == backend]
        out[backend] = {"mean": round(statistics.mean(f1), 3), "spread": round(max(f1) - min(f1), 3)}
    if len(out) == 2:
        first, second = (value["mean"] for value in out.values())
        out["gap"] = round(abs(first - second), 3)
    return out


def report(rows, row, first, answers):
    if first is not None:
        row["identical"] = round(identical_share(first, answers), 4)
    rows.append(row)
    print(json.dumps(row), flush=True)


def ced(a):
    import torch
    import torch.distributed as dist
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from arguments import get_args
    from ced_eval import check_gen_backend, evaluate
    from data_utils.lm_datasets import LMTrainDataset
    from utils import get_tokenizer

    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{a.port}", rank=0, world_size=1)
    sys.argv = ["gen_parity", "--model-path", a.base, "--model-type", "qwen", "--type", "lm", "--data-dir", a.data_dir,
                "--max-length", "768", "--max-prompt-length", "460", "--eval-batch-size", "128",
                "--eval-loss-batch-size", "32", "--top-k", "0", "--top-p", "0.95", "--temperature", "0.5",
                "--eval-gen", "--num-workers", "0"] + (["--strict-generation"] if a.strict else [])
    args = get_args()
    args.dynamic_pad_effective = False
    tokenizer = get_tokenizer(args)
    test = LMTrainDataset(args, tokenizer, a.data_dir, "test", -1, 1, random.Random(0))
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).cuda()
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    rows = []
    for seed in a.seeds:
        first = None
        for backend in a.backends:
            args.seed, args.gen_backend = seed, backend
            args.save = os.path.join(a.out, f"{backend}_s{seed}")
            os.makedirs(args.save, exist_ok=True)
            check_gen_backend(args, model, tokenizer)
            torch.manual_seed(seed)
            start = time.time()
            evaluate(args, tokenizer, model, test, "test", 0, "cuda")
            seconds = round(time.time() - start, 1)
            answers = [json.loads(line)["text"] for line in open(os.path.join(args.save, "eval", "0", "answers.jsonl"))]
            f1 = last_test_metrics(os.path.join(args.save, "log.txt"))["trigger"]["f1"]
            report(rows, {"backend": backend, "seed": seed, "rows": len(answers), "seconds": seconds, "f1": f1},
                   first, answers)
            first = first if first is not None else answers
    print("SUMMARY", json.dumps(summary(rows)), flush=True)


def cllora(a):
    import torch
    from peft import get_peft_model, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from cl_lora import engine
    from cl_lora.multi_adapter import CLLoRAManager
    from gen_backend import find_vllm_python

    os.makedirs(a.out, exist_ok=True)
    args = argparse.Namespace(cl_method=a.method, rank=16, alpha=64, dropout=0.1, data_root=a.data_root,
                              max_length=768, max_prompt_length=460, limit=-1, eval_batch_size=128,
                              model_path=a.base, save=a.out, gen_backend="hf", vllm_py=None)
    # the engine's own checkpoint, loaded as engine.main() loads it: it holds RNG and manager
    # state that weights_only=True refuses
    checkpoint = torch.load(a.checkpoint, map_location="cuda", weights_only=False)
    tok = AutoTokenizer.from_pretrained(a.base)
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16),
                           engine.lora_config(args), adapter_name="task0").cuda()
    mgr = CLLoRAManager(model, orth_lambda=0.0)
    mgr.register_task0("task0")
    for task_id in range(1, checkpoint["completed_task"] + 1):
        mgr.start_new_task(task_id, engine.lora_config(args))
    for adapter, state in checkpoint["adapters"].items():
        set_peft_model_state_dict(model, state, adapter_name=adapter)
    rows, first = [], None
    for backend in a.backends:
        args.gen_backend = backend
        args.vllm_py = find_vllm_python()[0] if backend == "vllm" else None
        start = time.time()
        metrics, preds, _ = engine.eval_task(args, model, tok, "cuda", a.task, mgr)
        report(rows, {"backend": backend, "rows": len(preds), "seconds": round(time.time() - start, 1),
                      "f1": metrics["trigger"]["f1"]}, first, preds)
        first = first if first is not None else preds
    print("SUMMARY", json.dumps(summary(rows)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("ced")
    c.add_argument("--base", required=True)
    c.add_argument("--adapter", default=None)
    c.add_argument("--data-dir", required=True)
    c.add_argument("--strict", action="store_true")
    c.add_argument("--seeds", type=int, nargs="+", default=[1])
    c.add_argument("--port", type=int, default=29581)
    cl = sub.add_parser("cllora")
    cl.add_argument("--base", required=True)
    cl.add_argument("--checkpoint", required=True)
    cl.add_argument("--method", required=True, choices=["inclora", "olora", "inflora", "tree"])
    cl.add_argument("--data-root", required=True)
    cl.add_argument("--task", type=int, required=True)
    for parser in (c, cl):
        parser.add_argument("--out", required=True)
        parser.add_argument("--backends", nargs="+", default=["hf", "vllm"], choices=["hf", "vllm"])
    a = ap.parse_args()
    ced(a) if a.mode == "ced" else cllora(a)


if __name__ == "__main__":
    main()
```

Run: `dev_pytest tests/test_gen_parity.py`
Expected: `2 passed`.

- [ ] **Step 3: Check 2, greedy parity on CL-LoRA**

```bash
sync_dev && ${BOX} "${DEV_GPU_ENV} && .venv/bin/python tools/gen_parity.py cllora --base models/Qwen3-0.6B \
    --checkpoint /workspace/dev_runs/results_2026-10-05/qwen3/ced/cllora_inclora_perm0_tacred_cre_s42/checkpoint_latest.pt \
    --method inclora --data-root data/tacred_perm0 --task 9 --out /workspace/vllm_checks/cllora 2>&1 \
    | grep -E '^\{|SUMMARY|Error' | tail -5"
```
Expected:
- an `hf` line with `"rows": 1240` and `"f1"` 0.6306 (±0.002). That is round 1's task-9 result
  from the same checkpoint. Another value means the rebuilt model differs from the trained one:
  stop and investigate.
- a `vllm` line with `"identical"` ≥ 0.99, and `"f1"` within 0.003 of the hf line (0.3 points).

- [ ] **Step 4: Check 3, distillation-path parity: greedy (strict), then sampled over 3 seeds**

```bash
A=/workspace/dev_runs/results_2026-10-05/qwen3/ced/bench_t0_fewrel/task0/e1-bs16-lr0.0002-G2-N1-NN1-lora-16-64-0.1/40
${BOX} "${DEV_GPU_ENV} && .venv/bin/python tools/gen_parity.py ced --base models/Qwen3-0.6B --adapter ${A} \
    --data-dir processed_data/fewrel_perm0/0/qwen/ --strict --seeds 1 --out /workspace/vllm_checks/ced_greedy 2>&1 \
    | grep -E '^\{|SUMMARY|Error' | tail -4"
${BOX} "${DEV_GPU_ENV} && .venv/bin/python tools/gen_parity.py ced --base models/Qwen3-0.6B --adapter ${A} \
    --data-dir processed_data/fewrel_perm0/0/qwen/ --seeds 1 2 3 --out /workspace/vllm_checks/ced_sampled 2>&1 \
    | grep -E '^\{|SUMMARY|Error' | tail -8"
```
Expected:
- **Strict greedy:** the `vllm` line has `"identical"` ≥ 0.95 and `"rows": 1120`. This is
  informational: the LoRA path under true greedy decoding.
- **Sampled (today's evaluation settings):** six lines, then `SUMMARY`. It passes when either
  holds:
  - `gap` ≤ the larger of `hf.spread` and `vllm.spread`;
  - `gap` ≤ 1.0.

  Otherwise stop and investigate.

- [ ] **Step 5: Check 4, pseudo-label parity (ACE task 1, confidence filter on)**

```bash
T=/workspace/dev_runs/results_2026-10-05/qwen3/ced/bench_t0_ace/task0/merged
${BOX} "${DEV_GPU_ENV} && for b in hf vllm; do s=\$(date +%s); .venv/bin/python tools/ced_pseudo_label.py \
    --teacher ${T} --data-dir data/ace_b10_perm0/1 --streams data/ace_b10_perm0/streams.json --task-id 1 \
    --batch-size 32 --conflict-dedup 1 --conf-filter percentile --conf-percentile 70 --lexicon-filter 1 \
    --gen-backend \$b --gpu 0 --out /workspace/vllm_checks/pl_\$b > /workspace/vllm_checks/pl_\$b.log 2>&1; \
    echo \"PL \$b exit \$? seconds \$(( \$(date +%s) - s ))\"; done; .venv/bin/python - <<'EOF'
import json
a = [json.loads(l) for l in open('/workspace/vllm_checks/pl_hf/train.jsonl')]
b = [json.loads(l) for l in open('/workspace/vllm_checks/pl_vllm/train.jsonl')]
sa = json.load(open('/workspace/vllm_checks/pl_hf/pl_stats.json'))
sb = json.load(open('/workspace/vllm_checks/pl_vllm/pl_stats.json'))
print('PL rows identical', round(sum(x == y for x, y in zip(a, b)) / len(a), 4), 'of', len(a),
      '| aug_rows', sa['aug_rows'], sb['aug_rows'])
for k in ('mean', 'p10', 'p30', 'p50', 'p70', 'p90'):
    if k in sa.get('score_dist', {}):
        print('PL score', k, round(sa['score_dist'][k], 4), round(sb['score_dist'][k], 4))
EOF"
```
Expected:
- `PL hf exit 0` and `PL vllm exit 0`, each with its seconds;
- `PL rows identical` ≥ 0.99;
- each `PL score` pair within 0.01.

- [ ] **Step 6: Check 5, speed on the largest test set, and the engine mode**

```bash
A=/workspace/dev_runs/results_2026-10-05/qwen3/ced/bench_t0_fewrel/task0/e1-bs16-lr0.0002-G2-N1-NN1-lora-16-64-0.1/40
R="tools/gen_parity.py ced --base models/Qwen3-0.6B --adapter ${A} --seeds 1"
${BOX} "${DEV_GPU_ENV} && for d in 9 0; do \
    .venv/bin/python ${R} --data-dir processed_data/fewrel_perm0/\$d/qwen/ --backends hf --out /workspace/vllm_checks/speed_t\$d_hf; \
    for e in 0 1; do VLLM_EAGER=\$e .venv/bin/python ${R} --data-dir processed_data/fewrel_perm0/\$d/qwen/ \
        --backends vllm --port 2959\$e --out /workspace/vllm_checks/speed_t\$d_e\$e; done; done 2>&1 \
    | grep -E '^\{|Error' | tail -8"
${BOX} "${DEV_GPU_ENV} && (VLLM_EAGER=0 .venv/bin/python ${R} --data-dir processed_data/fewrel_perm0/0/qwen/ \
    --backends vllm --port 29597 --out /workspace/vllm_checks/conc_a & \
    VLLM_EAGER=0 .venv/bin/python ${R} --data-dir processed_data/fewrel_perm0/0/qwen/ \
    --backends vllm --port 29598 --out /workspace/vllm_checks/conc_b; wait) 2>&1 | grep -E '^\{|Error' | tail -4"
```
Expected:
- **Speed runs:** six JSON lines.
  - Task 9 (`"rows": 11200`): hf, vllm with graphs (`e0`), vllm eager (`e1`).
  - Task 0 (`"rows": 1120`): the same three.
  - On each set, the vllm `seconds` are well below the hf `seconds`. If vLLM is not faster,
    stop and investigate.
- **Concurrency run:** two vllm lines, both with `"rows": 1120`. Two instances sharing vLLM's
  compile cache both finish.

- [ ] **Step 7: Rule on the engine mode**

Add each mode's `seconds` over the two sets: graphs (`e0` lines) against eager (`e1` lines).

- If eager is lower: set `EAGER_DEFAULT = "1"` in `gen_backend.py`, and ledger
  `Task 7: Ruling: VLLM_EAGER default 1 — eager <sum> s vs graphs <sum> s over FewRel test sets of 11,200 and 1,120 rows`.
  Then run `dev_pytest tests/test_gen_backend.py`; it must still pass. Commit:

```bash
cd ${REPO} && git add gen_backend.py && git commit -q -m "perf: vLLM without CUDA graphs by default (measured)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- Otherwise: keep `"0"` and ledger the two sums as a note.

- [ ] **Step 8: Commit the tool**

```bash
cd ${REPO} && git add tools/gen_parity.py tests/test_gen_parity.py && \
git commit -q -m "feat: tools/gen_parity.py, Hugging Face against vLLM on one model

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Ledger every check's numbers in one line each: checks 2 to 5, with the Expected comparison.

---

### Task 8: End-to-end check, the H200 default, README

**Files:**
- Modify:
  - `scripts/qwen/lib.sh`: `gpu_defaults` gives `vllm` on H200-class cards, and
    `apply_card_defaults` falls back to `hf` without vLLM. Also `NEED_GPU_MB`, if Step 3 measures
    a larger peak.
  - `README.md`: results.
- Test: `tests/test_scripts.py`, `tests/test_scheduler.py`

**Interfaces:**
- Consumes: everything above; the round-1 numbers (RKL 65.56, IncLoRA 63.06; 3,889 s and
  4,213 s).
- Produces: the default `GEN_BACKEND=vllm` on H200-class cards when a vLLM environment is found.

- [ ] **Step 1: Launch TACRED order 0, RKL and IncLoRA, with `GEN_BACKEND=vllm`, in the dev tree**

```bash
sync_dev && ${BOX} 'cat > /workspace/vllm_checks/e2e.sh' <<'EOF'
#!/bin/bash
# TACRED perm0, RKL and IncLoRA, GEN_BACKEND=vllm, in the dev tree, next to the FewRel run
set -u
cd /workspace/opened_dev
export ENV_BIN=/workspace/opened_dev/.venv/bin PY=/workspace/opened_dev/.venv/bin/python HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/usr/local/cuda DISK_PATH=. \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_PY=/venv/main/bin/python GEN_BACKEND=vllm \
    PHYS_BS=8 NEED_GPU_MB=1000 NEED_LORA_MB=1000
C=/workspace/vllm_checks
t () { local s=$(date +%s); "$@"; echo "WALL $(( $(date +%s) - s ))"; }
(t env METHODS=rkl MASTER_PORT=29561 bash scripts/qwen/cre/run_cre_dist.sh tacred 0 0) > ${C}/e2e_rkl.log 2>&1 &
P1=$!
(t bash scripts/qwen/ced/run_cllora.sh --method inclora --data-root data/tacred_perm0 --num-tasks 10 \
    --batch-size 2 --grad-accum 16 --gpu 0 --py ${PY} --protocol tacred_cre --eval-batch-size 128) \
    > ${C}/e2e_inclora.log 2>&1 &
P2=$!
# GPU memory of this check's processes (trainers and their vLLM children) every 30 s
( while sleep 30; do
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | while IFS=', ' read -r pid mib; do
        cmd=$(tr '\0' ' ' < /proc/${pid}/cmdline 2>/dev/null)
        case "${cmd}" in *opened_dev*|*vllm_generate*) echo "$(date +%s) ${pid} ${mib} ${cmd:0:200}";; esac
    done
  done ) >> ${C}/e2e_mem.log 2>&1 &
M=$!
wait ${P1} ${P2}
kill ${M}
echo E2E_DONE
EOF
${BOX} "setsid nohup bash /workspace/vllm_checks/e2e.sh > /workspace/vllm_checks/e2e.out 2>&1 < /dev/null & disown; echo started"
```
Expected: `started`. Wait for `E2E_DONE` in `/workspace/vllm_checks/e2e.out` with a background
until-loop. It will be roughly an hour or more, since the card is shared with the FewRel run.

- [ ] **Step 2: Compare with round 1**

```bash
${BOX} 'cd /workspace/opened_dev && grep -h WALL /workspace/vllm_checks/e2e_rkl.log /workspace/vllm_checks/e2e_inclora.log; \
f=$(find results/qwen3/ced/cre_tacred_rkl_perm0/task9 -name log.txt | head -1); \
echo "rkl task9: $(grep "^test |" "$f" | tail -1 | grep -oE "'"'"'trigger'"'"': \{[^}]*" | grep -oE "f1.: [0-9.]+")"; \
.venv/bin/python -c "import json; d=json.load(open(\"results/qwen3/ced/cllora_inclora_perm0_tacred_cre_s42/cl_results.json\")); print(\"inclora task9:\", round(d[\"task9\"][\"trigger\"][\"f1\"], 4))"; \
grep -c -i -E "traceback|error" /workspace/vllm_checks/e2e_rkl.log /workspace/vllm_checks/e2e_inclora.log; \
grep -rh "generation backend" results/qwen3/ced/cre_tacred_rkl_perm0 logs/tacred_cllora_inclora_perm0_train.log 2>/dev/null | sort | uniq -c | head -3'
```
Expected:
- **WALL:** two lines, RKL and IncLoRA.
- **F1:**
  - `rkl task9: f1': X` with |X − 0.6556| ≤ 0.020 (round 1 sampled at T=0.5, about 1 point of
    noise);
  - `inclora task9: Y` with |Y − 0.6306| ≤ 0.020.
- **Errors:** 0 error lines in each log.
- **Backend:** at least one `generation backend: vLLM 0.27.1` line, which shows that vLLM did
  the evaluation.

If an F1 gap is above 2 points, stop and investigate before any default changes.

- [ ] **Step 3: Read the memory peaks**

```bash
${BOX} '/workspace/opened_dev/.venv/bin/python - <<"EOF"
from collections import defaultdict
peak = defaultdict(int)
by_time = defaultdict(lambda: defaultdict(int))
for line in open("/workspace/vllm_checks/e2e_mem.log"):
    t, pid, mib, cmd = line.split(" ", 3)
    run = "inclora" if "cllora" in cmd else "rkl"
    by_time[t][run] += int(mib)
for t, runs in by_time.items():
    for run, mib in runs.items():
        peak[run] = max(peak[run], mib)
print("MEM peak MiB per run (trainer + its vLLM):", dict(peak))
EOF'
```
Expected: `MEM peak MiB per run ...` with both runs at or below 39833, the H200 `NEED_GPU_MB`.

If a peak is above it: in Step 5, also raise the H200 `NEED_GPU_MB` in `gpu_defaults` to that
peak rounded up to the next 1024, and ledger a ruling.

- [ ] **Step 4: The H200 default, tests first**

In `tests/test_scripts.py`:
- the `gpu_defaults` H200 case expects `"3 8 1 39833 18432 vllm"`;
- the other two cases keep `hf`.

`test_apply_card_defaults_fills_only_what_the_caller_left_unset` gets a `vllm` parameter, and the
test sets `VLLM_PY` to `vllm_python(tmp_path)` when it is `"found"` and to
`str(tmp_path / "missing")` otherwise:

```python
@pytest.mark.parametrize("mib,preset,vllm,expected", [
    ("143771", "", "found", "slots=3 phys=8 mps=1 gpu=39833 lora=18432 gen=vllm"),
    ("143771", "", "missing", "slots=3 phys=8 mps=1 gpu=39833 lora=18432 gen=hf"),
    ("46068", "", "found", "slots=1 phys=unset mps=0 gpu=unset lora=unset gen=hf"),
    ("143771", "PHYS_BS=16 SLOTS_PER_GPU=2 USE_MPS=0 GEN_BACKEND=hf", "found",
     "slots=2 phys=16 mps=0 gpu=39833 lora=18432 gen=hf"),
])
def test_apply_card_defaults_fills_only_what_the_caller_left_unset(tmp_path, mib, preset, vllm, expected):
    bin_dir = fake_bin(tmp_path, **{"nvidia-smi": fake_smi(mib)})
    preset = f"export {preset};" if preset else ""
    vllm_py = vllm_python(tmp_path) if vllm == "found" else str(tmp_path / "missing")
    out = subprocess.run(["bash", "-c", f'source {os.path.abspath(LIB)}; {preset} apply_card_defaults 0 1; '
                          'echo "slots=$SLOTS_PER_GPU phys=${PHYS_BS:-unset} mps=$USE_MPS '
                          'gpu=${NEED_GPU_MB:-unset} lora=${NEED_LORA_MB:-unset} gen=${GEN_BACKEND:-unset}"'],
                         env={k: v for k, v in {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                                                "VLLM_PY": vllm_py, "PY": sys.executable}.items()
                              if k not in ("PHYS_BS", "SLOTS_PER_GPU", "USE_MPS", "NEED_GPU_MB", "NEED_LORA_MB",
                                           "GEN_BACKEND")},
                         capture_output=True, text=True)
    assert out.stdout.strip() == expected, out.stderr
```

Move `vllm_python` above this test.

In `tests/test_scheduler.py`:
- `train_step_line` sets `VLLM_PY` in its environment to a fake that reports vLLM. Use the
  `vllm_python` helper's body, written to `tmp_path / "vllm_python"`.
- The H200 line expects `GEN_BACKEND=vllm`; the 46 GB line keeps `GEN_BACKEND=hf`.

Run: `dev_pytest tests/test_scripts.py tests/test_scheduler.py`
Expected: the H200 cases fail (`gen=hf` where `gen=vllm` is expected).

- [ ] **Step 5: The H200 default, code**

In `scripts/qwen/lib.sh`:
- `gpu_defaults` gives `3 8 1 39833 18432 vllm` for ≥ 130000 MiB;
- its comment's last line becomes
  `# GEN_BACKEND vllm on H200-class cards since the end-to-end TACRED check (README); hf elsewhere.`;
- in `apply_card_defaults`, before the `GEN_BACKEND` export, add:

```bash
    # vLLM only where gen_backend.find_vllm_python() finds an environment for it
    if [ "${d_gen}" = "vllm" ] && ! "${PY:-python3}" -c "from gen_backend import find_vllm_python; find_vllm_python()" \
            > /dev/null 2>&1; then
        d_gen=hf
    fi
```

Run: `dev_pytest tests/test_scripts.py tests/test_scheduler.py`
Expected: all pass.

- [ ] **Step 6: README results**

In the H200 section, after the **Generation backend.** bullet's paragraph, add this text.
The numbers come from Task 7's ledger lines and from Steps 2 and 3:

```markdown
vLLM against Hugging Face `generate()`, measured on 1× H200 NVL next to a running FewRel job
(`tools/gen_parity.py`):

| check | result |
|---|---|
| CL-LoRA greedy, IncLoRA TACRED task 9 (1,240 rows) | (identical share), F1 (hf) → (vllm) |
| Distillation path, strict greedy, FewRel task 0 (1,120 rows) | (identical share) |
| Distillation path, sampled at T=0.5, 3 seeds each | mean F1 (hf ± spread) vs (vllm ± spread) |
| Pseudo-labels, ACE task 1, confidence filter on | (identical rows), score quantiles within (max gap) |
| Answer generation, FewRel task 9 (11,200 rows) | (hf s) → (vllm s) |
| Answer generation, FewRel task 0 (1,120 rows) | (hf s) → (vllm s) |

End to end, TACRED order 0, all 10 tasks, `GEN_BACKEND=vllm` against the round-1 runs above:
RKL final-task trigger F1 65.56 → (Step 2), IncLoRA 63.06 → (Step 2); wall-clock 3,889 s →
(Step 2) and 4,213 s → (Step 2), both on a shared card. Since this check, H200-class cards
default to `GEN_BACKEND=vllm` when a vLLM environment is found (`scripts/qwen/lib.sh`);
`GEN_BACKEND=hf` restores Hugging Face generation.
```

Fill every parenthesis with the measured value before committing.

- [ ] **Step 7: Commit**

```bash
cd ${REPO} && git add scripts/qwen/lib.sh tests/test_scripts.py tests/test_scheduler.py README.md && \
git commit -q -m "feat: vLLM generation by default on H200-class cards, after the end-to-end check

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Run: `dev_pytest tests/` to confirm the whole suite on the final tree.
Expected: every test passes.
