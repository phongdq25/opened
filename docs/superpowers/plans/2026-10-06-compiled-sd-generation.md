# Compiled Sampling for Self-Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sample with a static KV cache inside training steps (self-distillation, DistiLLM/AMiD
student generation), so transformers compiles the decoding forward. Do it behind
`--compile-generation` / `COMPILE_GEN=1`, and turn it on for H200-class cards after a training
check.

**Architecture:** `gen_config.train_generation_kwargs(args)` adds `cache_implementation="static"`
to the keyword arguments of the two in-training `generate()` calls. transformers 4.57.3 then
compiles the decoding step (CUDA graphs). Evaluation keeps its own keyword arguments.

**Tech Stack:** Python 3.11, PyTorch 2.9.1, transformers 4.57.3, peft 0.18.0, DeepSpeed, bash
runners.

**Spec:** `docs/superpowers/specs/2026-10-06-compiled-sd-generation-design.md`

## Global Constraints

- **Same sampling:** only the cache layout and the compiled forward change.
- **Same objectives.**
- **Off by default** until Task 3's check passes; then on for H200-class cards only.
- **Evaluation is not affected.**
- **Never touch `/workspace/opened`.** Development and checks use `/workspace/opened_dev`, as in
  the vLLM plan's "Working on the box" (`sync_dev`, `dev_pytest`, `DEV_GPU_ENV`).
- **Git:** commit only on `perf/fast-generation`; every commit message ends with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A short last batch of an epoch** makes generation recompile once for its size.
   - It must still produce answers, and its row count must reach `sd_prepare`'s parsing
     unchanged.
   - Covered by Task 3's runs: 320 rows at physical batch 8 have no short batch, but the eager
     and compiled losses are compared step by step.
2. **Resuming a run under the other `COMPILE_GEN`** is refused. Task 2 adds
   `test_the_ced_runner_records_compiled_generation_in_its_manifest`.
3. **`--strict-generation` together with `--compile-generation`:** both keyword arguments
   reach `generate()`. Task 1 adds `test_strict_and_compiled_generation_combine`.
4. **Evaluation never compiles,** even with the flag on. Task 1 adds
   `test_evaluation_keeps_its_own_generation_kwargs`.
5. **The DistiLLM sampler under the flag.** Task 1 adds
   `test_student_generation_samples_with_a_static_cache_when_compiling`; Task 3 runs it.

---

### Task 1: `--compile-generation` and the two in-training generators

**Files:**
- Modify:
  - `gen_config.py`: add `train_generation_kwargs`.
  - `arguments.py`: add `--compile-generation`.
  - `ced_losses.py` (`sd_prepare`) and `distillm/sampler.py` (`run_sample`): use
    `train_generation_kwargs`.
  - `tests/conftest.py`: `ced_args` gets `compile_generation=False`.
- Test: `tests/test_compiled_generation.py`

**Interfaces:**
- Produces: `gen_config.train_generation_kwargs(args) -> dict`, and the flag
  `--compile-generation` (`args.compile_generation`).

- [ ] **Step 1: Write the failing tests**

`tests/conftest.py`: in `ced_args`, extend the last line of `base` with `compile_generation=False,`.

`tests/test_compiled_generation.py`:

```python
"""--compile-generation: a static KV cache for the generate() calls inside training steps."""
import inspect
from argparse import Namespace

import pytest
import torch

from conftest import ced_args


class Stop(Exception):
    pass


class RecordingModel(torch.nn.Module):
    def __init__(self, seen):
        super().__init__()
        self.seen = seen

    def generate(self, **kwargs):
        self.seen.update(kwargs)
        raise Stop


@pytest.mark.parametrize("strict,compiled,expected", [
    (False, False, {}),
    (False, True, {"cache_implementation": "static"}),
    (True, False, {"use_model_defaults": False}),
])
def test_train_generation_kwargs_add_the_static_cache_only_when_compiling(strict, compiled, expected):
    from gen_config import train_generation_kwargs
    assert train_generation_kwargs(Namespace(strict_generation=strict, compile_generation=compiled)) == expected


def test_strict_and_compiled_generation_combine():
    from gen_config import train_generation_kwargs
    assert train_generation_kwargs(Namespace(strict_generation=True, compile_generation=True)) == \
        {"use_model_defaults": False, "cache_implementation": "static"}


@pytest.mark.parametrize("compiled", [False, True])
def test_self_distillation_samples_with_a_static_cache_when_compiling(tokenizer, compiled):
    import ced_losses
    seen = {}
    gen_data = {"input_ids": torch.full((1, 4), 7), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    with pytest.raises(Stop):
        ced_losses.sd_prepare(ced_args(compile_generation=compiled), tokenizer, RecordingModel(seen), {},
                              gen_data, {"t_prompt_ids": []}, "cpu")
    assert seen.get("cache_implementation") == ("static" if compiled else None)


@pytest.mark.parametrize("compiled", [False, True])
def test_student_generation_samples_with_a_static_cache_when_compiling(compiled):
    from distillm.sampler import SampleGenerator
    args = ced_args(compile_generation=compiled, max_length=20, max_prompt_length=10)
    sampler = SampleGenerator(args, Namespace(pad_token_id=0, eos_token_id=1))
    seen = {}
    gen_data = {"input_ids": torch.full((1, 10), 7), "attention_mask": torch.ones(1, 10, dtype=torch.long)}
    with pytest.raises(Stop):
        sampler.run_sample(RecordingModel(seen), gen_data)
    assert seen.get("cache_implementation") == ("static" if compiled else None)


def test_evaluation_keeps_its_own_generation_kwargs():
    import ced_eval
    source = inspect.getsource(ced_eval)
    assert "train_generation_kwargs" not in source and "**generation_kwargs(args)" in source
```

- [ ] **Step 2: Run them to see them fail**

Run: `dev_pytest tests/test_compiled_generation.py`
Expected: failures with `ImportError: cannot import name 'train_generation_kwargs'`. The
`sd_prepare` and `run_sample` cases with `compiled=True` fail on
`assert None == 'static'`.

- [ ] **Step 3: Implement**

`gen_config.py`, append:

```python
def train_generation_kwargs(args):
    """generation_kwargs(args) for the generate() calls inside training steps (self-distillation,
    DistiLLM/AMiD). With --compile-generation they also ask for a static KV cache, and transformers
    then compiles the decoding forward (CUDA graphs). Evaluation keeps generation_kwargs: its batch
    sizes vary from call to call, and every new size would cost a compile."""
    kwargs = generation_kwargs(args)
    if getattr(args, "compile_generation", False):
        kwargs["cache_implementation"] = "static"
    return kwargs
```

`arguments.py`, in `add_gen_args` after `--gen-backend`:

```python
    group.add_argument("--compile-generation", action="store_true",
                       help="sample inside training steps (self-distillation, DistiLLM/AMiD) with a static KV "
                            "cache, which transformers compiles (gen_config.train_generation_kwargs)")
```

`ced_losses.py`:
- the import becomes `from gen_config import generation_kwargs, train_generation_kwargs`, or
  only `train_generation_kwargs` if nothing else uses `generation_kwargs` there;
- in `sd_prepare`, `**generation_kwargs(args)` becomes `**train_generation_kwargs(args)`.

`distillm/sampler.py`:
- the import becomes `from gen_config import train_generation_kwargs`;
- in `run_sample`, `**generation_kwargs(self.args)` becomes `**train_generation_kwargs(self.args)`.

- [ ] **Step 4: Run the tests to see them pass**

Run: `dev_pytest tests/test_compiled_generation.py tests/test_gen_config.py`
Expected: `9 passed` from `test_compiled_generation.py`; `test_gen_config.py` passes as before.

- [ ] **Step 5: Commit**

```bash
cd ${REPO} && git add gen_config.py arguments.py ced_losses.py distillm/sampler.py tests/conftest.py \
    tests/test_compiled_generation.py && \
git commit -q -m "feat: --compile-generation, a static KV cache for sampling inside training steps

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `COMPILE_GEN` in the runner, the manifest and the card defaults

**Files:**
- Modify:
  - `scripts/qwen/lib.sh`: `gpu_defaults` gets a seventh value, `COMPILE_GEN` (0), and
    `apply_card_defaults` exports it.
  - `scripts/qwen/ced/run_ced_v2.sh`: the knob, the flag, `MANIFEST_CONFIG` and the echo line.
  - `project_commands.sh`: the step-3 line and the header's knob list.
  - `README.md`: the knob.
- Test: `tests/test_scripts.py`, `tests/test_scheduler.py`

**Interfaces:**
- Consumes: Task 1's `--compile-generation`.
- Produces: `gpu_defaults` prints
  `"<slots> <phys> <mps> <gpu MB> <lora MB> <gen backend> <compile gen>"`, and
  `apply_card_defaults` exports `COMPILE_GEN`.

- [ ] **Step 1: Update and add the tests (failing first)**

In `tests/test_scripts.py`:
- every expected `gpu_defaults` string gets ` 0` appended; for example `"3 8 1 39833 18432 hf"`
  becomes `"3 8 1 39833 18432 hf 0"`;
- `apply_card_defaults` expectations get ` cgen=0`, and the echo gets
  ` cgen=${COMPILE_GEN:-unset}`;
- `"COMPILE_GEN"` joins the stripped variables, and the preset case adds `COMPILE_GEN=1`
  (expected `cgen=1`).

Append:

```python
def test_the_ced_runner_records_compiled_generation_in_its_manifest():
    script = open("scripts/qwen/ced/run_ced_v2.sh").read()
    manifest_line = next(line for line in script.splitlines() if line.startswith("MANIFEST_CONFIG="))
    assert ";cgen=${COMPILE_GEN};" in manifest_line
    assert '[ "${COMPILE_GEN}" = "1" ] && OPTS+=" --compile-generation"' in script
```

In `tests/test_scheduler.py`:
- each expected step-3 line gets ` COMPILE_GEN=0` after the `GEN_BACKEND` value;
- `"COMPILE_GEN"` joins the stripped variables.

Run: `dev_pytest tests/test_scripts.py tests/test_scheduler.py`
Expected: the `gpu_defaults`, `apply_card_defaults`, manifest and scheduler-line cases fail.

- [ ] **Step 2: Implement**

`scripts/qwen/lib.sh`:
- `gpu_defaults` prints a seventh value, `0`, for both card classes;
- its comment names `<COMPILE_GEN>`;
- `apply_card_defaults` reads `d_cgen` and exports `COMPILE_GEN=${COMPILE_GEN:-${d_cgen}}`.

`scripts/qwen/ced/run_ced_v2.sh`:
- knob `COMPILE_GEN=${COMPILE_GEN:-0}        # 1 = sample inside training steps with a static, compiled cache`;
- flag `--compile-gen) COMPILE_GEN=$2; shift 2;;`;
- in `OPTS`: `[ "${COMPILE_GEN}" = "1" ] && OPTS+=" --compile-generation"`;
- `MANIFEST_CONFIG`: `gen=${GEN_BACKEND};` becomes `gen=${GEN_BACKEND};cgen=${COMPILE_GEN};`;
- the echo line gets ` cgen=${COMPILE_GEN}`.

`project_commands.sh`:
- the step-3 line gets ` COMPILE_GEN=${COMPILE_GEN}`;
- the header's knob list gets
  `#   COMPILE_GEN   1 = compiled sampling inside training steps (default: by card size)`.

`README.md`, after the **Generation backend.** bullet:

```markdown
- **Compiled sampling.** `COMPILE_GEN=1` (`--compile-generation`) samples inside training
  steps (self-distillation in Ours, DistiLLM/AMiD student generation) with a static KV cache,
  which transformers compiles. The settings and the random stream are unchanged.
```

Run: `dev_pytest tests/test_scripts.py tests/test_scheduler.py`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
cd ${REPO} && git add scripts/qwen/lib.sh scripts/qwen/ced/run_ced_v2.sh project_commands.sh README.md \
    tests/test_scripts.py tests/test_scheduler.py && \
git commit -q -m "feat: COMPILE_GEN in the runner, the manifest and the card defaults

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Training check on the GPU, the H200 default, README

**Files:**
- Modify:
  - `scripts/qwen/lib.sh`: `gpu_defaults` gives `COMPILE_GEN` 1 on H200-class cards, if the
    check passes.
  - `README.md`: results.
- Test: `tests/test_scripts.py`, `tests/test_scheduler.py`

- [ ] **Step 1: Give the dev tree the task-0 sources the bench flags name**

```bash
${BOX} 'cd /workspace/opened_dev && mkdir -p results/qwen3/ced && for r in bench_t0_ace bench_t0_fewrel; do
    [ -e results/qwen3/ced/$r ] || cp -a /workspace/dev_runs/results_2026-10-05/qwen3/ced/$r results/qwen3/ced/; done;
    ls results/qwen3/ced'
```
Expected: `bench_t0_ace` and `bench_t0_fewrel` are listed.

- [ ] **Step 2: Ours and DistiLLM, eager then compiled (physical batch 8, one GPU, sequential)**

```bash
sync_dev && ${BOX} 'cat > /workspace/vllm_checks/cgen.sh' <<'EOF'
#!/bin/bash
# Ours (ACE task 1, 320 rows) and DistiLLM (FewRel task 1, 320 rows), COMPILE_GEN 0 then 1
set -u
cd /workspace/opened_dev
export ENV_BIN=/workspace/opened_dev/.venv/bin PY=/workspace/opened_dev/.venv/bin/python HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/usr/local/cuda DISK_PATH=. \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PHYS_BS=8 NEED_GPU_MB=1000 NEED_LORA_MB=1000
BASE=(--rank 16 --alpha 64 --lr 0.0002 --seed 42 --greedy 1 --gpus 0 --epochs 1 --bs 2 --acc 16 --extra "--log-interval 5")
OURS=(--mode ce_kd --kd-type sfkl --w-span 2.0 --kd-ratio 0.9 --skew 0.1 --span-metric cosine --layers "22 25 28"
      --data-prefix ace_b10_perm --perm 0 --pl 1 --replay-boost 5 --pl-conf percentile --pl-conf-pct 70 --sd 1
      --start-task 1 --end-task 1 --task0-source-run bench_t0_ace --train-num 320)
DIST=(--mode ce_kd --kd-type adaptive-srkl --w-span 0 --kd-ratio 0.9 --skew 0.1 --span-metric cosine --layers "22 25 28"
      --data-prefix fewrel_perm --perm 0 --start-task 1 --end-task 1 --task0-source-run bench_t0_fewrel
      --train-num 320 --dev-num 320)
for c in 0 1; do
    rm -rf results/qwen3/ced/cgen_ours_$c results/qwen3/ced/cgen_dist_$c
    s=$(date +%s); COMPILE_GEN=$c MASTER_PORT=2957$c bash scripts/qwen/ced/run_ced_v2.sh --run-name cgen_ours_$c \
        "${OURS[@]}" "${BASE[@]}" > /workspace/vllm_checks/cgen_ours_$c.log 2>&1; echo "OURS cgen=$c exit $? wall $(( $(date +%s) - s ))"
    s=$(date +%s); COMPILE_GEN=$c MASTER_PORT=2957$c bash scripts/qwen/ced/run_ced_v2.sh --run-name cgen_dist_$c \
        "${DIST[@]}" "${BASE[@]}" --extra "--log-interval 5 --student-gen --init-threshold 0.0 --loss-eps 0.1 --capacity 1000" \
        > /workspace/vllm_checks/cgen_dist_$c.log 2>&1; echo "DIST cgen=$c exit $? wall $(( $(date +%s) - s ))"
done
echo CGEN_DONE
EOF
${BOX} "setsid nohup bash /workspace/vllm_checks/cgen.sh > /workspace/vllm_checks/cgen.out 2>&1 < /dev/null & disown; echo started"
```
Expected: `started`. Wait for `CGEN_DONE` in `/workspace/vllm_checks/cgen.out`.

- [ ] **Step 3: Compare**

```bash
${BOX} 'cd /workspace/opened_dev && cat /workspace/vllm_checks/cgen.out; for r in cgen_ours_0 cgen_ours_1 cgen_dist_0 cgen_dist_1; do
    f=$(find results/qwen3/ced/$r/task1 -name log.txt | head -1); echo "== $r"; grep -E "global iter" $f | cut -c1-200 | head -8; done'
```
Expected:
- four `exit 0` lines;
- in each pair, the compiled run's `step time` is lower than the eager run's at the same
  global iterations;
- Ours' `loss` at the first two logged global iterations agrees within 1%, and later ones within
  about 7% (round 1's bf16 bound).

If a compiled run fails or is not faster, stop: the default stays 0, and README records the
result.

- [ ] **Step 4: The H200 default (tests first)**

- In `tests/test_scripts.py`, the H200 `gpu_defaults` case ends in ` 1`, and the H200
  `apply_card_defaults` cases expect `cgen=1`.
- In `tests/test_scheduler.py`, the H200 step-3 line expects `COMPILE_GEN=1`.

Run: `dev_pytest tests/test_scripts.py tests/test_scheduler.py`
Expected: those cases fail.

Then in `scripts/qwen/lib.sh`, the H200 `gpu_defaults` line's seventh value becomes `1`.
Run the same command. Expected: all pass.

- [ ] **Step 5: README results and commit**

In the H200 section, after the **Compiled sampling.** bullet, add the measured step times and
losses from Step 3:

```markdown
  Measured on 1× H200 NVL next to running jobs, physical batch 8: Ours (ACE task 1)
  (eager s) → (compiled s) per update, DistiLLM (FewRel task 1) (eager s) → (compiled s);
  logged losses agree with the eager runs within bf16 noise. On by default on H200-class cards.
```

Fill every parenthesis with the measured value.

```bash
cd ${REPO} && git add scripts/qwen/lib.sh tests/test_scripts.py tests/test_scheduler.py README.md && \
git commit -q -m "feat: compiled sampling by default on H200-class cards, after the training check

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Run: `dev_pytest tests/` on the final tree.
Expected: every test passes.
