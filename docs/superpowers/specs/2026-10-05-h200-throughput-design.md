# H200 throughput optimization: design

Date: 2026-10-05. Branch: `perf/h200-throughput`.

## Goal

Run the existing experiment matrix much faster on 1 to 4 H200s, while keeping every
result statistically comparable with the runs the authors already have. The matrix
covers:

- the 7 distillation baselines and the 8 CL-LoRA baselines;
- "Ours" and its ablations;
- CRE (TACRED, FewRel) and CED (ACE, MAVEN, RAMS, GENEVA).

The target is a FewRel baseline sweep in about 1 day on 4×H200, down from about 12.

## Constraints

- **Same objective for every method.** The per-micro-batch CE/KD weighting
  (`ced_finetune.py:1097-1100`) stays as it is, including its dependence on the
  micro-batch size (see §1).
- **Same everything else.** Data, hyperparameters, effective batch (32), learning-rate
  schedules and metrics are unchanged.
- **Same effective decoding as today** (§6), unless `--strict-generation` is given.
- **Statistically equivalent, not bit-identical.** Different batch shapes change
  floating-point rounding. Fewer generation calls change the random-number stream,
  because evaluation samples today (§6).
- **Old behaviour stays reachable.** Every flag defaults to today's behaviour. The runner
  scripts opt into the new settings.

## Non-goals

- Fixing the CE/KD gating, or making generation strict by default. Both change results.
  Strict generation is available as an opt-in flag.
- vLLM or other inference engines.
- Changing the experiment matrix, or which host runs what.

## Baseline (measured 2026-10-05, current code, 1× H200 NVL)

| Workload | Speed | GPU util | Peak memory |
|---|---|---|---|
| Distillation training (RKL), micro-batch 2×16 | 2.57 s per 32 rows | 33% | 31 GiB |
| Same at micro-batch 16×2 (different objective) | 0.86 s per 32 rows | 43% | 104 GiB |
| Evaluation generation (FewRel), batch 32 | about 3.0 s per batch | not measured | not measured |
| Ours with self-distillation (ACE), micro-batch 2×16 | 45.2 s per 32 rows | 24% | 32 GiB |
| CL-LoRA (IncLoRA), 2 tasks of 640 rows each, with evaluation | 518 s | 26% | 6 GiB |
| 4 distillation runs sharing one GPU | each run 1.78× slower | 42% | 123 GiB |

The roughly 30 GiB peaks at micro-batch 2 come from evaluation's full-vocabulary loss pass.

Token lengths (train, task 1):

- median sequence length is 284 to 355 tokens, and the maximum is 376 to 624;
- 53 to 63% of each 768-token row is padding;
- label tokens are 5 to 8% of positions.

Extrapolations:

- **One FewRel distillation run (one method, one task order): about 19 h.** About 16 h of
  that is generating answers on the full cumulative dev and test sets after every epoch,
  roughly 605K generations per run.
- **The full FewRel baseline sweep: about 1,100 GPU-hours.**
- **Ours: about 15 h per task order on ACE, and about 5 days on MAVEN.**

## Design

### 1. Logical micro-batch vs. physical batch

**Today.** `--batch-size` sets both the GPU batch and the unit the loss is normalized and
gated over.

**New flag.** `--loss-group-size g` is the logical micro-batch. Its default is
`--batch-size`, which reproduces today's behaviour exactly.

**How a step works.** The runner sets a physical batch `P = k·g` and an accumulation of
`G' = G/k`. `P` must be a multiple of `g`, and `G` must be a multiple of `k`; the trainer
refuses anything else. Each step then:

1. runs one student forward, one teacher forward and one self-distillation sampling call,
   all over `P` rows;
2. computes the loss for each group of `g` consecutive rows exactly as today's code
   computes it for one micro-batch;
3. averages over the groups.

DeepSpeed accumulates `G'` steps per update.

**Why the objective is unchanged.** Old update: `(1/G)·Σ_i ∇L_i`. New update:
`(1/G')·Σ_s (1/k)·Σ_j ∇L_sj`. These are equal because `G'·k = G`.

The sampler yields the same permutation, so each group is one of today's micro-batches.
The one difference: `drop_last` now drops up to `P−1` rows per epoch instead of `g−1`.

**Logical micro-batch per runner.** Each runner keeps its current `g`:

| Runner | `g` |
|---|---|
| task0, Ours, CRE distillation | 2 |
| CED distillation (`dist_queue.sh`) | 32 (KD, CSD) or 16 (the rest) |
| CL-LoRA | 2 (CRE) or 32 (CED) |

`P` and `G'` are new runner settings, `PHYS_BS` and the accumulation derived from it.
The runner scripts default to `P = g`, which is safe on the authors' 46 GB cards. The two
schedulers, `project_commands.sh` and `run.sh`, set the H200 default measured in §9.

### 2. Training step (`ced_step.py`, used by `ced_finetune.py` and `finetune.py`)

The new module `ced_step.py` imports nothing from DeepSpeed, so it can be unit-tested on
its own. It contains:

- `micro_batch_loss(...)`: today's loss block (`ced_finetune.py:995-1117`) for one
  logical micro-batch. It covers CE, every KD type, span loss, KD scope `pl`, LwF,
  `kd_only` replay mode and the self-distillation term.
- `grouped_step_loss(...)`: runs the forwards once for the physical batch, then calls
  `micro_batch_loss` once per group.

Changes inside the step that leave the result unchanged:

1. **Dynamic padding** (`--dynamic-pad`: off in the code, on in the runners). The collate
   pads to the longest row in the batch, rounded up to a multiple of 64, instead of to
   768. It is forced off in two cases:
   - with `--student-gen` (DistiLLM, AMiD), whose replay buffer stores 768-wide rows;
   - whenever the span loss is on (`--w-span-loss != 0`, i.e. Ours and its ablations).
     `compute_token_weights` averages attention over every query position, padded ones
     included, so padding length changes the span loss.
2. **Label-window logits.** The student forward passes `logits_to_keep` set to the
   position range that holds the batch's labels. This is supported by the pinned
   transformers, `modeling_qwen3.py:493`. CE and KD gather their positions from that
   window. Positions outside the window carry no label, so the loss values are the same.
   This is always on, because it is exact.
3. **Teacher forward only where needed.** It runs on memory rows, on pseudo-labelled rows
   when the KD scope is `pl`, and on LwF rows, with the same window.
4. **Span loss per group.** It is computed on each group's KD rows only. Rows without
   spans add nothing today either.
5. **Self-distillation.**
   - One `generate` call for all `P` rows, then one EMA-teacher scoring forward.
   - The SD loss is computed per group over that group's kept rows. A group with no kept
     rows adds no SD term, as today.
   - Weights only change at update boundaries, so the samples have the same distribution
     as today's per-micro-batch calls.
6. **DistiLLM/AMiD student generation.** The random draws and replay-buffer decisions
   happen per group, in group order, as today. The rows that need generating are
   generated in one batched call.
7. **Initial dev loss.** `evaluate_loss`, which the adaptive methods use, runs over
   batches of `g` rows, as today.

`finetune.py` (task0 and `mode sft`) gets CE grouping, dynamic padding and the label
window. With a teacher model or `--student-gen` it keeps physical = logical, but the
runners never use it that way.

### 3. CL-LoRA engine (`cl_lora/engine.py`)

- **`--loss-group-size`.** Per-group token-mean CE, averaged over groups. The O-LoRA
  orthogonality term is unchanged.
- **TreeLoRA.** The bandit step, signature and regularizer run once per group, in group
  order, so the random draws and the schedule match today.
- **MIGU.** Grouping is disabled, so physical = logical. Its magnitude statistic depends
  on per-step padding.
- **GainLoRA and EPI.** Gate-gradient projection and EPI routing are unchanged. Both are
  linear or per-row.
- **Runners.** They raise `--eval-batch-size` from 16 to 128.

### 4. Evaluation (`ced_finetune.py`, `finetune.py`)

**`--eval-gen-mode final|every`.** The code default is `every`, which is today's behaviour.
The runners pass `final`. In `final` mode:

- answers are generated only on the test set, after the last update of each task;
- per-epoch dev passes compute only the loss, and only for DistiLLM/AMiD, which read it;
- task0's pre-training dev evaluation is skipped unless the method is adaptive;
- `--select-best-dev 1` forces `every`.

**Batches.**

- The generation batch is `--eval-batch-size`; the runners use 128.
- The loss pass runs in chunks of `--eval-loss-batch-size`. Its default is 32, today's
  evaluation batch, so the logged dev loss and the adaptive threshold see the same
  values.
- The loss pass covers answer positions only, in today's dtype.

**Decoding.** The reported test numbers keep today's decoding settings (§6).

### 5. Scheduling (`project_commands.sh`, `run.sh`, runner scripts)

**GPU slots.**

- The GPU list is detected with `nvidia-smi`, and `POOL_GPUS` overrides it. This replaces
  the hardcoded `4 5 6 7`.
- `SLOTS_PER_GPU` repeats each GPU in the slot list. Each slot gets its own torchrun port,
  `29500 + slot`.

**CRE baselines become per-method jobs.** Per task order:

1. one task0 job trains the shared CE teacher;
2. one job per distillation method then waits for it;
3. one job per CL-LoRA method runs independently.

**Runner fixes.**

- `run_cre_dist.sh` gets a `T0_ONLY=1` mode, which trains the shared task0 and exits.
- `run_cre_dist.sh` gets `KEEP_T0=1`, which skips deleting the shared task0 model
  (`:139`) while sibling jobs still use it.
- A method job requires the shared task0 to exist, and never recreates it.
- `run_cre_cllora.sh:64` checks for the same run (method plus data root), not for any run
  of that method.

**`run.sh` (CED baselines).**

- GPU ids may repeat in `GPU_DIST_ALL` and `GPU_CLLORA_ALL`. Each entry gets one
  sub-queue, and methods are split round-robin as today.
- Each sub-queue gets its own `MASTER_PORT`.

**Memory guards** (`NEED_GPU_MB` and similar) are set from measured peaks.

**CUDA MPS** is optional (`USE_MPS=1`). The pool starts the MPS daemon if the container
allows it, and keeps it only if §9 shows it beats time-slicing.

### 6. Generation settings

**What happens today.** When transformers 4.57.3 receives a `GenerationConfig`, it
replaces every field left at the library default with the model's own default. For
Qwen3-0.6B that is `do_sample=True, temperature=0.6, top_p=0.95, top_k=20` (see
`generation/utils.py`, around line 1788). Resolved settings today:

| Call | Intended | Actual |
|---|---|---|
| `evaluate()`: reported F1 for task0, the distillation baselines, Ours and ablations | greedy | sampling, T 0.5, top_p 0.95 |
| `sd_prepare()`: Ours self-distillation | T 1.0, top_p 1.0 | T 0.6, top_p 0.95 |
| `SampleGenerator`: DistiLLM/AMiD | T 1.0, top_p 1.0 | T 0.6, top_p 0.95 |
| CL-LoRA evaluation, pseudo-labelling | greedy | greedy |

**Default: unchanged.** The configs are built exactly as today, so the resolved settings
are the same.

**`--strict-generation`** (runner setting `STRICT_GEN=1`) sets every field explicitly
from the arguments:

- `evaluate()` becomes greedy;
- SD and DistiLLM/AMiD sample at the temperature and top_p they ask for.

This changes results, so turning it on is the authors' decision.

### 7. Robustness fix

`--train-num` and `--dev-num` larger than the split are clamped to the split size.
Today `DistributedMMapIndexedDataset.__getitem__` loops forever in that case
(`distributed_indexed.py:201`).

### 8. Settings

| Setting | Where | Default | Effect |
|---|---|---|---|
| `--loss-group-size` | ced_finetune, finetune, engine | `--batch-size` | logical micro-batch |
| `PHYS_BS` | runner scripts (read), schedulers (set) | `g` in the runners; §9 value in the schedulers | physical batch |
| `--eval-gen-mode` | ced_finetune, finetune | `every` (runners: `final`) | when answers are generated |
| `--eval-batch-size` | all trainers | 32, engine 16 (runners: 128) | generation batch |
| `--eval-loss-batch-size` | ced_finetune, finetune | 32 | loss-pass chunk |
| `--dynamic-pad` | ced_finetune, finetune | off (runners: on) | pad to the batch's longest row |
| `--strict-generation` / `STRICT_GEN` | ced_finetune, finetune | off | literal generation settings |
| `POOL_GPUS`, `SLOTS_PER_GPU`, `USE_MPS` | project_commands, run.sh | detected / from §9 / off | packing |

### 9. Verification

1. **Equivalence tests** (`tests/`, pytest, run on the H200 in fp32).
   - **Setup:** a tiny randomly initialized Qwen3 (2 layers, real vocabulary) with LoRA
     and dropout 0.
   - **Reference:** today's loss code, copied verbatim from `main` into
     `tests/reference_ced_loss.py`, run on each group as its own micro-batch.
   - **New path:** the physical batch.
   - **Pass criterion:** losses and LoRA gradients agree within 1e-5 relative.
   - **Cases for `ced_finetune.py`:**
     - the KD types kd, rkl, sfkl, srkl, csd, amid (ab) and no;
     - span loss on and off (cosine, cka);
     - KD scope replay and pl;
     - LwF;
     - `kd_only`;
     - SD with fixed injected samples, including the omission mask;
     - groups with no memory rows or no kept SD rows;
     - DistiLLM with a fixed RNG.
   - **Cases for `cl_lora/engine.py`:** CE grouping, O-LoRA, and TreeLoRA with a fixed
     RNG.
2. **Benchmark re-run.** The same configs as the baseline, at `P ∈ {8, 16, 32}`, measured
   alone, with 2 to 4 runs sharing a GPU, and with MPS. The results set `PHYS_BS`,
   `SLOTS_PER_GPU` and the memory guards.
3. **Old-vs-new curves.** With LoRA dropout 0, the task1 loss curves of the old code
   (micro-batch 2) and the new code (`P=16`, `g=2`) overlap within bf16 noise over one
   epoch.
4. **End to end.** One full TACRED task order, RKL and IncLoRA, old code vs new. Final F1
   agrees within run-to-run noise, and the run records the real speedup.

### 10. Rollout

Stages, each one committed and tested on the branch:

1. evaluation mode, the strict-generation flag and the robustness fix;
2. `ced_step.py`, the training-step refactor and the equivalence tests;
3. the CL-LoRA engine;
4. scheduler and runner changes;
5. benchmark, defaults, and a README section.

Nothing is pushed. The branch is handed back to the user.

### 11. Risks

- **Equivalence subtleties.** Span-loss normalization, SD statistics and DistiLLM buffer
  order are covered by tests against the verbatim reference.
- **Memory with 32-row logical groups.** The CED distillation sweep's 32-row groups need
  `P ≥ 32`. If that doesn't fit beside other runs, those runs take one GPU slot each.
- **Resuming old runs.** Run manifests fingerprint the runtime files, so a run started
  with the old code can't be resumed with the new code. Finish it or restart it.
