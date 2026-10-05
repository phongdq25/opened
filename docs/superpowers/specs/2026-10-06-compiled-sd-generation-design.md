# Compiled sampling for self-distillation and DistiLLM: design

Date: 2026-10-06. Branch: `perf/fast-generation`. Sub-project 2 of the second speed-up round.
Sub-project 1 (vLLM for evaluation and pseudo-labelling) is
`docs/superpowers/specs/2026-10-06-vllm-generation-design.md`.

## Goal

Make the sampling that happens inside training steps faster: the self-distillation sample in
Ours (`ced_losses.sd_prepare`) and the student generation of DistiLLM/AMiD
(`distillm/sampler.py`, `SampleGenerator.run_sample`). Ours spends most of each update there
(8 to 20 s per update against under 1 s for RKL, round 1's benchmark).

vLLM does not fit here: the weights change at every update. Hugging Face `generate()` with a
static KV cache does. With a static cache, transformers 4.57.3 compiles the decoding forward
with `torch.compile` (mode `reduce-overhead`, CUDA graphs), which removes most of the per-token
launch overhead of a 0.6B model.

## Evidence (spike, 2026-10-05, shared H200, Qwen3-0.6B + a FewRel LoRA, batch 8, 64 prompts)

| | eager | static cache, compiled |
|---|---|---|
| greedy, 64 prompts | 36.2 s | 13.4 s (first pass 68.4 s, compiling) |
| sampled T=0.6/top-p 0.95, same seed | 31.4 s | 12.2 s |
| answers identical to eager | | 64/64 greedy, 64/64 sampled |

## Constraints

- **Same sampling.** Only the cache layout and the compiled forward change. The settings, the
  logits processors and the random stream stay as they are. Answers may differ in rare
  near-ties from kernel rounding (the spike saw none in 128).
- **Same objectives.** Nothing else in the step changes.
- **Off by default.** `--compile-generation` (runner `COMPILE_GEN=1`). The H200 defaults turn
  it on only after the training check in Verification passes.
- **Evaluation is not affected.** Evaluation generation stays with vLLM or Hugging Face as in
  sub-project 1.

## Design

- **`gen_config.train_generation_kwargs(args)`.** It returns `generation_kwargs(args)`, plus
  `cache_implementation="static"` when `args.compile_generation` is set. `generate()` reads it
  as a generation setting and then compiles the decoding step itself.
- **The two in-training generators use it.**
  - `ced_losses.sd_prepare` (self-distillation);
  - `SampleGenerator.run_sample` (DistiLLM/AMiD).

  Evaluation keeps `generation_kwargs(args)`: its last batch has another size every time, and
  each new batch size costs a compile.
- **Flag and runner.**
  - `--compile-generation` goes in `arguments.py`.
  - `run_ced_v2.sh` gets the knob `COMPILE_GEN` (default 0), passes the flag, and adds
    `cgen=` to `MANIFEST_CONFIG` so that a run cannot resume under the other setting.
  - `project_commands.sh` prints it on the step-3 line.
- **H200 defaults.** `gpu_defaults` gains a seventh value, `COMPILE_GEN`: 0 until Verification
  passes, then 1 on H200-class cards.
- **Recompiles.** One compile per distinct generation batch size: the full physical batch, and
  the short last batch of an epoch when there is one. Each compile takes about a minute. Ours
  makes thousands of sampling calls per run, so this is accepted.

## Verification

- **CPU tests:**
  - `train_generation_kwargs` adds the static cache only with the flag, and keeps the strict
    switch;
  - `sd_prepare` and `run_sample` pass it to `generate()`;
  - `evaluate()` does not.
- **GPU, Ours (ACE task 1, `tools/bench_gpu.sh`'s Ours flags, physical batch 8,
  `COMPILE_GEN` 0 then 1):**
  - the compiled run's step time is lower;
  - its logged losses at the first logged steps agree with the eager run's within bf16 noise:
    the first two logged values within 1%, later ones within round 1's bound of about 7%.
- **GPU, DistiLLM (FewRel task 1, 320 rows, `COMPILE_GEN` 0 then 1):** the compiled run's step
  time is lower and both runs finish.

  If either compiled run fails or is not faster, the default stays 0 and the result goes in the
  README.

## Risks

- **Compile time and recompiles:** about a minute each, per distinct batch size.
- **Memory:**
  - the static cache is batch × 768 tokens × 112 KiB (0.7 GB at batch 8);
  - CUDA graphs keep their own pools.
- **DeepSpeed, PEFT or MPS interplay** the spike did not cover. The GPU checks run through the
  real runner, with DeepSpeed and PEFT.
