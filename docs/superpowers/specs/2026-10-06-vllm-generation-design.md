# vLLM for evaluation and pseudo-labelling: design

Date: 2026-10-06. Branch: `perf/fast-generation`, on top of `perf/h200-throughput`.

This is sub-project 1 of the second speed-up round. Sub-project 2 (compiled generation for
self-distillation) gets its own spec. FlashAttention-2 with packing and Liger kernels were
considered and dropped: small gains, and both would change how the losses are computed.

## Goal

Generate evaluation answers and teacher pseudo-labels with vLLM instead of Hugging Face
`generate()`, while keeping every reported number statistically comparable with the runs
made so far. Target: about 30% less wall-clock for FewRel/TACRED-style baseline runs, where
answer generation is now the second-largest cost after training.

## Constraints

- **Same decoding settings.** vLLM receives exactly the settings Hugging Face would have used,
  after transformers fills in Qwen3's defaults (round 1 spec, §6): the same temperature,
  top-p, top-k, stop tokens and per-row length cap. Greedy stays greedy.
- **Same prompts and same text handling.** vLLM receives the token IDs the Hugging Face path
  would have fed `generate()`, and returns token IDs. Decoding to text, parsing and scoring
  stay in the training environment with the same code as today.
- **Same evaluated model.** The model vLLM runs computes the same function as the model the
  trainer would evaluate: base plus unmerged LoRA where the trainer uses LoRA.
- **Statistically equivalent, not bit-identical.** vLLM's kernels round differently, so a few
  greedy answers can differ. Sampled evaluations (the "greedy" evaluation of task0, the
  distillation baselines and Ours samples at T=0.5) draw different random numbers.
- **Off by default.** `--gen-backend hf` stays the code default. The H200 defaults switch
  to vLLM only after the end-to-end check in §8 passes.
- **Training environment untouched.** vLLM runs in its own Python environment, as a separate
  process. The training environment's packages do not change.
- **The FewRel run in `/workspace/opened` is not touched.** Development happens in a separate
  copy of the code on the box.

## Non-goals

- vLLM for generation inside training steps (self-distillation, DistiLLM/AMiD sampling).
  The weights change every update there; that is sub-project 2.
- GainLoRA and EPI on vLLM. Their adapters are chosen per input (gates, router). They keep
  Hugging Face generation, whatever the flag says (§4).
- A persistent vLLM server, multi-GPU vLLM, or changes to decoding settings.

## Baseline (measured 2026-10-05, round-1 code, 1× H200 NVL, 3 jobs sharing the card)

- FewRel task0, test set of 1,120 rows: answer generation and the loss pass took 63 s, about
  0.056 s per row. GPU utilization during it: about 40%.
- A FewRel distillation job generates 60,480 test answers (tasks 1 to 9), and a CL-LoRA job
  61,600 (tasks 0 to 9). At 0.056 s per row that is about 57 minutes per job, roughly 35-40%
  of the job.
- The box already has vLLM 0.27.1 (PyTorch 2.13, CUDA 13.0) in `/venv/main`, from the Vast
  vLLM template. The training environment is PyTorch 2.9.1 with transformers 4.57.3.

## Design

### 1. `tools/vllm_generate.py` (runs in the vLLM environment)

A command-line program that runs one batch of generation requests and exits.

- **Inputs.**
  - `--model DIR`: a full model directory, the base model or a merged teacher.
  - `--lora DIR` (optional), with `--max-lora-rank N`.
  - `--requests FILE`: JSON lines, one per row, in order:
    `{"prompt_token_ids": [...], "max_tokens": n, "seed": s}`.
  - `--params FILE`: JSON with `temperature` (0 = greedy), `top_p`, `top_k`,
    `repetition_penalty`, `min_p`, `stop_token_ids`, `logprobs` (bool).
  - `--out FILE`, `--gpu-memory-gb`, `--max-model-len`, `--enforce-eager`.
- **Engine.** `vllm.LLM` with `skip_tokenizer_init=True`, bf16, `generation_config="vllm"`
  (the model's own defaults are not applied; every setting comes from `--params`),
  `enable_lora` when `--lora` is given, and `gpu_memory_utilization` computed from
  `--gpu-memory-gb` and the card's total memory.
- **Outputs.** JSON lines in request order: `{"token_ids": [...]}`, plus `"logprobs": [...]`
  (the chosen token's log-probability at each position) when asked. No detokenization.
- **Exit status.** Non-zero on any failure, including fewer outputs than requests.

### 2. `gen_backend.py` (runs in the training environment)

- **`resolve_generation_config(model, generation_config, **kwargs)`.** Asks transformers for
  the settings `generate()` would actually use: `_prepare_generation_config` on the
  unwrapped Hugging Face model, with the same `use_model_defaults` that
  `gen_config.generation_kwargs(args)` passes. Pinned by a test, since the method is private.
- **`vllm_params(config)`.** Maps the resolved config to the `--params` JSON:
  - `do_sample=False` gives `temperature=0`.
  - Otherwise temperature, top-p and top-k carry over; top-k 0 or None means off.
  - `repetition_penalty` and `min_p` carry over when set.
  - `stop_token_ids` = the resolved `eos_token_id` list.
  - Any other field set away from the library default (beams, n-gram blocking, typical-p,
    epsilon/eta cutoffs, bad words, ...) raises `UnsupportedForVLLM`.
- **`run_vllm(...)`.** Writes the request and parameter files under the run directory,
  starts `tools/vllm_generate.py` with `VLLM_PY`, waits, reads the outputs, removes the
  files, and returns token IDs (and log-probabilities).
  - The child environment drops the variables torchrun sets (`RANK`, `LOCAL_RANK`,
    `WORLD_SIZE`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, `GROUP_RANK`,
    `ROLE_*`, `TORCHELASTIC_*`), so vLLM cannot join the trainer's process group.
  - It keeps `CUDA_VISIBLE_DEVICES`, the CUDA MPS variables and the offline flags.
  - On failure it raises with the last 50 lines of the vLLM log.
- **`find_vllm_python()`.** `VLLM_PY` if set; otherwise the first of `<repo>/.venv-vllm/bin/python`
  and `/venv/main/bin/python` that imports `vllm`. Returns the path and the vLLM version.
- **Before starting vLLM**, `run_vllm` calls `torch.cuda.empty_cache()`, so the cache the
  training step left behind goes back to the card.

### 3. Evaluation in the distillation/CE/Ours trainers (`ced_eval.evaluate`)

`ced_finetune.py` and `finetune.py` share this function. With `--gen-backend vllm`:

- **The loss pass is unchanged.** It stays in Hugging Face.
- **Prompts and length caps.** The generation pass walks the same loader as today, in
  batches of `--eval-batch-size`.
  - Each row's prompt token IDs are its `input_ids` where `attention_mask` is 1.
  - Each row's `max_tokens` is `max_length - width`, `width` being its batch's padded prompt
    width: exactly today's cap.
- **The model handed to vLLM.**
  - LoRA: the adapter is saved to a temporary directory under the run directory, and vLLM
    loads `--model-path` (the task's starting model) with that adapter, unmerged.
  - No LoRA: the full model is saved instead.
- **Settings.** The `GenerationConfig` built today goes through
  `resolve_generation_config` with today's `generation_kwargs(args)`, then `vllm_params`.
- **Seeds.** Each row's seed derives from (run seed, split, epoch, row index), so a rerun
  reproduces its answers.
- **Responses.** They are padded to `max_length - width` with the pad token and decoded
  with the same `batch_decode(..., skip_special_tokens=True)` call. `answers.jsonl`, the
  metrics and the `log.txt` line are written by today's code.
- **Startup checks.** The trainer refuses to start when:
  - the world size is not 1;
  - no vLLM environment is found;
  - the evaluation settings do not map (`UnsupportedForVLLM`).

  It checks right after the model loads, before the first update.

### 4. Evaluation in the CL-LoRA engine (`cl_lora/engine.py`, `eval_task`)

With `--gen-backend vllm`, what vLLM runs depends on the method:

| Method | Model handed to vLLM |
|---|---|
| `inclora`, `olora`, `inflora`, `tree` (base plus the sum of all task adapters) | One concatenated adapter: `A = [A_1; ...; A_n]`, `B = [s_1 B_1, ..., s_n B_n]`, rank `Σ r_i`, `lora_alpha = Σ r_i` (scaling 1), which computes exactly the sum. `--max-lora-rank` is the smallest vLLM-allowed rank at or above `Σ r_i`. |
| `migu` (full fine-tuning) | The full model, saved. |
| `gainlora_o`, `gainlora_inf`, `epi` | Not handed over. They keep Hugging Face generation. |

- **Settings.** Today's call is greedy with stop tokens `[eos, 151643]` and
  `max_new_tokens = max_length - max_prompt_length`. It goes through the same
  resolution and mapping as §3.
- **The manifest records the backend the method actually uses** (`gen_backend`: `vllm` or
  `hf`), so a table can say which numbers came from where.

### 5. Pseudo-labelling (`tools/ced_pseudo_label.py`)

With `--gen-backend vllm`:

- **The teacher** is already a full merged model directory, and vLLM loads it directly. The
  Hugging Face model is not loaded at all.
- **Prompts.** Each candidate row is built with today's `apply_chat_template` call and
  tokenized with today's arguments (truncation at 1,024). Without padding, a row's IDs are
  the same as its unpadded part today.
- **Settings.** Greedy, `max_tokens = --max-new-tokens`, and the stop tokens that
  `resolve_generation_config` gives for today's `generate()` call.
- **Confidence filter.** When it is on, vLLM returns each chosen token's log-probability,
  computed from the raw logits as `compute_transition_scores(normalize_logits=True)` does
  for greedy decoding. `gid` and `lp` are built in today's layout, so `event_conf_score` and
  everything after it run unchanged.
- **Batching.** All candidate rows go to vLLM in one call. The progress line is printed once.

### 6. Runners and scheduler

- **`GEN_BACKEND=hf|vllm`** (default `hf`) is read by:
  - `run_ced_v2.sh`, which passes it to the trainer (`--gen-backend`) and to
    pseudo-labelling;
  - `run_cllora.sh`, which passes it to the engine.

  `run_cre_dist.sh`, `run_cre_cllora.sh`, `run_all_cllora.sh` and `project_commands.sh`
  pass it through.
- **`VLLM_PY`** overrides the vLLM interpreter. `VLLM_GPU_GB` (default 10) caps each vLLM
  instance.
- **Fail fast.** With `GEN_BACKEND=vllm`, each runner checks once, before training, that
  `find_vllm_python()` succeeds.
- **The manifest records the backend.**
  - `run_ced_v2.sh`'s `MANIFEST_CONFIG` gets `gen=<backend>`.
  - The CL-LoRA manifest gets `gen_backend`.

  A run cannot resume under the other backend.
- **The H200 defaults** (`scripts/qwen/lib.sh` `gpu_defaults`) gain a sixth value, the
  generation backend. It stays `hf` until §8's end-to-end check passes, and then becomes
  `vllm` on H200-class cards when a vLLM environment is found (the same lookup as
  `find_vllm_python()`). Smaller cards stay on `hf`.
- **Installing vLLM** where the box has none is documented in the README:
  `uv venv .venv-vllm --python 3.12` and then `uv pip install --python .venv-vllm vllm==0.27.1`.

### 7. Memory and concurrency

- Each vLLM instance is capped at `VLLM_GPU_GB` (default 10 GB). Qwen3-0.6B needs about
  1.2 GB of weights, which leaves room for about 50K tokens of KV cache.
- vLLM runs as a CUDA MPS client like the trainers when the scheduler started MPS.
- The scheduler's memory guards (`NEED_GPU_MB`) are re-measured with vLLM evaluation on
  (§8), and `gpu_defaults` is updated if the per-job peak grows.
- **Engine mode.** Whether vLLM uses CUDA graphs (`--enforce-eager` off) or eager mode is
  decided by measurement: start-up time against generation speed on the FewRel task-9 test
  set (11,200 rows). Concurrent instances sharing vLLM's compile cache are part of that
  measurement.

### 8. Verification

**CPU tests (no vLLM needed; a fake `VLLM_PY` script stands in):**

- `resolve_generation_config` on a tiny Qwen3-shaped model with Qwen3's
  `generation_config.json`:
  - today's evaluation config resolves to sampling at T=0.5 / top-p 0.95 with the
    library-default fields filled in;
  - `--strict-generation` resolves to greedy;
  - the CL-LoRA and pseudo-label calls resolve to greedy.
- `vllm_params`:
  - greedy maps to temperature 0;
  - top-k 0 maps to off;
  - beams or n-gram blocking raise `UnsupportedForVLLM`.
- **Prompts and caps from a left-padded batch:** the unpadded IDs, and
  `max_tokens = max_length - width`.
- **Round trip through the fake script:**
  - the same token IDs give the same decoded text, `answers.jsonl` and `log.txt` line as the
    Hugging Face path;
  - a short output file raises;
  - the torchrun variables are absent from the child environment.
- **Concatenated CL-LoRA adapter:** on a tiny model, its logits equal the consolidated
  multi-adapter forward (fp32, `atol=1e-5`), for 1, 2 and 3 task adapters.
- **Pseudo-labelling from vLLM-shaped outputs** (IDs and log-probabilities) gives the same
  rows, scores and `pl_stats.json` as the same tokens in Hugging Face's layout.
- **Startup refusals:** world size above 1, no vLLM environment, unsupported settings.

**GPU checks on the box** (a separate code copy, next to the FewRel run):

1. **Smoke test.** `vllm_generate.py` on Qwen3-0.6B, with a full model and with a LoRA
   adapter. Greedy token IDs agree with Hugging Face greedy for at least 95% of 64 prompts.
2. **Greedy parity on CL-LoRA.** Round 1's IncLoRA TACRED order-0 checkpoint, task-9 test
   set (1,240 rows), Hugging Face against vLLM:
   - at least 99% identical answers;
   - trigger F1 within 0.3 points.
3. **Sampled parity on distillation.** One RKL checkpoint's test set, 3 seeds per backend:
   the mean trigger F1 differs by no more than the larger of the two seed spreads, or by
   1 point at most.
4. **Pseudo-label parity.** One ACE task's pseudo-labelling, with the confidence filter on:
   - at least 99% identical pseudo-label rows;
   - event scores within 0.01.
5. **Speed.** FewRel task-9 test set (11,200 rows), Hugging Face against vLLM, and the
   engine-mode measurement from §7.
6. **End to end.** TACRED order 0, RKL and IncLoRA, with `GEN_BACKEND=vllm`:
   - final-task trigger F1 within 2 points of round 1's (RKL 65.56, IncLoRA 63.06);
   - wall-clock recorded against round 1's (3,889 s and 4,213 s). It is shared-card time,
     so the comparison is noted as approximate.

   If F1 is off by more than 2 points, stop and investigate before any default changes.

### 9. Rollout

1. Land the code with `hf` as the default everywhere.
2. Run §8. Write the measured results into the README's H200 section.
3. If check 6 passes, switch the H200 defaults to `vllm` (§6).
4. Runs started before the switch, such as the FewRel run in progress, keep their backend.
   Their manifests refuse a resume under the other one.

### 10. Risks

- **Greedy answers differ on a few rows** because of kernel rounding. Measured by check 2.
  CL-LoRA tables mix vLLM and Hugging Face numbers (GainLoRA, EPI); the manifest records
  which.
- **Sampled evaluations draw different random numbers.** The scores move within the same
  noise as a different seed today. Measured by check 3.
- **vLLM version drift** could change sampling details. 0.27.1 is the tested version, and
  `find_vllm_python()` prints the version into the run log.
- **Memory.** Three trainers plus three vLLM instances on one card. Each instance is capped;
  the guards are re-measured.
- **Start-up cost.** Each evaluation pays vLLM start-up (estimated 20-40 s). That is small
  against the 60-600 s it replaces at the end of a task. Under `--eval-gen-mode every` it
  is paid twice per epoch.
- **Hosts without the Vast vLLM template** need the documented `.venv-vllm` install.
