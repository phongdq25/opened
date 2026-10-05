# OpenED — Continual Event Detection (CED) & Continual Relation Extraction (CRE)

Qwen3-0.6B trained continually across tasks (LoRA, teacher = previous task's checkpoint).
Two baseline families — distillation (KD/RKL/SFKL/SRKL/CSD/DistiLLM/AMiD) and CL-LoRA
(IncLoRA/O-LoRA/TreeLoRA/InfLoRA/EPI/MIGU/GainLoRA×2) — across two dataset families:

- **CED** (event detection, 5 tasks/permutation): ACE, MAVEN, RAMS, GENEVA
- **CRE** (relation extraction, 10 tasks/permutation): TACRED, FewRel

All datasets use 5 permutations (SharpSeq stream orders).

## Setup

```bash
conda create -n mta python=3.12 && conda activate mta
pip install -r requirements.txt
export CUDA_HOME=$HOME/miniconda3/envs/mta   # needed for deepspeed to detect nvcc
```

`opened.txt` is the same pin list for hosts that install with uv. It no longer carries the
`en_core_web_sm @ https://github.com/...whl` line: those hosts cannot reach GitHub, so the
spaCy model is fetched as a zip by `download.txt` instead and unpacked afterwards.

```bash
cd /mnt/local/$PROJECT/OpenED                # where download.txt drops the zip
python -c "import zipfile; zipfile.ZipFile('en_core_web_sm.zip').extractall('.')"
rm -rf __MACOSX en_core_web_sm.zip           # zip was packed on a Mac
```

Python's `zipfile` rather than `unzip`: the image has no unzip binary and no sudo. It unpacks
to `./en_core_web_sm/` (a model directory: `config.cfg` + `meta.json`, v3.8.0, which is what
the pinned `spacy==3.8.14` wants), not an installed package — so `spacy.load("en_core_web_sm")`
resolves it as a path and only works with the repo root as the working directory. Elsewhere,
pass the full path. Nothing in this repo imports spaCy today, so if the pins are ever cleaned
up, this model and the three `spacy*` lines in `opened.txt` can go with them.

## Data

`data/` holds the permutation splits. Every corpus is also published one dataset repo per
corpus — [cl-ace](https://huggingface.co/datasets/datht/cl-ace),
[cl-maven](https://huggingface.co/datasets/datht/cl-maven),
[cl-rams](https://huggingface.co/datasets/datht/cl-rams),
[cl-geneva](https://huggingface.co/datasets/datht/cl-geneva),
[cl-tacred](https://huggingface.co/datasets/datht/cl-tacred),
[cl-fewrel](https://huggingface.co/datasets/datht/cl-fewrel) — in a uniform shape
(`raw/`, `perm<k>/`, `groups/`). `download.txt` fetches them file by file straight into the
directory names the runners read (`data/<ds>`, `data/<ds>_b10_perm<k>` for CED,
`data/<ds>_perm<k>` and `data/<ds>_groups` for CRE), so nothing needs renaming afterwards.
Re-publish with `python tools/push_hf_datasets.py --dry-run` (drop the flag to push).

## Run

```bash
bash run.sh                                                      # everything, see below
bash run.sh <tacred|fewrel|maven|rams|geneva> ["perms"] [gpu_dist] [gpu_cllora] [queue]
```

`bash run.sh` with **no arguments** runs **every baseline run still missing**, one dataset at
a time (dist + CL-LoRA concurrently within a dataset, next dataset only starts once both
queues of the current one finish, so no GPU ever gets two queues). The plan lives in
`MISSING_PLAN` at the top of `run.sh` as `<dataset>:<perms>:<queue>` entries:

| Dataset | Perms | Queue | Why |
|---|---|---|---|
| maven | 0-4 | both | perm0-1 done, perm2 partial, perm3-4 never ran — the finished parts skip themselves |
| rams | 0-4 | both | never run |
| geneva | 0-4 | both | never run |

Both CRE datasets are out of the plan — they are already queued on the old host, so this is
a CED-only sweep. Put one back if its queue is lost:

```bash
MISSING_PLAN="tacred:4:cllora"        bash run.sh   # cllora only: the dist queue would
                                                    # retrain five task0 teachers for nothing
MISSING_PLAN="fewrel:0 1 2 3 4:both"  bash run.sh   # nothing usable yet, full sweep
```

Only missing work trains: every runner checks its own completion marker per method+perm, so
listing a full perm range is a no-op for whatever already finished. Progress:
`tail -f logs_run_all.log` (plus the per-dataset logs listed below once a dataset starts).
Override with `MISSING_PLAN="ds:perms:queue;..."`, or `RUN_ALL_DATASETS="ds1 ds2 ..."` for a
plain list at all perms and both queues.

### Running on a new server

`data/` is in git, so the perm splits arrive with the clone and only tokenization runs per
host. `results/` is **not** — none of the completion markers come along, so nothing is
skipped and a plain `bash run.sh` retrains all 15 baselines on all five datasets (~3500
single-task trainings), finished work included. Pick one first:

```bash
# either: bring the finished runs over, then run normally
rsync -a --include='*/' --include='.complete' --include='cl_results.json' \
      --include='log.txt' --exclude='*' OLDHOST:OpenED/results/ results/
# or: give this host only part of the work
MISSING_PLAN="fewrel:0 1 2 3 4:both" bash run.sh
```

Host-specific knobs, all optional, forwarded to the queues only when set:

| Var | Default | Set it when |
|---|---|---|
| `GPU_DIST_ALL` / `GPU_CLLORA_ALL` | 0 / 1 | the two free GPUs are not 0 and 1 |
| `PY` | `envs/nuquant/bin/python` (CED), `envs/mta/...` (CRE) | conda env is named differently |
| `ENV_BIN` | `$HOME/miniconda3/envs/mta/bin` | same, for the KD runners |
| `DISK_PATH` | `.` | the free-space guard should watch another filesystem |
| `HF_HUB_OFFLINE` | 1 for CED CL-LoRA | the HF cache does not already hold Qwen3-0.6B (set 0) |

**Not covered by either queue** (no runner script exists): LwF (ACE perm3-4), SeqLoRA-merge
(ACE perm1-4), and f12_pl / ours (MAVEN perm1-4, TACRED, FewRel). LwF is a `run_ced_v2.sh`
flag (`--kd-new`) and ours is `run_ced_v2.sh --mode ce_kd --pl 1`, but their exact configs
are not checked in anywhere, so launch those by hand.

One call runs one dataset end to end: tokenizes the requested permutations, then launches
the distillation queue (7 methods) on `gpu_dist` and the CL-LoRA queue (8 methods) on
`gpu_cllora`, each working through its permutations in the background. No other setup step
is needed. Tokenizing means running `tools/process_data.py` per task before training —
`run_ced_v2.sh` does NOT do this itself (it only tokenizes PL/balance side-data, never the
base task data), so `run.sh` does it explicitly for every task, same as CRE's `prep_cre.sh`;
already-tokenized tasks are skipped on a re-run.

```bash
bash run.sh rams                   # perms 0-4, distillation on gpu0, CL-LoRA on gpu1
bash run.sh geneva "0 1 2"         # only perm0-2
bash run.sh maven "0" 0 1          # single perm, explicit GPU assignment
bash run.sh tacred                 # CRE dataset, same interface
bash run.sh rams "3 4" 0 1 cllora  # one queue only (dist|cllora|both, default both)
bash run.sh geneva "" 0 1 prep     # tokenize all perms and stop, train nothing

tail -f logs_ced_dist_rams.log logs_ced_cllora_rams.log
```

CED datasets need their perm split built once (raw sentences sliced into tasks/streams,
separate from tokenization above). **ACE, MAVEN, RAMS and GENEVA are already built** —
`data/<ds>_b10_perm{0..4}/{0..4}/{train,dev,test}.jsonl` plus `streams.json`. To rebuild, or
for a new dataset:

```bash
python tools/build_maven_perms.py --src data/rams --out-prefix rams_b10_perm --cap 10
python tools/build_maven_perms.py --src data/geneva --out-prefix geneva_b10_perm --cap 10
```

(the script is dataset-agnostic despite the filename — it only assumes the
`{system_prompt, user_prompt, response}` record schema shared by ACE/MAVEN/RAMS/GENEVA)

Notes:
- CED run names carry no dataset of their own (`cllora_olora_perm0_v2_s42`), so `run.sh`
  folds the dataset into the `PROTOCOL` tag for every CED dataset except ACE and MAVEN:
  RAMS perm0 O-LoRA is `cllora_olora_perm0_rams_v2_s42`. Without that, RAMS/GENEVA at a
  given perm would collide with the ACE/MAVEN run of the same perm on the same box. ACE and
  MAVEN keep the bare `v2` so their existing run dirs stay recognized.
- `RESUME=1 bash run.sh ...` forwards `--resume` to both CED queues, so a perm that died
  mid-task continues from its manifest instead of aborting with "partial run exists".
- Never run two queues on the same GPU. The memory/disk guards in the per-family runners
  (`scripts/qwen/cre/run_cre_*.sh` for CRE, `scripts/qwen/ced/{dist_queue,run_all_cllora}.sh`
  for CED) are snapshots, not reservations, and two queues racing for one card's memory
  between tasks will OOM each other.
- Safe to re-run: every step (tokenize/prep, both queue scripts) checks for its own
  completion marker before starting and skips finished work, so a re-run after any
  interruption just resumes.

## Running on H200s (branch `perf/h200-throughput`)

Same experiments, same objectives, much less wall-clock. What changed and how to control it:

- **Physical vs logical batch.** Each runner keeps its micro-batch (`--bs`) as the unit the
  loss is defined over. `PHYS_BS` (environment) sets how many rows go through the GPU at once.
  The loss is still computed per `--bs` rows and averaged, and the accumulation shrinks so
  rows per update stay 32. Runner scripts default to `PHYS_BS = --bs`. `project_commands.sh`
  and `run.sh` set the H200 value below.
- **Evaluation.** `run_ced_v2.sh` now generates answers once per task, for the test set after
  the last update (`EVAL_GEN_MODE=final`; set `every` for the old per-epoch dev+test
  evaluation, which `--select-best-dev 1` forces). Generation runs in batches of 128
  (`EVAL_BS`).
- **Padding.** Batches are padded to their longest row (`DYNAMIC_PAD=1`), except for runs
  with a span loss (Ours and its ablations) or DistiLLM/AMiD.
- **Packing.** `project_commands.sh` uses every GPU (`POOL_GPUS` to restrict) with
  `SLOTS_PER_GPU` runs per card, a torchrun port per slot, and, on H200, optional CUDA MPS
  (`USE_MPS`). CRE baselines run as one job per method after each order's shared task0.
  `run.sh` accepts repeated GPU ids (`GPU_DIST_ALL=0,0,1,1`).
- **Decoding (read this).** transformers 4.57 fills `GenerationConfig` fields left at their
  library default from Qwen3's `generation_config.json`. As a result:
  - the `--greedy 1` evaluation of task0, the distillation baselines and Ours **samples at
    T=0.5, top_p 0.95**;
  - SD and DistiLLM/AMiD sample at **T=0.6, top_p 0.95** instead of the requested 1.0/1.0;
  - CL-LoRA evaluation is really greedy.

  This branch keeps that behaviour so new numbers match existing ones. `STRICT_GEN=1`
  (`--strict-generation`) makes every config literal: greedy evaluation and the requested
  temperatures. That changes results, so decide it for the whole table.
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
- **Compiled sampling.** `COMPILE_GEN=1` (`--compile-generation`) samples inside training
  steps (self-distillation in Ours, DistiLLM/AMiD student generation) with a static KV cache,
  which transformers compiles. The settings and the random stream are unchanged.

Measured on 1× H200 NVL (`bash tools/bench_gpu.sh h200`):

| phase | phys | runs | mps | wall_s | s_per_update | peak_gib | mean_util |
|---|---|---|---|---|---|---|---|
| rkl | 2 | 1 | 0 | 159 | 2.67 | 13.7 | 21 |
| ours | 2 | 1 | 0 | 704 | 49.42 | 31.9 | 24 |
| cl | 2 | 1 | 0 | 284 | - | 15.8 | 40 |
| rkl | 8 | 1 | 0 | 77 | 0.72 | 13.6 | 28 |
| ours | 8 | 1 | 0 | 366 | 20.39 | 36.9 | 29 |
| cl | 8 | 1 | 0 | 136 | - | 16.0 | 63 |
| rkl | 16 | 1 | 0 | 83 | 0.41 | 21.1 | 35 |
| ours | 16 | 1 | 0 | 290 | 13.66 | 67.5 | 34 |
| cl | 16 | 1 | 0 | 142 | - | 24.1 | 69 |
| rkl | 32 | 1 | 0 | 63 | 0.32 | 36.1 | 31 |
| ours | 32 | 1 | 0 | 239 | 8.24 | 104.1 | 36 |
| cl | 32 | 1 | 0 | 150 | - | 46.2 | 69 |
| share | 16 | 1 | 0 | 84 | 0.46 | 21.1 | 35 |
| share | 16 | 2 | 0 | 107 | 0.68 | 42.1 | 55 |
| share | 16 | 3 | 0 | 142 | 0.96 | 63.2 | 67 |
| share | 16 | 4 | 0 | 268 | 1.32 | 84.3 | 48 |
| share | 16 | 1 | 1 | 83 | 0.41 | 21.1 | 36 |
| share | 16 | 2 | 1 | 110 | 0.52 | 42.2 | 34 |
| share | 16 | 3 | 1 | 112 | 0.68 | 63.3 | 58 |
| share | 16 | 4 | 1 | 288 | 0.96 | 84.3 | 26 |

The defaults come from these rows (the rule at the end of `tools/bench_gpu.sh`):
`PHYS_BS=8`, because Ours peaks at 67.5 GiB at 16 and two Ours runs would not fit on one
card; `SLOTS_PER_GPU=3` with `USE_MPS=1`, which finished the most runs per hour (three
RKL runs in 112 s, one alone in 83 s). Matrices without Ours runs (the FewRel/TACRED
baselines) can set `PHYS_BS=16`: RKL then takes 0.41 s per update instead of 0.72 s
(measured for RKL and CL-LoRA; DistiLLM/AMiD at 16 are not measured).

End-to-end check, TACRED perm0, all 10 tasks, old code (`4257d86`) vs this branch, sharing one H200:

| method | final-task trigger F1, old → new | wall-clock, old → new |
|---|---|---|
| RKL (distillation) | 63.79 → 65.56 | 3 h 24 min → 1 h 05 min (3.1×) |
| IncLoRA (CL-LoRA) | 63.95 → 63.06 | 3 h 26 min → 1 h 10 min (2.9×) |

The four runs ran at the same time on one H200 NVL, without MPS. The new runs used this
branch's defaults: `PHYS_BS=8`; one test evaluation per task where the old RKL runner
evaluated dev and test after each of its 5 epochs; CL-LoRA evaluation batches of 128
instead of 16. The final-task test set has 1,240 triggers, so one point is about 12 of
them. RKL's evaluation samples at T=0.5 in both trees (see Decoding above). IncLoRA's
evaluation is greedy; its per-task F1 differs between the trees in both directions, by
3.4 points at most (after task 1).

## Results

```bash
python tools/ced_collect.py --host-label <label> [--upload]
```
