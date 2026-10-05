#! /bin/bash
# Full 5-task CL-LoRA baseline runs, one method after another on a single GPU.
# Split methods across GPUs by launching two of these (one per GPU).
# Batch 32x1 since 2026-09-29 (128 and 64 OOM). Runs finished before 2026-09-28 (RAMS perm0) used 2x16 = 32.
#   bash scripts/qwen/ced/run_all_cllora.sh 0 inclora olora tree inflora
#   bash scripts/qwen/ced/run_all_cllora.sh 1 migu epi gainlora_o gainlora_inf
set -euo pipefail

GPU=${1:?GPU id required}; shift
DATA_ROOT=${DATA_ROOT:-data_ced/ace_b10_perm0}
EPOCHS=${EPOCHS:-5}
NUM_TASKS=${NUM_TASKS:-5}
PY=${PY:-$HOME/miniconda3/envs/nuquant/bin/python}
cd "$(dirname "$0")/../../.." || exit 1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} PYTHONPATH=.
RESUME_ARGS=()
[ "${RESUME:-0}" = "1" ] && RESUME_ARGS+=(--resume)
for M in "$@"; do
    echo "===== FULL RUN ${M} on GPU${GPU} ($(date +%H:%M)) ====="
    bash scripts/qwen/ced/run_cllora.sh \
        --method "${M}" --data-root "${DATA_ROOT}" --num-tasks "${NUM_TASKS}" \
        --rank "${RANK:-16}" --alpha 64 --lr 2e-4 --epochs "${EPOCHS}" \
        --batch-size 32 --grad-accum 1 --eval-batch-size "${EVAL_BS:-128}" \
        --gpu "${GPU}" --py "${PY}" --protocol "${PROTOCOL:-v2}" \
        "${RESUME_ARGS[@]}"
done
echo "ALL FULL RUNS DONE (GPU${GPU}): $*"
