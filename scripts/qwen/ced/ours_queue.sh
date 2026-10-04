#!/bin/bash
# OUR method (Phase 1 / CL-DETR-inspired) on one dataset permutation.
# f12 protocol + H2 conflict-dedup (always on in the tool) + H1 confidence
# filter + H3 matched-distribution calibration epoch, on top of the shared
# task0 checkpoint used by dist_queue.sh (same SFT config, so it is reused
# and only trained once per perm).
#
# Env knobs:
#   PERM=0 GPU=0 SEED=42 PROTOCOL=v2 DATA_PREFIX=ace_b10_perm RESUME=0
#   OURS_VARIANT=full   # h2   = +pseudo-label dedup + boost (no conf filter)
#                       # h12  = h2  + confidence percentile filter
#                       # full = h12 + matched calibration epoch (H3)
#   PL_PCT=70           # H1 percentile
#   OURS_SD=0|1         # on-policy self-distillation on top (run name gets _sd)
#   SD_ARGS="..."       # extra runner flags for SD, e.g. "--sd-temp 0.7 --sd-top-p 0.9"
#   RUN_SUFFIX=_t07     # appended to the run name; keeps ablation/smoke arms apart
set -euo pipefail

cd "$(dirname "$0")/../../.." || exit 1
PERM=${PERM:-0}
SEED=${SEED:-42}
PROTOCOL=${PROTOCOL:-v2}
GPU=${GPU:-0}
RESUME=${RESUME:-0}
DATA_PREFIX=${DATA_PREFIX:-ace_b10_perm}
OURS_VARIANT=${OURS_VARIANT:-full}
PL_PCT=${PL_PCT:-70}
OURS_SD=${OURS_SD:-0}
SD_ARGS=${SD_ARGS:-}
RUN_SUFFIX=${RUN_SUFFIX:-}
SHARED="dist_shared_task0_perm${PERM}_${PROTOCOL}_s${SEED}"

# shared task0 (plain SFT) — identical config to dist_queue's, reuse or create
if [ ! -f "results/qwen3/ced/${SHARED}/.complete" ]; then
    [ ! -e "results/qwen3/ced/${SHARED}" ] || {
        echo "incomplete shared task0 already exists: ${SHARED}"
        exit 1
    }
    echo "===== shared task0 ${SHARED} $(date) ====="
    bash scripts/qwen/ced/run_ced_v2.sh \
        --run-name "${SHARED}" --mode sft --data-prefix "${DATA_PREFIX}" --perm "${PERM}" \
        --rank 16 --alpha 64 --epochs 5 --lr 0.0002 --seed "${SEED}" \
        --bs 2 --acc 16 --greedy 1 --gpus "${GPU}" --end-task 0 \
        > "logs_${SHARED}.log" 2>&1
fi

VARIANT_ARGS=()
case "${OURS_VARIANT}" in
    h2)   ;;
    h12)  VARIANT_ARGS+=(--pl-conf percentile --pl-conf-pct "${PL_PCT}") ;;
    full) VARIANT_ARGS+=(--pl-conf percentile --pl-conf-pct "${PL_PCT}"
                         --balanced-epoch 1 --balance-dist matched) ;;
    *) echo "unknown OURS_VARIANT '${OURS_VARIANT}' (h2|h12|full)"; exit 1 ;;
esac

# SD gets its own run name: without it `ours` and `ours + SD` at the same perm collide, and
# the second is either refused (run exists) or, under RESUME=1, silently continues the first.
SD_TAG=""
if [ "${OURS_SD}" = "1" ]; then
    # SD_ARGS is split on purpose: it carries several runner flags
    # shellcheck disable=SC2206
    VARIANT_ARGS+=(--sd 1 ${SD_ARGS})
    SD_TAG="_sd"
fi

RUN_NAME="ours_${OURS_VARIANT}${SD_TAG}${RUN_SUFFIX}_perm${PERM}_${PROTOCOL}_s${SEED}"
START_TASK=1
RESUME_ARGS=()
if [ -e "results/qwen3/ced/${RUN_NAME}" ]; then
    [ "${RESUME}" = "1" ] || { echo "run exists: ${RUN_NAME} (set RESUME=1 to continue)"; exit 1; }
    RUN_ROOT="results/qwen3/ced/${RUN_NAME}"
    [ -f "${RUN_ROOT}/run_manifest.json" ] || { echo "no manifest to resume from: ${RUN_NAME}"; exit 1; }
    START_TASK=$(${PY:-${ENV_BIN:-$HOME/miniconda3/envs/mta/bin}/python} -c \
        "import json; print(json.load(open('${RUN_ROOT}/run_manifest.json'))['completed_task'] + 1)")
    RESUME_ARGS+=(--resume)
fi

echo "===== ${RUN_NAME} $(date) ====="
bash scripts/qwen/ced/run_ced_v2.sh --run-name "${RUN_NAME}" --mode ce_kd --data-prefix "${DATA_PREFIX}" --perm "${PERM}" \
    --kd-type sfkl --w-span 2.0 --kd-ratio 0.9 --skew 0.1 --span-metric cosine --layers "22 25 28" \
    --rank 16 --alpha 64 --epochs 5 --lr 0.0002 --seed "${SEED}" --bs 2 --acc 16 \
    --pl 1 --replay-boost 5 \
    --greedy 1 --gpus "${GPU}" --start-task "${START_TASK}" \
    --task0-source-run "${SHARED}" "${VARIANT_ARGS[@]}" "${RESUME_ARGS[@]}" "$@" \
    >> "logs_${RUN_NAME}.log" 2>&1
[ -f "results/qwen3/ced/${RUN_NAME}/.complete" ] || {
    echo "missing completion marker: ${RUN_NAME}"
    exit 1
}
echo "DONE ${RUN_NAME}"
