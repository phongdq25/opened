#!/bin/bash
# Seven distillation baselines with a shared task0 checkpoint and the f12_pl protocol.
# Eff batch 32 everywhere. Since 2026-09-29 micro batch 32 per GPU (128 and 64 OOM), capped at 16
# for rkl/sfkl/srkl/distillm/amid since 2026-09-30 (32 OOM in the loss); GPU="a b" splits it
# across GPUs. Runs finished before 2026-09-28 (RAMS perm0) used 2x16 = 32.
set -euo pipefail

cd "$(dirname "$0")/../../.." || exit 1
PERM=${PERM:-0}
SEED=${SEED:-42}
PROTOCOL=${PROTOCOL:-v2}
GPU=${GPU:-0}   # one id, or several space-separated for one data-parallel run
NGPU=$(wc -w <<< "${GPU}")
RESUME=${RESUME:-0}
DATA_PREFIX=${DATA_PREFIX:-ace_b10_perm}
# Which of the seven methods to train (default all). run.sh splits them across GPUs, and first
# calls this with DIST_METHODS="" so only the shared task0 trains before the split.
DIST_METHODS=${DIST_METHODS-"kd rkl sfkl srkl csd distillm amid"}
SHARED="dist_shared_task0_perm${PERM}_${PROTOCOL}_s${SEED}"
NUM_TASKS=$(${ENV_BIN:-$HOME/miniconda3/envs/mta/bin}/python -c \
    "import json,sys; print(len(json.load(open(sys.argv[1]))))" "data/${DATA_PREFIX}${PERM}/streams.json")

# Flat logs/, one prefix per run, logs/<ds>_dist_<method>_perm<p>_: steps.log (runner output),
# task<t>.log (training output), results.log (per-task train/eval log.txt) and merge.log
# (LoRA merge output), the last two written when the run ends.
# run_ced_v2.sh writes the training output to results/<run>/task<t>/train.log and is
# fingerprinted into every run manifest, so it stays as is: task<t>.log is a symlink to that
# file while the run trains, a copy once it ends.
LOGS="logs/${DATA_PREFIX%%_*}_dist"
mkdir -p logs
micro_acc () {  # $1=max micro batch per GPU -> "micro acc" giving eff batch 32 over NGPU GPUs
    local mb=$((32 / NGPU))
    [ "${mb}" -le "$1" ] || mb=$1
    echo "${mb} $((32 / (mb * NGPU)))"
}
link_logs () {  # $1=run name $2=log prefix $3..=tasks
    local run=$1 pre=$2 t; shift 2
    for t in "$@"; do
        ln -sfn "${PWD}/results/qwen3/ced/${run}/task${t}/train.log" "${pre}_task${t}.log"
    done
}
freeze_logs () {  # $1=run name $2=log prefix
    local run=$1 pre=$2 f
    for f in "${pre}"_task*.log; do
        [ -L "${f}" ] || continue
        if [ -e "${f}" ]; then cp --remove-destination "$(readlink -f "${f}")" "${f}"; else rm -f "${f}"; fi
    done
    : > "${pre}_results.log"
    for f in $(find "results/qwen3/ced/${run}" -name log.txt 2>/dev/null | sort -V); do
        echo "===== ${f#results/qwen3/ced/${run}/} =====" >> "${pre}_results.log"
        cat "${f}" >> "${pre}_results.log"
    done
    : > "${pre}_merge.log"
    for f in $(find "results/qwen3/ced/${run}" -name merge.log 2>/dev/null | sort -V); do
        echo "===== ${f#results/qwen3/ced/${run}/} =====" >> "${pre}_merge.log"
        cat "${f}" >> "${pre}_merge.log"
    done
}

if [ ! -f "results/qwen3/ced/${SHARED}/.complete" ]; then
    [ ! -e "results/qwen3/ced/${SHARED}" ] || {
        echo "incomplete shared task0 already exists: ${SHARED}"
        exit 1
    }
    echo "===== shared task0 ${SHARED} $(date) ====="
    PRE="${LOGS}_shared_task0_perm${PERM}"
    link_logs "${SHARED}" "${PRE}" 0
    rc=0
    read -r BS ACC <<< "$(micro_acc 32)"
    bash scripts/qwen/ced/run_ced_v2.sh \
        --run-name "${SHARED}" --mode sft --data-prefix "${DATA_PREFIX}" --perm "${PERM}" \
        --rank 16 --alpha 64 --epochs 5 --lr 0.0002 --seed "${SEED}" \
        --bs "${BS}" --acc "${ACC}" --greedy 1 --gpus "${GPU}" --end-task 0 \
        > "${PRE}_steps.log" 2>&1 || rc=$?
    freeze_logs "${SHARED}" "${PRE}"
    [ "${rc}" -eq 0 ] || exit "${rc}"
fi

run_dist () {  # $1=method label $2=kd-type ($3...=optional runner flags)
    local METHOD=$1; local KD_TYPE=$2; shift 2
    [[ " ${DIST_METHODS} " == *" ${METHOD} "* ]] || return 0
    local RUN_NAME="dist_${METHOD}_perm${PERM}_${PROTOCOL}_s${SEED}"
    local RUN_ROOT="results/qwen3/ced/${RUN_NAME}"
    # RKL/SKL/AMiD losses hold ~10 vocab-sized fp32 tensors, 13.9 GiB each at micro batch 32:
    # OOM on a 178 GiB GPU (RAMS 2026-09-29), so they cap at 16 per GPU.
    local BS ACC MAX_MB=32
    case ${METHOD} in rkl|sfkl|srkl|distillm|amid) MAX_MB=16 ;; esac
    read -r BS ACC <<< "$(micro_acc "${MAX_MB}")"
    if [ -f "${RUN_ROOT}/.complete" ]; then
        echo "SKIP complete ${RUN_NAME}"
        return
    fi
    local START_TASK=1
    local RESUME_ARGS=()
    if [ -e "${RUN_ROOT}" ]; then
        [ "${RESUME}" = "1" ] || {
            echo "partial run exists; set RESUME=1: ${RUN_NAME}"
            exit 1
        }
        START_TASK=$(${ENV_BIN:-$HOME/miniconda3/envs/mta/bin}/python -c \
            "import json; print(json.load(open('${RUN_ROOT}/run_manifest.json'))['completed_task'] + 1)")
        RESUME_ARGS+=(--resume)
    fi
    echo "===== ${RUN_NAME} (${KD_TYPE}) $(date) ====="
    local PRE="${LOGS}_${METHOD}_perm${PERM}" rc=0
    link_logs "${RUN_NAME}" "${PRE}" $(seq "${START_TASK}" $((NUM_TASKS - 1)))
    bash scripts/qwen/ced/run_ced_v2.sh --run-name "${RUN_NAME}" --mode ce_kd --data-prefix "${DATA_PREFIX}" --perm "${PERM}" \
        --kd-type "${KD_TYPE}" --w-span 0 --kd-ratio 0.9 --skew 0.1 --span-metric cosine --layers "22 25 28" \
        --rank 16 --alpha 64 --epochs 5 --lr 0.0002 --seed "${SEED}" --bs "${BS}" --acc "${ACC}" \
        --greedy 1 --gpus "${GPU}" --start-task "${START_TASK}" \
        --task0-source-run "${SHARED}" "${RESUME_ARGS[@]}" "$@" \
        >> "${PRE}_steps.log" 2>&1 || rc=$?
    freeze_logs "${RUN_NAME}" "${PRE}"
    [ "${rc}" -eq 0 ] || exit "${rc}"
    [ -f "results/qwen3/ced/${RUN_NAME}/.complete" ] || {
        echo "missing completion marker: ${RUN_NAME}"
        exit 1
    }
    echo "DONE ${RUN_NAME}"
}

run_dist kd kd
run_dist rkl rkl
run_dist sfkl sfkl
run_dist srkl srkl
run_dist csd csd
run_dist distillm adaptive-srkl \
    --extra "--student-gen --gen-do-sample --gen-top-p 1.0 --gen-temperature 1.0 --gen-num-beams 1 --init-threshold 0.0 --loss-eps 0.1 --capacity 1000"
run_dist amid adaptive-amid \
    --extra "--student-gen --gen-do-sample --gen-top-p 1.0 --gen-temperature 1.0 --gen-num-beams 1 --init-threshold 0.0 --loss-eps 0.1 --capacity 1000 --amid-div-name ab --amid-div-order pr --amid-alpha 0.5 --amid-lam 0.5"
echo "ALL DISTILL DONE $(date)"
