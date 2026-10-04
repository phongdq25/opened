#!/bin/bash
# CRE distillation baselines (7 methods) for one dataset+permutation, on ONE GPU.
#
#   bash scripts/qwen/cre/run_cre_dist.sh tacred 0 0          # dataset perm gpu
#   PORT_PREFIX=67 bash scripts/qwen/cre/run_cre_dist.sh fewrel 0 1
#
# All seven methods share a single task0 (plain CE, no KD) so the comparison is not
# polluted by task0 training noise, which we measured at +-1.6 F1 on ACE.
#
# Guards, each one paid for by a real failure earlier in this project:
#   disk    - /mnt hit 0 bytes twice and killed runs mid torch.save
#   gpu     - other users' jobs grew into our card and OOM'd the eval step (needs one
#             ~7 GB allocation), so wait for real headroom before starting
#   port    - two queues drew the same torchrun master port -> EADDRINUSE; give each
#             concurrent queue its own PORT_PREFIX
#   skip    - completion check must accept BOTH log layouts (taskN/log.txt and
#             taskN/<config>/log.txt) or a relaunch silently retrains finished runs
# IMPORTANT: run at most ONE queue per GPU. The gpu guard is a snapshot, not a
# reservation, so two queues on one card will still collide between tasks.
set -uo pipefail

DS=${1:?dataset required: tacred|fewrel}
PERM=${2:?perm required: 0..4}
GPU=${3:?gpu id required}
NTASK=${NTASK:-10}
LAST=$((NTASK - 1))
RANK=${RANK:-16}; ALPHA=${ALPHA:-64}; EPOCHS=${EPOCHS:-5}; LR=${LR:-0.0002}; SEED=${SEED:-42}
BS=${BS:-2}; ACC=${ACC:-16}
NEED_GPU_MB=${NEED_GPU_MB:-25000}
NEED_DISK_GB=${NEED_DISK_GB:-10}
PORT_PREFIX=${PORT_PREFIX:-66}
METHODS=${METHODS:-"rkl csd sfkl fkl srkl amid distillm"}
TAG=${TAG:-cre}

cd "$(dirname "$0")/../../.."
ENV_BIN=${ENV_BIN:-$HOME/miniconda3/envs/mta/bin}
export CUDA_HOME=${CUDA_HOME:-$HOME/miniconda3/envs/mta}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

BASE_RUNNER=scripts/qwen/ced/run_ced_v2.sh
RUNNER=${BASE_RUNNER}
if [ "${PORT_PREFIX}" != "66" ]; then
    RUNNER=scripts/qwen/ced/run_ced_v2_p${PORT_PREFIX}.sh
    sed "s/MASTER_PORT=66\$/MASTER_PORT=${PORT_PREFIX}\$/" ${BASE_RUNNER} > ${RUNNER}
fi

ADAPT="--student-gen --init-threshold 0.0 --loss-eps 0.1 --capacity 1000"
AMID_ARGS="--amid-div-name ab --amid-div-order pr --amid-alpha 0.5 --amid-lam 0.5"
T0RUN=${TAG}_${DS}_task0_perm${PERM}
TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i ${GPU} | tr -dc '0-9')

log()  { echo "[$(date '+%m-%d %H:%M')] $*"; }
done_already () {   # $1 = run name
    ls results/qwen3/ced/$1/task${LAST}/log.txt   >/dev/null 2>&1 && return 0
    ls results/qwen3/ced/$1/task${LAST}/*/log.txt >/dev/null 2>&1 && return 0
    return 1
}
task_f1 () {  # $1=run $2=task number
    local f
    f=$(find results/qwen3/ced/$1/task$2 -name log.txt 2>/dev/null | head -1)
    [ -n "$f" ] && grep '^test |' "$f" | tail -1 \
        | grep -oE "'trigger': \{'precision': [0-9.]+, 'recall': [0-9.]+, 'f1': [0-9.]+" \
        | grep -oE '[0-9.]+$'
}
final_f1 () { task_f1 "$1" "${LAST}"; }   # F1 of the run's last task
wait_disk () {
    local free
    while true; do
        # DISK_PATH: /mnt is the scratch mount on A40_3/A40_4. On a host without it df
        # errors, free comes back empty, and the loop below waits forever instead of training.
        free=$(df --output=avail -BG "${DISK_PATH:-.}" 2>/dev/null | tail -1 | tr -dc '0-9')
        [ "${free:-0}" -ge "${NEED_DISK_GB}" ] && break
        log "DISK-WAIT ${free}G < ${NEED_DISK_GB}G"; sleep 300
    done
}
wait_gpu () {
    local used free
    while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${GPU} | tr -dc '0-9')
        free=$((TOTAL - ${used:-TOTAL}))
        [ "${free}" -ge "${NEED_GPU_MB}" ] && break
        log "GPU-WAIT gpu${GPU} free=${free}MiB < ${NEED_GPU_MB}"; sleep 600
    done
}
dump_fail () {
    local t
    t=$(ls -t results/qwen3/ced/$1/task*/train.log 2>/dev/null | head -1)
    [ -n "$t" ] && grep -E "OutOfMemory|Error|Traceback|Killed|Signal|EADDRINUSE|assert" "$t" | tail -4
}

log "CRE-DIST START ds=${DS} perm=${PERM} gpu=${GPU} tasks=${NTASK} methods='${METHODS}'"

# ---- shared task0 (plain CE) ----
if [ ! -d results/qwen3/ced/${T0RUN}/task0/merged ]; then
    # merged/ is deleted by the cleanup at the bottom of this script after every successful
    # sweep (disk space), so a re-run always lands here even when the rest of ${T0RUN} (log.txt,
    # run_manifest.json) is still on disk from that earlier sweep. run_ced_v2.sh refuses to
    # write into a non-empty run dir, so without this rm -rf the retrain fails immediately with
    # "refusing to overwrite existing run" (this is exactly what stranded cre_tacred_task0_perm0
    # with no F1 -- the guard fired on every later attempt and nobody cleared the stale dir).
    rm -rf results/qwen3/ced/${T0RUN}
    wait_disk; wait_gpu
    log "=== ${T0RUN} (task0, plain CE) ==="
    bash ${RUNNER} --run-name ${T0RUN} --data-prefix ${DS}_perm --perm ${PERM} \
        --rank ${RANK} --alpha ${ALPHA} --epochs ${EPOCHS} --lr ${LR} --seed ${SEED} \
        --bs ${BS} --acc ${ACC} --greedy 1 --gpus "${GPU}" --start-task 0 --end-task 0 \
        > logs_${T0RUN}.log 2>&1 || { log "FAILED ${T0RUN}"; dump_fail ${T0RUN}; exit 1; }
    log "TASK0 DONE f1=$(task_f1 ${T0RUN} 0 || true)"
fi
T0=$PWD/results/qwen3/ced/${T0RUN}/task0/merged
[ -d "${T0}" ] || { log "khong co teacher task0"; exit 1; }

# ---- the seven methods, tasks 1..LAST from that shared task0 ----
for M in ${METHODS}; do
    RUN=${TAG}_${DS}_${M}_perm${PERM}
    if done_already ${RUN}; then log "SKIP ${RUN}"; continue; fi
    case ${M} in
        distillm) KD_TYPE=adaptive-srkl; EXTRA="${ADAPT}" ;;
        amid)     KD_TYPE=adaptive-amid; EXTRA="${ADAPT} ${AMID_ARGS}" ;;
        *)        KD_TYPE=${M};          EXTRA="" ;;
    esac
    wait_disk; wait_gpu
    rm -rf results/qwen3/ced/${RUN}
    log "=== ${RUN} (type=${KD_TYPE}) ==="
    # --task0-source-run reads the teacher straight from ${T0RUN}/task0/merged and copies
    # its log for provenance; it does this AFTER the runner's own overwrite guard, so we
    # must not pre-create ${RUN}/task0 ourselves (that used to trip the guard).
    if bash ${RUNNER} --run-name ${RUN} --mode ce_kd --data-prefix ${DS}_perm --perm ${PERM} \
        --kd-type "${KD_TYPE}" --w-span 0 --kd-ratio 0.9 --skew 0.1 --span-metric cosine \
        --layers "22 25 28" --rank ${RANK} --alpha ${ALPHA} --epochs ${EPOCHS} --lr ${LR} --seed ${SEED} \
        --bs ${BS} --acc ${ACC} --greedy 1 --gpus "${GPU}" --start-task 1 --end-task ${LAST} \
        --task0-source-run ${T0RUN} --extra "${EXTRA}" > logs_${RUN}.log 2>&1; then
        log "OK ${RUN} final=$(final_f1 ${RUN} || echo NA)"
    else
        log "FAILED ${RUN}"; dump_fail ${RUN}
    fi
    rm -rf results/qwen3/ced/${RUN}/task*/merged
done
rm -rf results/qwen3/ced/${T0RUN}/task0/merged
log "CRE-DIST ALL DONE ds=${DS} perm=${PERM}"
