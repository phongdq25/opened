#!/bin/bash
# CRE CL-LoRA baselines (8 methods) for one dataset+permutation, on ONE GPU.
# The standalone engine reads the raw JSONL, so no processed_data is needed for these.
#
#   bash scripts/qwen/cre/run_cre_cllora.sh tacred 0 1                 # dataset perm gpu
#   METHODS="tree inclora" bash scripts/qwen/cre/run_cre_cllora.sh fewrel 0 1
#
# A single CL-LoRA run takes roughly 8-10 GB and does not saturate an A40 on its own, so
# 2-3 of these can share one card productively. Check utilisation first: once the card
# reads ~100% there is nothing left to win and extra runs only add OOM risk.
# MIGU is full fine-tuning (all 606M params) and needs far more memory than the LoRA
# methods, hence its own threshold.
set -uo pipefail

DS=${1:?dataset required: tacred|fewrel}
PERM=${2:?perm required: 0..4}
GPU=${3:?gpu id required}
NTASK=${NTASK:-10}
RANK=${RANK:-16}; ALPHA=${ALPHA:-64}; EPOCHS=${EPOCHS:-5}; LR=${LR:-2e-4}; SEED=${SEED:-42}
BS=${BS:-2}; ACC=${ACC:-16}; EBS=${EBS:-16}
NEED_LORA_MB=${NEED_LORA_MB:-14000}
NEED_MIGU_MB=${NEED_MIGU_MB:-28000}
NEED_DISK_GB=${NEED_DISK_GB:-10}
PROTOCOL=${PROTOCOL:-cre}
METHODS=${METHODS:-"tree inclora olora inflora epi migu gainlora_o gainlora_inf"}

cd "$(dirname "$0")/../../.."
PY=${PY:-$HOME/miniconda3/envs/mta/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# peft needs to reach the hub to validate the base model; offline mode raises instead of
# falling back to the local cache on older huggingface_hub builds.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i ${GPU} | tr -dc '0-9')

log () { echo "[$(date '+%m-%d %H:%M')] $*"; }
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
wait_mem () {   # $1 = MiB needed
    local need=$1 used free
    while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${GPU} | tr -dc '0-9')
        free=$((TOTAL - ${used:-TOTAL}))
        [ "${free}" -ge "${need}" ] && break
        log "MEM-WAIT gpu${GPU} free=${free}MiB < ${need}"; sleep 600
    done
}

log "CRE-CLLORA START ds=${DS} perm=${PERM} gpu=${GPU} tasks=${NTASK} methods='${METHODS}'"
for M in ${METHODS}; do
    # let run_cllora.sh own the naming so the run dir and its log file stay in sync:
    # RUN_ID = cllora_<method>_<permtag>_<protocol>_s<seed>
    RUN_ID=cllora_${M}_perm${PERM}_${DS}_${PROTOCOL}_s${SEED}
    RUN=results/qwen3/ced/${RUN_ID}
    if [ -f "${RUN}/.complete" ]; then log "SKIP ${M} (xong)"; continue; fi
    # another lane on the same card may already be on this method
    if ps aux | grep "cl-method ${M} " | grep -v grep > /dev/null; then
        log "${M} dang chay o lane khac, cho..."
        while ps aux | grep "cl-method ${M} " | grep -v grep > /dev/null; do sleep 300; done
        [ -f "${RUN}/.complete" ] && { log "SKIP ${M} (lane khac lam xong)"; continue; }
    fi
    if [ "${M}" = "migu" ]; then NEED=${NEED_MIGU_MB}; else NEED=${NEED_LORA_MB}; fi
    wait_disk; wait_mem ${NEED}
    rm -rf "${RUN}"
    log "=== ${M} ==="
    if bash scripts/qwen/ced/run_cllora.sh --method ${M} --data-root data/${DS}_perm${PERM} \
        --num-tasks ${NTASK} --rank ${RANK} --alpha ${ALPHA} --lr ${LR} --epochs ${EPOCHS} \
        --batch-size ${BS} --grad-accum ${ACC} --eval-batch-size ${EBS} \
        --gpu ${GPU} --py ${PY} --protocol ${DS}_${PROTOCOL}; then
        V=$(${PY} -c "
import json
d = json.load(open('${RUN}/cl_results.json'))
k = 'task$((NTASK - 1))'
print(round(d[k]['trigger']['f1'], 4) if k in d else 'NA')
" 2>/dev/null)
        log "OK ${M} final=${V:-NA}"
    else
        log "FAILED ${M}, see logs_${RUN_ID}.log"
        grep -E "Error|Traceback|Killed|OutOfMemory|EADDRINUSE|assert|dtype" logs_${RUN_ID}.log 2>/dev/null | tail -5
    fi
    rm -f "${RUN}/checkpoint_latest.pt" "${RUN}/checkpoint_latest.pt.tmp"
done
log "CRE-CLLORA ALL DONE ds=${DS} perm=${PERM}"
