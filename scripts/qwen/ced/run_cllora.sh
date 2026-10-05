#! /bin/bash
# CL-LoRA baseline runner (STANDALONE engine, single GPU, Qwen3-0.6B).
# Runs the whole task sequence in one process so per-task adapters persist (Family A).
# Reuses our raw JSONL data + ed_eval for a fair comparison. No deepspeed/torchrun —
# plain python under any torch+peft+transformers env (default: nuquant on A100).
#   bash scripts/qwen/ced/run_cllora.sh --method olora --data-root data_ced/ace_b10_perm0 --gpu 0
# methods: inclora olora migu tree inflora gainlora_o gainlora_inf epi
set -euo pipefail

cd "$(dirname "$0")/../../.."
# local copy from download.txt when present (see run.sh), else the hub name
[ -f models/Qwen3-0.6B/config.json ] && MODEL_PATH=${MODEL_PATH:-models/Qwen3-0.6B}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}

METHOD=""; DATA_ROOT="data_ced/ace_b10_perm0"; NUM_TASKS=5; RANK=16; ALPHA=64
LR=0.0002; EPOCHS=5; BS=2; GA=16; EBS=128; REG=0.5; MIGU=0.7; SEED=42; GPU=0
PROTOCOL="v2"; SAVE=""
RESUME=0
END_TASK=""
LIMIT=-1
PY=$HOME/miniconda3/envs/nuquant/bin/python
while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD=$2; shift 2;;
        --data-root) DATA_ROOT=$2; shift 2;;
        --num-tasks) NUM_TASKS=$2; shift 2;;
        --rank) RANK=$2; shift 2;;
        --alpha) ALPHA=$2; shift 2;;
        --lr) LR=$2; shift 2;;
        --epochs) EPOCHS=$2; shift 2;;
        --batch-size) BS=$2; shift 2;;
        --grad-accum) GA=$2; shift 2;;
        --eval-batch-size) EBS=$2; shift 2;;
        --reg) REG=$2; shift 2;;
        --migu-ratio) MIGU=$2; shift 2;;
        --gpu) GPU=$2; shift 2;;
        --seed) SEED=$2; shift 2;;
        --py) PY=$2; shift 2;;
        --protocol) PROTOCOL=$2; shift 2;;
        --save) SAVE=$2; shift 2;;
        --resume) RESUME=1; shift;;
        --end-task) END_TASK=$2; shift 2;;
        --limit) LIMIT=$2; shift 2;;
        *) echo "unknown flag $1"; exit 1;;
    esac
done
[ -z "${METHOD}" ] && { echo "--method required"; exit 1; }
source scripts/qwen/lib.sh
read -r PBS PGA <<< "$(phys_split "${BS}" "${GA}" "${PHYS_BS:-${BS}}")"
[ "${METHOD}" = "migu" ] && { PBS=${BS}; PGA=${GA}; }   # MIGU's gradient mask comes from each step's activations
[ -x "${PY}" ] || { echo "python not executable: ${PY}"; exit 1; }
[ -s "${DATA_ROOT}/streams.json" ] || { echo "missing ${DATA_ROOT}/streams.json"; exit 1; }
for TASK in $(seq 0 $((NUM_TASKS - 1))); do
    for SPLIT in train dev test; do
        [ -s "${DATA_ROOT}/${TASK}/${SPLIT}.jsonl" ] || {
            echo "missing ${DATA_ROOT}/${TASK}/${SPLIT}.jsonl"
            exit 1
        }
    done
done

PERM_TAG=$(basename "${DATA_ROOT}" | sed -n 's/.*\(perm[0-9][0-9]*\).*/\1/p')
[ -n "${PERM_TAG}" ] || { echo "cannot infer permutation from ${DATA_ROOT}"; exit 1; }
RUN_ID="cllora_${METHOD}_${PERM_TAG}_${PROTOCOL}_s${SEED}"
[ -n "${SAVE}" ] || SAVE="results/qwen3/ced/${RUN_ID}"
# A finished run is a no-op, not an error. run.sh sweeps whole perm ranges so that only the
# missing runs train; without this the overwrite guard below turns "already done" into exit 1,
# which kills the rest of the queue (run_all_cllora.sh runs under set -e) the moment it reaches
# a perm that finished earlier. run_cre_cllora.sh has always skipped on this marker; the CED
# path did not, so re-running MAVEN perms 0-4 died on perm0.
if [ -f "${SAVE}/.complete" ]; then
    echo "SKIP complete: ${SAVE}"
    exit 0
fi
if [ "${RESUME}" = "0" ]; then
    [ ! -e "${SAVE}/run_manifest.json" ] && [ ! -e "${SAVE}/cl_results.json" ] || {
        echo "refusing to overwrite existing run: ${SAVE}"
        exit 1
    }
fi
mkdir -p "${SAVE}"
# offline by default; override with HF_HUB_OFFLINE=0 on hosts where transformers'
# tokenizer load hits the hub API (4.57 _patch_mistral_regex bug) and network is available
export CUDA_VISIBLE_DEVICES=${GPU} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} PYTHONPATH=.

RESUME_ARG=()
[ "${RESUME}" = "1" ] && RESUME_ARG+=(--resume)
END_TASK_ARG=()
[ -n "${END_TASK}" ] && END_TASK_ARG+=(--end-task "${END_TASK}")
# Flat logs/<ds>_cllora_<method>_perm<p>_: train.log is the engine's whole output. The engine
# trains every task in one process, so there is one file per perm, not per task; results.log is
# the per-task metrics from cl_results.json.
DS=$(basename "${DATA_ROOT}")
LOG_PRE="logs/${DS%%_*}_cllora_${METHOD}_${PERM_TAG}"
mkdir -p logs
LOG="${LOG_PRE}_train.log"
if [ "${RESUME}" = "1" ]; then
    echo "===== RESUME $(date -Iseconds) =====" >> "${LOG}"
else
    : > "${LOG}"
fi
${PY} cl_lora/engine.py \
    --cl-method "${METHOD}" --data-root "${DATA_ROOT}" --num-tasks "${NUM_TASKS}" \
    --rank "${RANK}" --alpha "${ALPHA}" --lr "${LR}" --epochs "${EPOCHS}" \
    --batch-size "${PBS}" --grad-accum "${PGA}" --loss-group-size "${BS}" --eval-batch-size "${EBS}" \
    --cl-reg "${REG}" --cl-migu-ratio "${MIGU}" --seed "${SEED}" \
    --limit "${LIMIT}" --model-path "${MODEL_PATH}" \
    --save "${SAVE}" "${RESUME_ARG[@]}" "${END_TASK_ARG[@]}" \
    >> "${LOG}" 2>&1
"${PY}" -m json.tool "${SAVE}/cl_results.json" > "${LOG_PRE}_results.log"
if [ -z "${END_TASK}" ] || [ "${END_TASK}" -eq $((NUM_TASKS - 1)) ]; then
    [ -f "${SAVE}/.complete" ] || { echo "missing completion marker: ${SAVE}"; exit 1; }
    echo "CLLORA RUNNER DONE ${METHOD} -> ${SAVE}"
else
    grep -q "CL-LORA PARTIAL ${METHOD}" "${LOG}" || {
        echo "missing partial completion marker: ${METHOD}"
        exit 1
    }
    echo "CLLORA RUNNER PARTIAL ${METHOD} -> ${SAVE}"
fi
