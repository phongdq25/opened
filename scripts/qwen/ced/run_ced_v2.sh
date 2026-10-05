#! /bin/bash
# CED runner v2 — F-ladder: composable pseudo-label (F2), replay oversampling (F1),
# masked-KD scope (F3, needs engine support), balanced calibration epoch (F4).
# New file so running sequences on older runners are never disturbed.
#
# Usage:
#   bash scripts/qwen/ced/run_ced_v2.sh --run-name f1_boost --rank 16 --data-prefix ace_b10_perm --replay-boost 5
# Flags on top of run_ced_seq.sh:
#   --pl 0|1             teacher pseudo-label with lexicon filter   [0]
#   --replay-boost K     duplicate replay exemplars K-1 extra times [1]
#   --kd-scope M         replay | pl (KD also on old-event tokens of pseudo rows) [replay]
#   --balanced-epoch 0|1 extra calibration epoch on type-balanced pool [0]
#   --balance-per-type N [10]   --balance-lr L [2e-5]
#   --sd 0|1             on-policy self-distillation, EMA teacher reads gold+PL answer [0]
#   --w-sd W [1.0]   --sd-mu M [0.99]   --sd-temp T [1.0]   --sd-div fkl|rkl [fkl]
#   ablation only, defaults = SDFT: --sd-top-p P [1.0]  --sd-skip-unparsed 0|1 [0]  --sd-warmup F [0]
#   --sd-omask 0|1 [0]   omission-mask: keep grounded old records missing from y~ out of the SD loss

set -euo pipefail

MODE=ce_kd; PERM=0; DATA_PREFIX=ace_perm; RUN_NAME=""
KD_RATIO=0.9; W_SPAN=2.0; KD_TYPE=sfkl; SKEW=0.1; SPAN_METRIC=cosine; LAYERS="22 25 28"
BS=2; ACC=8; LR=0.0002; LR_LATER=""; EPOCHS=5; SEED=42; RANK=8; ALPHA=64
GREEDY=0; START_TASK=0; END_TASK=""   # empty = derive from streams.json (ACE 5 tasks, CRE 10)
PL=0; BOOST=1; KD_SCOPE=replay; BAL=0; BAL_PT=10; BAL_LR=0.00002
PL_DEDUP=1      # H2: drop pseudo events whose trigger overlaps a gold trigger (any type)
PL_CONF=none    # H1: teacher-confidence filter {none, percentile, thresh}
PL_CONF_PCT=70; PL_CONF_THRESH=""
PL_LEXICON=1    # F2: (trigger,type) must occur in the old tasks' gold train data
BAL_DIST=uniform  # H3: calibration pool distribution {uniform (legacy F4), matched (CL-DETR-style)}
SELECT_BEST=0   # 1 = merge best-dev-F1 epoch per task instead of last epoch
KDNEW=0         # LwF: KD weight on new-task rows' non-new-type tokens (0 = off)
SD=0; W_SD=1.0; SD_MU=0.99; SD_TEMP=1.0; SD_DIV=fkl   # on-policy self-distillation (SDFT)
SD_TOP_P=1.0; SD_SKIP_UNPARSED=0; SD_WARMUP=0       # SD ablation knobs, SDFT defaults
SD_OMASK=0
GPUS_ARG="0 1"  # which GPUs to use (space-separated); e.g. "0" for single-GPU
EXTRA_ARGS=""   # raw extra flags appended to the ced_finetune step (e.g. DistiLLM off-policy)
TASK0_SOURCE_RUN=""
TRAIN_NUM=-1; DEV_NUM=-1
SMOKE_ROWS=0
EVAL_BS=${EVAL_BS:-128}                 # generation batch; the loss pass keeps 32-row chunks
PHYS_BS=${PHYS_BS:-}                    # physical batch target, empty = --bs (scripts/qwen/lib.sh phys_split)
EVAL_GEN_MODE=${EVAL_GEN_MODE:-final}   # final: test answers after each task's last update only
DYNAMIC_PAD=${DYNAMIC_PAD:-1}           # the trainer keeps fixed padding with a span loss or DistiLLM
STRICT_GEN=${STRICT_GEN:-0}             # 1 = generation configs used as written (gen_config.py): changes results
RESUME=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --run-name) RUN_NAME=$2; shift 2;;
        --mode) MODE=$2; shift 2;;
        --perm) PERM=$2; shift 2;;
        --data-prefix) DATA_PREFIX=$2; shift 2;;
        --kd-ratio) KD_RATIO=$2; shift 2;;
        --w-span) W_SPAN=$2; shift 2;;
        --kd-type) KD_TYPE=$2; shift 2;;
        --skew) SKEW=$2; shift 2;;
        --span-metric) SPAN_METRIC=$2; shift 2;;
        --layers) LAYERS=$2; shift 2;;
        --bs) BS=$2; shift 2;;
        --acc) ACC=$2; shift 2;;
        --lr) LR=$2; shift 2;;
        --lr-later) LR_LATER=$2; shift 2;;
        --epochs) EPOCHS=$2; shift 2;;
        --seed) SEED=$2; shift 2;;
        --rank) RANK=$2; shift 2;;
        --alpha) ALPHA=$2; shift 2;;
        --greedy) GREEDY=$2; shift 2;;
        --start-task) START_TASK=$2; shift 2;;
        --end-task) END_TASK=$2; shift 2;;
        --pl) PL=$2; shift 2;;
        --pl-dedup) PL_DEDUP=$2; shift 2;;
        --pl-conf) PL_CONF=$2; shift 2;;
        --pl-conf-pct) PL_CONF_PCT=$2; shift 2;;
        --pl-conf-thresh) PL_CONF_THRESH=$2; shift 2;;
        --pl-lexicon) PL_LEXICON=$2; shift 2;;
        --balance-dist) BAL_DIST=$2; shift 2;;
        --replay-boost) BOOST=$2; shift 2;;
        --kd-scope) KD_SCOPE=$2; shift 2;;
        --balanced-epoch) BAL=$2; shift 2;;
        --balance-per-type) BAL_PT=$2; shift 2;;
        --balance-lr) BAL_LR=$2; shift 2;;
        --select-best-dev) SELECT_BEST=$2; shift 2;;
        --kd-ratio-new) KDNEW=$2; shift 2;;
        --sd) SD=$2; shift 2;;
        --w-sd) W_SD=$2; shift 2;;
        --sd-mu) SD_MU=$2; shift 2;;
        --sd-temp) SD_TEMP=$2; shift 2;;
        --sd-div) SD_DIV=$2; shift 2;;
        --sd-top-p) SD_TOP_P=$2; shift 2;;
        --sd-skip-unparsed) SD_SKIP_UNPARSED=$2; shift 2;;
        --sd-warmup) SD_WARMUP=$2; shift 2;;
        --sd-omask) SD_OMASK=$2; shift 2;;
        --gpus) GPUS_ARG=$2; shift 2;;
        --extra) EXTRA_ARGS=$2; shift 2;;
        --task0-source-run) TASK0_SOURCE_RUN=$2; shift 2;;
        --train-num) TRAIN_NUM=$2; shift 2;;
        --dev-num) DEV_NUM=$2; shift 2;;
        --smoke-rows) SMOKE_ROWS=$2; shift 2;;
        --eval-bs) EVAL_BS=$2; shift 2;;
        --phys-bs) PHYS_BS=$2; shift 2;;
        --eval-gen-mode) EVAL_GEN_MODE=$2; shift 2;;
        --dynamic-pad) DYNAMIC_PAD=$2; shift 2;;
        --strict-gen) STRICT_GEN=$2; shift 2;;
        --resume) RESUME=1; shift;;
        *) echo "unknown flag $1"; exit 1;;
    esac
done
[ -z "${RUN_NAME}" ] && { echo "--run-name required"; exit 1; }
[ -z "${LR_LATER}" ] && LR_LATER=${LR}
source scripts/qwen/lib.sh
read -r PBS PACC <<< "$(phys_split "${BS}" "${ACC}" "${PHYS_BS:-${BS}}")"
[ "${SELECT_BEST}" = "1" ] && EVAL_GEN_MODE=every   # pick_best_ckpt.py reads the per-epoch dev F1

GPUS=(${GPUS_ARG})
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
ENV_BIN=${ENV_BIN:-$HOME/miniconda3/envs/mta/bin}   # override on a host with a differently named env
if [ -x "${ENV_BIN}/nvcc" ]; then
    export CUDA_HOME=$(dirname "${ENV_BIN}")
fi
export PATH=${ENV_BIN}:$PATH

BASE_PATH=.
# local copy from download.txt when present (see run.sh), else the hub name
[ -f models/Qwen3-0.6B/config.json ] && MODEL_PATH=${MODEL_PATH:-models/Qwen3-0.6B}
BASE_MODEL=${MODEL_PATH:-Qwen/Qwen3-0.6B}
GPUS_PER_NODE=${#GPUS[@]}

RUN_ROOT="${BASE_PATH}/results/qwen3/ced/${RUN_NAME}"
STREAMS_FILE="${BASE_PATH}/data/${DATA_PREFIX}${PERM}/streams.json"
if [ -z "${END_TASK}" ]; then
    # streams.json when present (build_ced_perms / cre_materialize_perm both write it),
    # else count the numeric task subdirs so notebook-built dirs still work.
    DATA_ROOT_DIR="${BASE_PATH}/data/${DATA_PREFIX}${PERM}"
    if [ -f "${STREAMS_FILE}" ]; then
        END_TASK=$(${ENV_BIN}/python -c "import json,sys; print(len(json.load(open(sys.argv[1])))-1)" "${STREAMS_FILE}")
        echo "END_TASK not given, derived ${END_TASK} from streams.json"
    else
        END_TASK=$(find "${DATA_ROOT_DIR}" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' | wc -l)
        END_TASK=$((END_TASK - 1))
        [ "${END_TASK}" -ge 0 ] || { echo "no task subdirs under ${DATA_ROOT_DIR}; pass --end-task"; exit 1; }
        echo "END_TASK not given, derived ${END_TASK} from task subdirs of ${DATA_ROOT_DIR}"
    fi
fi
if [ -d "${RUN_ROOT}" ] && [ -n "$(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ] \
   && [ "${RESUME}" = "0" ]; then
    echo "refusing to overwrite existing run: ${RUN_ROOT}"
    exit 1
fi
mkdir -p ${RUN_ROOT}
MANIFEST="${RUN_ROOT}/run_manifest.json"
MANIFEST_METHOD=${KD_TYPE}; [ "${MODE}" = "sft" ] && MANIFEST_METHOD=sft
MANIFEST_CONFIG="mode=${MODE};kd_ratio=${KD_RATIO};w_span=${W_SPAN};kd_type=${KD_TYPE};skew=${SKEW};span_metric=${SPAN_METRIC};layers=${LAYERS};pl=${PL};pl_dedup=${PL_DEDUP};pl_conf=${PL_CONF}/${PL_CONF_PCT}/${PL_CONF_THRESH};pl_lexicon=${PL_LEXICON};boost=${BOOST};kd_scope=${KD_SCOPE};balance=${BAL}/${BAL_PT}/${BAL_LR}/${BAL_DIST};select_best=${SELECT_BEST};kd_new=${KDNEW};sd=${SD}/${W_SD}/${SD_MU}/${SD_TEMP}/${SD_DIV}/top_p=${SD_TOP_P}/skip_unparsed=${SD_SKIP_UNPARSED}/warmup=${SD_WARMUP}/omask=${SD_OMASK};lr=${LR}/${LR_LATER};train_num=${TRAIN_NUM};dev_num=${DEV_NUM};smoke_rows=${SMOKE_ROWS};eval_bs=${EVAL_BS};phys=${PBS}x${PACC};eval_gen=${EVAL_GEN_MODE};dyn_pad=${DYNAMIC_PAD};strict_gen=${STRICT_GEN};extra=${EXTRA_ARGS}"
MANIFEST_ARGS=(
    init --output "${MANIFEST}" --run "${RUN_NAME}" --method "${MANIFEST_METHOD}"
    --permutation "${PERM}" --seed "${SEED}" --data-root "${BASE_PATH}/data/${DATA_PREFIX}${PERM}"
    --data-path "${BASE_PATH}/data/${DATA_PREFIX}${PERM}/streams.json"
    --data-path "${BASE_PATH}/processed_data/${DATA_PREFIX}${PERM}"
    --runtime-file "${BASE_PATH}/arguments.py"
    --runtime-file "${BASE_PATH}/finetune.py"
    --runtime-file "${BASE_PATH}/ced_finetune.py"
    --runtime-file "${BASE_PATH}/ced_omask.py"
    --runtime-file "${BASE_PATH}/ced_losses.py"
    --runtime-file "${BASE_PATH}/ced_step.py"
    --runtime-file "${BASE_PATH}/ced_eval.py"
    --runtime-file "${BASE_PATH}/gen_config.py"
    --runtime-file "${BASE_PATH}/scripts/qwen/lib.sh"
    --runtime-file "${BASE_PATH}/data_utils/lm_datasets.py"
    --runtime-file "${BASE_PATH}/distillm/buffer.py"
    --runtime-file "${BASE_PATH}/distillm/losses.py"
    --runtime-file "${BASE_PATH}/distillm/sampler.py"
    --runtime-file "${BASE_PATH}/utils.py"
    --runtime-file "${BASE_PATH}/ed_eval.py"
    --runtime-file "${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
    --runtime-file "${BASE_PATH}/scripts/qwen/ced/run_ced_v2.sh"
    --runtime-file "${BASE_PATH}/tools/ced_sd_prompts.py"
    --model "${BASE_MODEL}" --rank "${RANK}" --alpha "${ALPHA}" --dropout 0.1
    --gpu-count "${GPUS_PER_NODE}" --micro-batch "${BS}"
    --gradient-accumulation "${ACC}" --epochs "${EPOCHS}" --start-task "${START_TASK}"
    --config "${MANIFEST_CONFIG}"
)
[ "${GREEDY}" = "1" ] && MANIFEST_ARGS+=(--greedy)
[ -n "${TASK0_SOURCE_RUN}" ] && MANIFEST_ARGS+=(--parent-task0-run "${TASK0_SOURCE_RUN}")
[ "${RESUME}" = "1" ] && MANIFEST_ARGS+=(--resume)
${ENV_BIN}/python ${BASE_PATH}/tools/ced_run_manifest.py "${MANIFEST_ARGS[@]}"
if [ "${RESUME}" = "1" ]; then
    for STALE_TASK in $(seq "${START_TASK}" "${END_TASK}"); do
        rm -rf "${RUN_ROOT}/task${STALE_TASK}"
    done
fi
echo "run=${RUN_NAME} mode=${MODE} perm=${PERM} data=${DATA_PREFIX} pl=${PL} boost=${BOOST} kd_scope=${KD_SCOPE} bal=${BAL}/${BAL_PT}/${BAL_LR} kd_ratio=${KD_RATIO} w_span=${W_SPAN} ${KD_TYPE}/${SKEW}/${SPAN_METRIC} layers='${LAYERS}' bs=${BS}x${ACC} lr=${LR}/${LR_LATER} ep=${EPOCHS} seed=${SEED} lora=${RANK}/${ALPHA} greedy=${GREEDY} sd=${SD}/${W_SD}/${SD_MU}/${SD_TEMP}/${SD_DIV}/top_p=${SD_TOP_P}/skip_unparsed=${SD_SKIP_UNPARSED}/warmup=${SD_WARMUP}/omask=${SD_OMASK} task0_source=${TASK0_SOURCE_RUN:-none} phys=${PBS}x${PACC} eval_gen=${EVAL_GEN_MODE} dyn_pad=${DYNAMIC_PAD} strict_gen=${STRICT_GEN}" \
    | tee ${RUN_ROOT}/run_config.txt

tokenize () {  # $1=raw dir  $2=processed dir  [$3=teacher prompt cap, default 640]
    PYTHONPATH=${BASE_PATH} ${ENV_BIN}/python ${BASE_PATH}/tools/process_data.py \
        --data-dir $1/ --processed-data-dir $2 \
        --model-path ${BASE_MODEL} --data-process-workers 4 \
        --max-prompt-length 460 --t-max-prompt-length ${3:-640} \
        --dev-num 1000 --model-type qwen
}

train_once () {  # $1=engine $2=init $3=data_dir $4=save $5=lr $6=epochs $7=extra opts
    local MASTER_PORT=${MASTER_PORT:-66$(($RANDOM%90+10))}   # project_commands.sh pins one per GPU
    local OPTS=""
    OPTS+=" --base-path ${BASE_PATH} --model-path $2 --ckpt-name qwen3-0.6B --model-type qwen --n-gpu ${GPUS_PER_NODE}"
    OPTS+=" --data-dir $3 --num-workers 0 --train-num ${TRAIN_NUM} --dev-num ${DEV_NUM} --ced-smoke-rows ${SMOKE_ROWS}"
    OPTS+=" --lr $5 --batch-size ${PBS} --eval-batch-size ${EVAL_BS} --gradient-accumulation-steps ${PACC}"
    OPTS+=" --loss-group-size ${BS} --eval-gen-mode ${EVAL_GEN_MODE} --eval-loss-batch-size 32"
    [ "${DYNAMIC_PAD}" = "1" ] && OPTS+=" --dynamic-pad"
    [ "${STRICT_GEN}" = "1" ] && OPTS+=" --strict-generation"
    OPTS+=" --warmup-iters 0 --warmup-ratio 0.1 --lr-decay-style wrmup_cosine --weight-decay 1e-2 --clip-grad 1.0"
    OPTS+=" --epochs $6 --max-length 768 --max-prompt-length 460"
    OPTS+=" --do-train --do-valid --eval-gen --save-interval -1 --eval-interval -1 --log-interval 20 --mid-log-num -1"
    OPTS+=" --save $4 --seed ${SEED}"
    OPTS+=" --deepspeed --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
    OPTS+=" --top-k 0 --top-p 0.95 --temperature 0.5"
    [ "${GREEDY}" != "1" ] && OPTS+=" --do-sample"
    OPTS+=" --peft lora --peft-lora-r ${RANK} --peft-lora-alpha ${ALPHA} --peft-lora-dropout 0.1"
    OPTS+=" $7"
    mkdir -p $4
    ${ENV_BIN}/torchrun --nproc_per_node ${GPUS_PER_NODE} --nnodes 1 --node_rank 0 \
        --master_addr localhost --master_port ${MASTER_PORT} \
        ${BASE_PATH}/$1 ${OPTS} > $4/train.log 2>&1
}

last_adapter () {  # $1=dir
    find $1 -name adapter_config.json | xargs -r -n1 dirname | sort -V | tail -1
}

if [ -n "${TASK0_SOURCE_RUN}" ] && [ ${START_TASK} -eq 1 ]; then
    SOURCE_TASK0="${BASE_PATH}/results/qwen3/ced/${TASK0_SOURCE_RUN}/task0"
    INIT_MODEL="${SOURCE_TASK0}/merged"
    [ -d "${INIT_MODEL}" ] || { echo "missing shared task0 model: ${INIT_MODEL}"; exit 1; }
    mkdir -p "${RUN_ROOT}/task0"
    SOURCE_LOG=$(find "${SOURCE_TASK0}" -name log.txt | sort -V | tail -1)
    [ -n "${SOURCE_LOG}" ] || { echo "missing shared task0 log under ${SOURCE_TASK0}"; exit 1; }
    cp "${SOURCE_LOG}" "${RUN_ROOT}/task0/log.txt"
    echo "${TASK0_SOURCE_RUN}" > "${RUN_ROOT}/task0/source_run.txt"
elif [ -n "${TASK0_SOURCE_RUN}" ] && [ ${START_TASK} -lt 1 ]; then
    {
        echo "--task0-source-run requires --start-task >= 1"
        exit 1
    }
elif [ ${START_TASK} -eq 0 ]; then
    INIT_MODEL=${BASE_MODEL}
else
    INIT_MODEL="${RUN_ROOT}/task$((START_TASK-1))/merged"
fi

for T in $(seq ${START_TASK} ${END_TASK})
do
    RAW_DIR="${BASE_PATH}/data/${DATA_PREFIX}${PERM}/${T}"
    DATA_DIR="${BASE_PATH}/processed_data/${DATA_PREFIX}${PERM}/${T}/qwen/"
    SAVE_PATH="${RUN_ROOT}/task${T}"
    TASK_LR=${LR}; [ ${T} -gt 0 ] && TASK_LR=${LR_LATER}
    STAGE=${RAW_DIR}

    if [ ${T} -gt 0 ]; then
        if [ "${PL}" = "1" ]; then
            PL_DIR="${BASE_PATH}/data/stage_${RUN_NAME}/${T}_pl"
            echo "===== ${RUN_NAME} task${T}: pseudo-labeling (teacher=${INIT_MODEL}) ====="
            # keeping generate() scores for the confidence filter costs ~6GB VRAM
            # -> halve the generation batch when the filter is on
            PL_BS=64; [ "${PL_CONF}" != "none" ] && PL_BS=32
            PL_OPTS="--conflict-dedup ${PL_DEDUP} --conf-filter ${PL_CONF} --conf-percentile ${PL_CONF_PCT}"
            PL_OPTS+=" --lexicon-filter ${PL_LEXICON}"
            [ -n "${PL_CONF_THRESH}" ] && PL_OPTS+=" --conf-thresh ${PL_CONF_THRESH}"
            ${ENV_BIN}/python ${BASE_PATH}/tools/ced_pseudo_label.py \
                --teacher ${INIT_MODEL} --data-dir ${STAGE} \
                --streams ${STREAMS_FILE} --task-id ${T} \
                --batch-size ${PL_BS} ${PL_OPTS} \
                --out ${PL_DIR} > ${RUN_ROOT}/pl_task${T}.log 2>&1
            STAGE=${PL_DIR}
        fi
        if [ "${BOOST}" -gt 1 ]; then
            BO_DIR="${BASE_PATH}/data/stage_${RUN_NAME}/${T}_boost"
            echo "===== ${RUN_NAME} task${T}: replay oversample x${BOOST} ====="
            ${ENV_BIN}/python ${BASE_PATH}/tools/ced_oversample.py \
                --data-dir ${STAGE} --streams ${STREAMS_FILE} --task-id ${T} \
                --boost ${BOOST} --out ${BO_DIR} >> ${RUN_ROOT}/pl_task${T}.log 2>&1
            STAGE=${BO_DIR}
        fi
        T_CAP=640
        if [ "${SD}" = "1" ]; then
            # last data step: the teacher reference must be the final gold+PL answer
            SD_DIR="${BASE_PATH}/data/stage_${RUN_NAME}/${T}_sd"
            echo "===== ${RUN_NAME} task${T}: self-distillation teacher prompts ====="
            ${ENV_BIN}/python ${BASE_PATH}/tools/ced_sd_prompts.py \
                --data-dir ${STAGE} --out ${SD_DIR} >> ${RUN_ROOT}/pl_task${T}.log 2>&1
            STAGE=${SD_DIR}
            # prompt + template + reference answer. The gold-only teacher prompt peaks at 676
            # tokens over all CED train rows (0 above 800), but the reference here is gold+PL,
            # which pseudo-labels make longer and nobody has measured. [:max] truncation drops
            # the tail, i.e. the <|im_start|>assistant suffix, so give it room: sd_probe counts
            # any row that still ends up cut.
            T_CAP=1000
        fi
        if [ "${STAGE}" != "${RAW_DIR}" ]; then
            PROC="${BASE_PATH}/processed_data/stage_${RUN_NAME}/${T}"
            tokenize ${STAGE} ${PROC} ${T_CAP} >> ${RUN_ROOT}/pl_task${T}.log 2>&1
            DATA_DIR="${PROC}/qwen/"
        fi
    fi

    EXTRA=""
    ENGINE=finetune.py
    if [ "${MODE}" != "sft" ] && [ ${T} -gt 0 ]; then
        ENGINE=ced_finetune.py
        EXTRA+=" --type ${KD_TYPE} --skew-alpha ${SKEW} --kd-ratio ${KD_RATIO}"
        EXTRA+=" --teacher-model-path ${INIT_MODEL} --teacher-ckpt-name qwen3-0.6B-prev --teacher-model-fp16"
        EXTRA+=" --ced-streams-file ${STREAMS_FILE} --ced-task-id ${T} --ced-replay-mode ${MODE}"
        EXTRA+=" --ced-kd-scope ${KD_SCOPE}"
        EXTRA+=" --ced-kd-ratio-new ${KDNEW}"
        EXTRA+=" --teacher_layer_mapping ${LAYERS} --student_layer_mapping ${LAYERS}"
        EXTRA+=" --w-span-loss ${W_SPAN} --span_metric ${SPAN_METRIC}"
        if [ "${SD}" = "1" ]; then
            EXTRA+=" --ced-sd --ced-sd-weight ${W_SD} --ced-sd-ema-mu ${SD_MU}"
            EXTRA+=" --ced-sd-temperature ${SD_TEMP} --ced-sd-div ${SD_DIV}"
            EXTRA+=" --ced-sd-top-p ${SD_TOP_P} --ced-sd-warmup ${SD_WARMUP}"
            if [ "${SD_SKIP_UNPARSED}" = "1" ]; then
                EXTRA+=" --ced-sd-skip-unparsed"
            fi
            if [ "${SD_OMASK}" = "1" ]; then
                EXTRA+=" --ced-sd-omission-mask"
            fi
        fi
        EXTRA+=" ${EXTRA_ARGS}"
    else
        EXTRA+=" --type lm"
    fi

    echo "===== ${RUN_NAME} task${T}: train engine=${ENGINE} init=${INIT_MODEL} lr=${TASK_LR} data=${DATA_DIR} ====="
    train_once ${ENGINE} ${INIT_MODEL} ${DATA_DIR} ${SAVE_PATH} ${TASK_LR} ${EPOCHS} "${EXTRA}"

    if [ "${SELECT_BEST}" = "1" ]; then
        LAST_CKPT=$(${ENV_BIN}/python ${BASE_PATH}/tools/pick_best_ckpt.py --save-dir ${SAVE_PATH} 2>> ${RUN_ROOT}/pick_task${T}.log) || LAST_CKPT=""
        [ -z "${LAST_CKPT}" ] && LAST_CKPT=$(last_adapter ${SAVE_PATH})   # fallback: never abort a run on a picker hiccup
        echo "===== ${RUN_NAME} task${T}: best-dev ckpt = ${LAST_CKPT} (see pick_task${T}.log) ====="
    else
        LAST_CKPT=$(last_adapter ${SAVE_PATH})
    fi
    [ -z "${LAST_CKPT}" ] && { echo "no adapter checkpoint under ${SAVE_PATH}, aborting"; exit 1; }

    if [ "${BAL}" = "1" ] && [ ${T} -gt 0 ]; then
        POOL_RAW="${BASE_PATH}/data/stage_${RUN_NAME}/${T}_pool"
        POOL_PROC="${BASE_PATH}/processed_data/stage_${RUN_NAME}/${T}_pool"
        echo "===== ${RUN_NAME} task${T}: balanced calibration epoch ====="
        ${ENV_BIN}/python ${BASE_PATH}/tools/ced_balance_pool.py \
            --data-dir ${STAGE} --streams ${STREAMS_FILE} --task-id ${T} \
            --per-type ${BAL_PT} --dist ${BAL_DIST} --seed ${SEED} \
            --out ${POOL_RAW} > ${RUN_ROOT}/bal_task${T}.log 2>&1
        tokenize ${POOL_RAW} ${POOL_PROC} >> ${RUN_ROOT}/bal_task${T}.log 2>&1
        BAL_SAVE="${SAVE_PATH}_bal"
        train_once finetune.py ${INIT_MODEL} ${POOL_PROC}/qwen/ ${BAL_SAVE} ${BAL_LR} 1 \
            " --type lm --peft-path ${LAST_CKPT}"
        BAL_CKPT=$(last_adapter ${BAL_SAVE})
        [ -n "${BAL_CKPT}" ] && LAST_CKPT=${BAL_CKPT}
    fi

    echo "===== ${RUN_NAME} task${T}: merging ${LAST_CKPT} ====="
    ${ENV_BIN}/python ${BASE_PATH}/tools/merge_lora.py \
        --base-model-path ${INIT_MODEL} --peft-path ${LAST_CKPT} \
        --out ${SAVE_PATH}/merged > ${SAVE_PATH}/merge.log 2>&1

    ${ENV_BIN}/python ${BASE_PATH}/tools/ced_run_manifest.py update \
        --output "${MANIFEST}" --completed-task "${T}" --status running
    if [ ${T} -gt 0 ]; then
        rm -rf "${RUN_ROOT}/task$((T-1))/merged"
    fi
    INIT_MODEL="${SAVE_PATH}/merged"
done

${ENV_BIN}/python ${BASE_PATH}/tools/ced_run_manifest.py update \
    --output "${MANIFEST}" --completed-task "${END_TASK}" --status complete
echo "done" > "${RUN_ROOT}/.complete"
echo "CED SEQUENCE DONE ${RUN_NAME}"
