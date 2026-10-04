#!/usr/bin/env bash
# OUR method and its ablations, x 5 permutations, on this host.
# Round 3 (03/10) is what runs by default: on ACE the main-table run plus the ablations the
# paper tables need; on any other dataset only the main-table run. Rounds 1 and 2 stay in
# CONFIGS_ALL, commented out: uncomment a line and add its group to ONLY to re-run it.
#
#   bash project_commands.sh                            # ACE: g1_full, omission-mask arm, ablations
#   DS=maven bash project_commands.sh                   # main table on MAVEN (also rams, geneva)
#   DS=fewrel bash project_commands.sh                  # main table on FewRel (also tacred)
#   DRY=1 bash project_commands.sh                      # print the plan, train nothing
#   ONLY="c" bash project_commands.sh                   # only some groups
#
# Defaults match this host: venv, GPUs 4-7, one run per GPU, effective batch 2x16 = 32.
# Safe to re-run after a crash: a run with a .complete marker is skipped, and nothing is deleted.
#
# Effective batch stays 32 so every number is comparable with the f12_pl baseline. Per-device 32
# is not an option: the KD loss materialises batch x seq x vocab in fp32 (32 x 768 x 151936 x 4B
# = 14.9 GB per tensor) and OOMs a 46 GB card -- the same reason evaluate_loss() caps eval batch.
#
# Knobs (all optional):
#   VENV          venv to activate (default: /mnt/local/uvenvs/opened when none is active)
#   PY / ENV_BIN  interpreter and env bin/ for the runners (default: derived from `python`)
#   SKIP_INSTALL  1 = never touch dependencies
#   POOL_GPUS     one slot per GPU id (default "4 5 6 7")
#   PERMS         default "0 1 2 3 4"
#   DS            dataset, default ace: ace maven rams geneva (CED), tacred fewrel (CRE)
#   DATA_PREFIX   default <ds>_b10_perm (CED) or <ds>_perm (CRE), the names under data/
#   ONLY          groups to run, default "g1 m a c d l" on ACE and "g1" elsewhere
#                 (bash owns $GROUPS, hence ONLY)
set -euo pipefail
cd "$(dirname "$0")"

step () { echo; echo "=== $* ==="; }
have () { [ -e "$1" ]; }
DRY=${DRY:-0}


VENV=.venv

# ---------------------------------------------------------------- 1. environment
step "1. environment"
if [ -z "${VENV:-}" ] && [ -z "${VIRTUAL_ENV:-}" ] && [ -f /mnt/local/uvenvs/opened/bin/activate ]; then
    VENV=/mnt/local/uvenvs/opened
fi
if [ -n "${VENV:-}" ]; then
    # shellcheck disable=SC1091
    set +u; source "${VENV}/bin/activate"; set -u   # activate scripts read unset vars
    echo "activated ${VENV}"
else
    echo "no VENV given, using the current environment"
fi
PY=${PY:-$(command -v python || command -v python3)}
[ -x "${PY}" ] || { echo "no python found; set PY or activate an env"; exit 1; }
ENV_BIN=${ENV_BIN:-$(dirname "${PY}")}
export PY ENV_BIN
echo "PY=${PY}"; echo "ENV_BIN=${ENV_BIN}"; "${PY}" -V

if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    echo "SKIP_INSTALL=1, not touching dependencies"
elif "${PY}" -c "import torch, transformers, peft" 2>/dev/null; then
    echo "torch/transformers/peft already importable, skipping install"
else
    req=opened.txt; [ -f "${req}" ] || req=requirements.txt
    echo "installing from ${req}"
    "${PY}" -m pip install -r "${req}"
fi

# The pod exports PET_RDZV_BACKEND / PET_RDZV_ENDPOINT / PET_RDZV_ID, which torchrun reads as
# flag defaults. c10d ignores --master_port, so every torchrun would join the same rendezvous:
# the first runs, the rest die with RendezvousConnectionError when it exits.
for v in $(compgen -e | grep '^PET_' || true); do unset "${v}"; done
# building ace_b10_perm{1..4} reads datht/ace-short-generated-dataset, which is private
if [ -f .env ] && [ -z "${HF_TOKEN:-}" ]; then
    set -a; . ./.env; set +a
    echo "read HF_TOKEN from .env"
fi
if [ -f models/Qwen3-0.6B/config.json ]; then
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
fi

# ---------------------------------------------------------------- 2. data
step "2. data"
# Unzip what download.txt fetched: one archive per corpus (<ds>_all.tar.gz from
# datht/processed-cl-<ds>, ds = ace maven rams geneva tacred fewrel) instead of hundreds of
# loose files. Each unpacks to data/ and processed_data/, which is where everything below
# looks, and is deleted once unpacked.
# Python's tarfile, not the tar binary: the image is minimal (see the en_core_web_sm note).
for tgz in *_all.tar.gz; do
    have "${tgz}" || continue
    echo "  unpacking ${tgz}"
    if [ "${DRY}" != "1" ]; then
        "${PY}" -c "import sys, tarfile; tarfile.open(sys.argv[1]).extractall('.', filter='data')" "${tgz}"
        rm -f "${tgz}"
    fi
done
PERMS=${PERMS:-"0 1 2 3 4"}
DS=${DS:-ace}
case ${DS} in
    tacred|fewrel) DATA_PREFIX=${DATA_PREFIX:-${DS}_perm} ;;      # CRE splits ship in data/, built already
    *)             DATA_PREFIX=${DATA_PREFIX:-${DS}_b10_perm} ;;
esac

# The matrix is defined here, ahead of section 3, because some configs read their own data
# (--data-prefix in their flags) and this step has to build it.
# name | sd | runner flags | SD_ARGS       (sd = cl: a CL-LoRA baseline, run by run_cllora.sh)
CONFIGS_ALL=(
  # ---- round 3 (03/10): the runs the paper tables need
  #   g1  g1_full is the method: the "Ours" row of every main table, and the full-method row
  #       of the ablations. Same name as round 1, so an ACE run that already finished is reused
  #   a   components table: the method minus SD
  #   c   components table: the method minus PL (the SD teacher then reads y_t), and minus
  #       token KD (span KD, CE and SD stay)
  #   m   the method with the omission-mask on (--sd-omask 1); g1_full is the same run with it off
  #   d   rehearsal-free table, Ours row: the method with memory 0 (g1_full is Ours with memory 10).
  #       KD scope pl, else it has almost no KD rows
  #   l   rehearsal-free table: TreeLoRA / IncLoRA / O-LoRA with memory 0. Their numbers in the
  #       main table trained on <ds>_b10_perm, i.e. with the 10-per-type buffer in train.jsonl
  "g1_full|1||"
  "a_nosd|0||"
  "c_nopl|1|--pl 0|"
  "c_notkd|1|--kd-type no|"
  "m_omask|1||--sd-omask 1"
  "d_full|1|--data-prefix ${DS}_b0_perm --kd-scope pl|"
  "l_tree|cl|--data-prefix ${DS}_b0_perm|"
  "l_inclora|cl|--data-prefix ${DS}_b0_perm|"
  "l_olora|cl|--data-prefix ${DS}_b0_perm|"

  # ---- round 2 (01/10). Not used by any paper table any more; kept so it can be re-run.
  #   a_rand  KD/SD drawn at random per update (professor note 7b; --ced-sd-mix)
  #   d_ce    CE with memory 0 (the rehearsal table now compares with CL-LoRA instead)
  #   o_ce    oracle: CE on D_t with old-type events kept (no label stripping)
  #   x_*     SD teacher = current model (mu 0), PL without lexicon
  # "a_rand|1||--extra --ced-sd-mix=random"
  # "d_ce|0|--mode sft --pl 0 --data-prefix ${DS}_b0_perm|"
  # "o_ce|0|--mode sft --pl 0 --data-prefix ${DS}_oracle_b10_perm|"
  # "x_mu0|1||--sd-mu 0"
  # "x_nolex|0|--mode sft --pl 1 --pl-lexicon 0|"

  # ---- round 1 (started 01/10). Commented out, not deleted, so it can be re-run after a crash:
  # a run with a .complete marker is skipped, so only the unfinished ones train again.
  #   g1  does each component earn its place; g1_full is the method
  #   g2  does the pseudo-label filter earn its place (grounding + lexicon have no published
  #       precedent for LLM-generated spans, so this is where PL's novelty lives)
  #   g3  SD weight: at 1.0 the SD gradient measured ~1.2x the rest of the objective combined
  #   g4  span-loss metric; g1_full is cosine
  #   g5  SD sampling: SDFT's own T=1.0/top_p=1.0 is what every other SD arm uses, and the
  #       earlier notebook measured that exact setting at -8.13 F1, so these are the insurance
  # "g1_ce|0|--mode sft --pl 0|"
  # "g1_pl|0|--mode sft --pl 1|"
  # "g1_kd|0|--mode ce_kd --pl 0 --w-span 0|"
  # "g1_span|0|--mode ce_kd --pl 0 --kd-type no --w-span 2.0|"
  # "g1_sd|1|--mode ce_kd --pl 0 --kd-ratio 0 --w-span 0|"
  #  g1_full: active in round 3 above
  # "g2_nofilter|0|--mode sft --pl 1 --pl-dedup 0 --pl-conf none --pl-lexicon 0|"
  # "g2_ground|0|--mode sft --pl 1 --pl-conf none --pl-lexicon 0|"
  # "g3_wsd01|1||--w-sd 0.1"
  # "g3_wsd03|1||--w-sd 0.3"
  # "g3_wsd30|1||--w-sd 3.0"
  # "g4_nospan|1|--w-span 0|"
  # "g4_l2|1|--span-metric l2|"
  # "g4_cka|1|--span-metric cka|"
  # "g5_t07|1||--sd-temp 0.7 --sd-top-p 0.9"
  # "g5_warm|1||--sd-warmup 0.5"
  # "g5_rkl|1||--sd-div rkl"
)
if [ "${DS}" = "ace" ]; then ONLY=${ONLY:-"g1 m a c d l"}; else ONLY=${ONLY:-"g1"}; fi

CONFIGS=()
for c in "${CONFIGS_ALL[@]}"; do
    g=${c%%_*}
    for want in ${ONLY}; do [ "${g}" = "${want}" ] && CONFIGS+=("${c}") && break; done
done
[ ${#CONFIGS[@]} -gt 0 ] || { echo "no configs selected by ONLY='${ONLY}'"; exit 1; }

# the base data, plus every --data-prefix a selected config asks for
DATA_PREFIXES=("${DATA_PREFIX}")
for c in "${CONFIGS[@]}"; do
    IFS='|' read -r _ _ flags _ <<< "${c}"
    read -ra words <<< "${flags}"
    for i in "${!words[@]}"; do
        [ "${words[i]}" = "--data-prefix" ] || continue
        [[ " ${DATA_PREFIXES[*]} " == *" ${words[i+1]} "* ]] || DATA_PREFIXES+=("${words[i+1]}")
    done
done
build_flags () {  # data prefix -> tools/build_ced_perms.py flags
    case $1 in
        *_oracle_b10_perm) echo "--cap 10 --oracle" ;;
        *_b0_perm)         echo "--cap 0" ;;
        *_b10_perm)        echo "--cap 10" ;;
        *) return 1 ;;
    esac
}

# Raw ACE task splits are kept in the sibling open-ed checkout on this workspace.
# Override ACE_SOURCE_DIR when running elsewhere; it must contain 0..4/{train,dev,test}.jsonl.
ACE_SOURCE_DIR=${ACE_SOURCE_DIR:-../../open-ed/data/ace}
ace_task_splits_present () {
    local root=$1 p split
    for p in 0 1 2 3 4; do
        for split in train dev test; do
            [ -f "${root}/${p}/${split}.jsonl" ] || return 1
        done
    done
    return 0
}

# Copy only missing raw split files. Keep the source checkout intact and avoid overwriting
# any data already present in this OpenED tree.
ACE_RAW_READY=0
NEED_ACE_BUILD=0
for prefix in "${DATA_PREFIXES[@]}"; do
    for p in ${PERMS}; do
        have "data/${prefix}${p}/streams.json" || NEED_ACE_BUILD=1
    done
done
if [ "${DS}" = "ace" ] && [ "${NEED_ACE_BUILD}" = "1" ]; then
    if ace_task_splits_present data/ace; then
        ACE_RAW_READY=1
    elif ace_task_splits_present "${ACE_SOURCE_DIR}"; then
        if [ "${DRY}" = "1" ]; then
            echo "  DRY=1: would copy raw ACE task splits from ${ACE_SOURCE_DIR} to data/ace"
        else
            echo "  copying raw ACE task splits from ${ACE_SOURCE_DIR} to data/ace"
            mkdir -p data/ace
            for p in 0 1 2 3 4; do
                mkdir -p "data/ace/${p}"
                for split in train dev test; do
                    cp -an "${ACE_SOURCE_DIR}/${p}/${split}.jsonl" "data/ace/${p}/"
                done
            done
        fi
        ACE_RAW_READY=1
    fi
fi

for prefix in "${DATA_PREFIXES[@]}"; do
    for p in ${PERMS}; do
        have "data/${prefix}${p}/streams.json" && { printf '  %-40s ok\n' "data/${prefix}${p}"; continue; }
        bflags=$(build_flags "${prefix}") || bflags=""
        if [ "${DS}" = "ace" ] && [ "${ACE_RAW_READY}" = "1" ] && [ -n "${bflags}" ]; then
            # perm0 of cl-ace IS data/ace; the other four are re-split from the source corpus,
            # which is a private HF dataset, so this step needs HF_TOKEN and network.
            echo "  building data/${prefix}${p} from data/ace (${bflags})"
            [ "${DRY}" = "1" ] || OPENED_BASE=$(pwd) HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
                "${PY}" tools/build_ced_perms.py \
                ${bflags} --perms "${p}" --out-prefix "${prefix}" || {
                echo "  build failed for ${prefix}${p} (needs HF_TOKEN for datht/ace-short-generated-dataset)"
                exit 1; }
        else
            echo "  data/${prefix}${p} MISSING, and only ACE splits can be built here."
            echo "  Set ACE_SOURCE_DIR to a folder with {0..4}/{train,dev,test}.jsonl, or copy"
            echo "  data/${prefix}{0..4} from a host that already has the generated splits."
            exit 1
        fi
    done
done
# base task data is never tokenized by run_ced_v2.sh -- it only tokenizes PL/SD side-data.
# Only ${DATA_PREFIX} needs it: every config starts from the shared task0 run, whose task0 data
# is the same in every prefix, and ours_queue's --replay-boost re-tokenizes tasks 1+.
for p in ${PERMS}; do
    n_tasks=5  # TACRED / FewRel have 10; streams.json says how many
    have "data/${DATA_PREFIX}${p}/streams.json" && n_tasks=$("${PY}" -c \
        "import json,sys; print(len(json.load(open(sys.argv[1]))))" "data/${DATA_PREFIX}${p}/streams.json")
    for t in $(seq 0 $((n_tasks - 1))); do
        out="processed_data/${DATA_PREFIX}${p}/${t}"
        have "${out}/qwen/train_0.idx" && continue
        echo "  tokenising perm${p} task${t}"
        [ "${DRY}" = "1" ] && continue
        PYTHONPATH=. "${PY}" tools/process_data.py \
            --data-dir "data/${DATA_PREFIX}${p}/${t}/" --processed-data-dir "${out}" \
            --model-path "${MODEL_PATH:-Qwen/Qwen3-0.6B}" --data-process-workers 4 \
            --max-prompt-length 460 --t-max-prompt-length 640 \
            --dev-num 1000 --model-type qwen > "logs/tok_${DS}_p${p}t${t}.log" 2>&1 || {
            echo "  tokenize FAILED perm${p} task${t}, see logs/tok_${DS}_p${p}t${t}.log"; exit 1; }
    done
done

# ---------------------------------------------------------------- 3. the matrix
# (CONFIGS_ALL / ONLY / CONFIGS are defined in section 2)
SEED=${SEED:-42}
PROTOCOL=${PROTOCOL:-${DS}_v2}
VARIANT=${VARIANT:-h12}          # PL with dedup + lexicon + confidence, no H3 calibration epoch
R=results/qwen3/ced

run_dir () {  # $1=config name  $2=sd  $3=perm
    if [ "$1" = "task0" ]; then echo "${R}/dist_shared_task0_perm$3_${PROTOCOL}_s${SEED}"; return; fi
    # memory-0 CL-LoRA: its own protocol tag, so it never collides with the buffer-10 runs
    if [ "$2" = "cl" ]; then echo "${R}/cllora_${1#l_}_perm$3_${DS}_b0_v2_s${SEED}"; return; fi
    local tag=""; [ "$2" = "1" ] && tag="_sd"
    echo "${R}/ours_${VARIANT}${tag}_$1_perm$3_${PROTOCOL}_s${SEED}"
}

JOBS=()
for p in ${PERMS}; do JOBS+=("task0|0|||${p}"); done  # everything else waits on these; 5 fields like the rest
for c in "${CONFIGS[@]}"; do
    for p in ${PERMS}; do JOBS+=("${c}|${p}"); done
done

GPUS=(${POOL_GPUS:-4 5 6 7})
step "3. train ${#JOBS[@]} jobs on gpus ${GPUS[*]}"
mkdir -p logs
POOL_LOG=logs/${DS}_matrix_pool.log
echo "progress: ${POOL_LOG}   per-run logs: logs_ours_*.log and ${R}/<run>/task*/train.log"
log () { echo "[pool $(date '+%F %T')] $*" | tee -a "${POOL_LOG}"; }

launch () {  # $1=job $2=gpu -> starts it in the background
    local name sd flags sd_args perm
    IFS='|' read -r name sd flags sd_args perm <<< "$1"
    # DRY still goes through pick/run_dir/T0 with real job strings: the field-count bug that
    # broke T0[${perm}] only showed up once the scheduler ran, so it has to run here too.
    if [ "${DRY}" = "1" ]; then ( : ) & return; fi
    if [ "${name}" = "task0" ]; then
        # one task only, so a half-trained one is restarted rather than resumed
        rm -rf "$(run_dir task0 0 "${perm}")"
        MASTER_PORT=$((29500 + $2)) bash scripts/qwen/ced/run_ced_v2.sh \
            --run-name "$(basename "$(run_dir task0 0 "${perm}")")" --mode sft \
            --data-prefix "${DATA_PREFIX}" --perm "${perm}" \
            --rank 16 --alpha 64 --epochs 5 --lr 0.0002 --seed "${SEED}" \
            --bs 2 --acc 16 --greedy 1 --gpus "$2" --end-task 0
    elif [ "${sd}" = "cl" ]; then
        bash scripts/qwen/ced/run_cllora.sh --method "${name#l_}" --data-root "data/${DS}_b0_perm${perm}" \
            --protocol "${DS}_b0_v2" --seed "${SEED}" --gpu "$2" --py "${PY}"
    else
        PERM="${perm}" GPU="$2" DATA_PREFIX="${DATA_PREFIX}" SEED="${SEED}" PROTOCOL="${PROTOCOL}" \
        OURS_VARIANT="${VARIANT}" OURS_SD="${sd}" SD_ARGS="${sd_args}" RUN_SUFFIX="_${name}" \
        RESUME=0 MASTER_PORT=$((29500 + $2)) \
            bash scripts/qwen/ced/ours_queue.sh ${flags}
    fi >> "${POOL_LOG}" 2>&1 &
}

# task0 state per perm: done | pending | running | failed
declare -A T0
for p in ${PERMS}; do
    [ -f "$(run_dir task0 0 "${p}")/.complete" ] && T0[${p}]=done || T0[${p}]=pending
done

pick () {  # sets JOB to the first startable job and drops it from PENDING; 1 if none
    local i job name sd flags sd_args perm
    for i in "${!PENDING[@]}"; do
        job=${PENDING[i]}
        IFS='|' read -r name sd flags sd_args perm <<< "${job}"
        if [ -f "$(run_dir "${name}" "${sd}" "${perm}")/.complete" ]; then
            log "skip   ${name}/perm${perm} (already complete)"
            unset 'PENDING[i]'; continue
        fi
        if [ "${name}" != "task0" ] && [ "${sd}" != "cl" ]; then  # CL-LoRA trains its own task0
            case ${T0[${perm}]} in
                pending|running) continue ;;
                failed) log "FAILED ${name}/perm${perm} (task0 of perm${perm} failed)"
                        n_fail=$((n_fail + 1)); unset 'PENDING[i]'; continue ;;
            esac
        fi
        JOB=${job}; unset 'PENDING[i]'
        [ "${name}" = "task0" ] && T0[${perm}]=running
        return 0
    done
    return 1
}

if [ "${DRY}" = "1" ]; then
    for job in "${JOBS[@]}"; do
        IFS='|' read -r name sd flags sd_args perm <<< "${job}"
        printf '  %-14s perm%s  sd=%s  %s %s\n' "${name}" "${perm}" "${sd}" "${flags}" "${sd_args}"
    done
    echo "DRY=1: no training, but the scheduler below still runs against stub jobs."
fi

PENDING=("${JOBS[@]}")
declare -a SLOT_PID SLOT_JOB
n_fail=0
while :; do
    for i in "${!GPUS[@]}"; do
        pid=${SLOT_PID[i]:-}
        if [ -n "${pid}" ]; then
            kill -0 "${pid}" 2>/dev/null && continue
            rc=0; wait "${pid}" || rc=$?
            job=${SLOT_JOB[i]}; IFS='|' read -r name sd flags sd_args perm <<< "${job}"
            if [ "${rc}" -eq 0 ] && { [ "${DRY}" = "1" ] || [ -f "$(run_dir "${name}" "${sd}" "${perm}")/.complete" ]; }; then
                log "done   ${name}/perm${perm} (gpu${GPUS[i]})"
                [ "${name}" = "task0" ] && T0[${perm}]=done
            else
                log "FAILED ${name}/perm${perm} (gpu${GPUS[i]}, exit ${rc}), see ${POOL_LOG}"
                n_fail=$((n_fail + 1))
                [ "${name}" = "task0" ] && T0[${perm}]=failed
            fi
            SLOT_PID[i]=""
        fi
        pick || continue
        launch "${JOB}" "${GPUS[i]}"
        SLOT_PID[i]=$!; SLOT_JOB[i]=${JOB}
        IFS='|' read -r name sd flags sd_args perm <<< "${JOB}"
        log "start  ${name}/perm${perm} (gpu${GPUS[i]})"
    done
    busy=0
    for pid in ${SLOT_PID[@]+"${SLOT_PID[@]}"}; do [ -n "${pid}" ] && busy=1; done
    if [ "${busy}" = "0" ]; then
        for job in ${PENDING[@]+"${PENDING[@]}"}; do log "FAILED ${job} (never startable)"; n_fail=$((n_fail + 1)); done
        break
    fi
    sleep 20
done

step "4. collect F1 files"
# Before the failure exit below, so the runs that did finish are collected either way.
# The label carries the dataset and time: gather_logs.sh refuses to reuse a folder.
if [ "${DRY}" = "1" ]; then
    echo "DRY=1: would run gather_logs.sh ${DS}_$(date +%Y%m%d_%H%M)"
else
    bash gather_logs.sh "${DS}_$(date +%Y%m%d_%H%M)" || echo "gather_logs.sh failed, run it by hand"
fi

step "5. done"
if [ "${n_fail}" -gt 0 ]; then
    echo "${n_fail} jobs failed: grep FAILED ${POOL_LOG}"
    exit 1
fi
echo "all jobs finished, F1 files are under collected_logs/"
