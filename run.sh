#!/bin/bash
# Orchestrates the baseline sweep for one dataset, either family:
#   CRE  (continual relation extraction): tacred, fewrel
#   CED  (continual event detection):     maven, rams, geneva  (ace runs the same way too)
# Tokenizes each requested permutation, then launches the distillation queue (7 methods)
# and the CL-LoRA queue (8 methods) each on its own GPU set, in the background.
#
# Rule enforced by the per-family runners: at most ONE queue per GPU. This script respects
# that by giving the distillation queue and the CL-LoRA queue separate GPUs and running each
# one's permutations sequentially inside a single background process, so a GPU never has two
# queues racing for its memory between tasks.
#
# Each family gets a comma-separated GPU list (GPU_DIST_ALL / GPU_CLLORA_ALL below). With
# more than one GPU the methods are split round-robin, one single-GPU sub-queue per GPU, so
# every run keeps its original batch size and stays comparable with the single-GPU runs.
# CED dist trains the shared task0 on the first GPU before splitting. CRE ignores all but
# the first GPU.
#
# Usage:
#   bash run.sh                                                     # run everything (see below)
#   bash run.sh <tacred|fewrel|maven|rams|geneva> ["perms"] [gpus_dist] [gpus_cllora] [queue]
#
#   bash run.sh rams                       # perms 0-4, dist on gpu0-3, CL-LoRA on gpu4-7, both queues
#   bash run.sh geneva "0 1 2"             # only perm0-2
#   bash run.sh maven "0" 0 1              # single perm, explicit GPUs (one each)
#   bash run.sh rams "3 4" 0 1 cllora      # catch up missing perms on ONE queue only
#   bash run.sh geneva "" 0 1 prep         # tokenize only, train nothing
#
# No-argument mode (`bash run.sh`): every baseline run still missing, in the order
# maven, rams -- see MISSING_PLAN below for which perms and which
# queue each dataset needs and why. Every runner skips its own finished work (a completion
# marker per method+perm), so a dataset whose range is listed in full only trains the gaps,
# and the plan stays correct as runs land. Datasets run ONE AT A TIME (dist+CL-LoRA
# concurrently within a dataset, next dataset only after both queues of the current one
# finish) so at most one queue ever sits on a given GPU. Override the whole thing with
# MISSING_PLAN="ds:perms:queue;..." or, for a plain list at all perms and both queues,
# RUN_ALL_DATASETS="ds1 ds2 ...".
#
# ON A FRESH CLONE (new server): `results/` is gitignored, so none of the completion markers
# come with the repo and NOTHING is skipped -- `bash run.sh` would retrain all 15 baselines on
# all five datasets, roughly 3500 single-task trainings, including everything already finished
# elsewhere. Pick one before launching:
#   - copy the finished runs over first (results/qwen3/ced/*/ -- the .complete markers and
#     cl_results.json / taskN/log.txt are what the skips read), then run normally; or
#   - narrow the plan to what this host should own, e.g.
#       MISSING_PLAN="fewrel:0 1 2 3 4:both" bash run.sh
# `data/` IS in git, so the perm splits need no rebuild; only tokenization runs per host.
#
# NOT covered by either queue, because no runner script exists for them: LwF (ACE perm3-4),
# SeqLoRA-merge (ACE perm1-4) and f12_pl / ours (MAVEN perm1-4, TACRED, FewRel). LwF is a
# flag on run_ced_v2.sh (--kd-new) and ours is run_ced_v2.sh --mode ce_kd --pl 1, but the
# exact published configs are not checked in anywhere, so they are launched by hand.
#
# queue (5th arg, default "both"): dist|cllora|both|prep. Use dist/cllora instead of both when
# only one baseline family is missing for that dataset -- launching the other queue anyway
# would retrain and immediately discard a shared task0 checkpoint for nothing (CRE's
# run_cre_dist.sh deletes it at the end of every full sweep and recreates it if missing; the
# CED dist_queue.sh keeps it but running it again just to reuse an existing task0 is wasted
# work if CL-LoRA is the only thing missing). `prep` runs the data check/tokenization and
# stops, for warming a dataset up before the GPUs are free. Not available in no-argument mode
# (always both).
#
# RESUME=1 forwards --resume to both CED queues, so a perm that died mid-task continues from
# its manifest instead of aborting with "partial run exists". Default 0 (a partial run is an
# error you look at first).
#
# Logs, all flat in logs/ (CED): logs/<ds>_{dist,cllora}_queue.log for each queue, and per run
# logs/<ds>_{dist,cllora}_<method>_perm<p>_*.log (see dist_queue.sh / run_cllora.sh). CRE queues
# log to logs/<ds>_cre_{dist,cllora}_queue.log. No-argument mode also writes logs/run_all.log.
# Safe to re-run: every step below skips work that already completed (each run's own
# resume/skip-if-complete logic), whichever family the dataset belongs to.
set -uo pipefail

cd "$(dirname "$0")"
# exported, not plain: no-argument mode reads these inside a detached `bash -c` that only gets
# the functions (declare -f) and the environment, so an unexported CLLORA_METHODS reached
# run_all as an empty string there and the CED CL-LoRA queue silently trained nothing.
export CLLORA_METHODS="inclora olora tree inflora epi migu gainlora_o gainlora_inf"
export DIST_METHODS="kd rkl sfkl srkl csd distillm amid"   # labels dist_queue.sh knows
export RESUME=${RESUME:-0}   # both CED queues read this
export PHYS_BS=${PHYS_BS:-8}   # physical micro-batch target on H200 (tools/bench_gpu.sh)

# What is still missing, dataset by dataset (checked 2026-09-26). One entry per dataset,
# entries separated by ";", fields by ":" -> <dataset>:<perms>:<queue>.
# Both CRE datasets are NOT here -- they are already queued on the old host, so this plan is
# CED only and the CRE runners never fire:
#   tacred  one gap left, GainLoRA(InfLoRA) perm4. Back with MISSING_PLAN="tacred:4:cllora"
#           if that queue is lost -- cllora only, since run_cre_dist.sh deletes the shared
#           task0 merged/ dir at the end of every sweep and retrains all five if it runs again.
#   fewrel  nothing usable yet, so it needs the full sweep: MISSING_PLAN="fewrel:0 1 2 3 4:both"
#   geneva  also queued elsewhere. Back with MISSING_PLAN="geneva:0 1 2 3 4:both" -- it has
#           never run, so it needs every perm and both queues.
# Still here:
#   maven   perm0-1 are done for all 15 baselines, perm2 is partial, perm3-4 never ran. The
#           full range is listed anyway: each runner skips its own finished work, so listing
#           0-4 costs nothing and repairs perm2 without anyone having to track where it stopped.
#   rams    nothing has ever run, everything trains.
# Anything already complete is skipped by the runners themselves, so this stays correct as
# runs land; override with MISSING_PLAN="ds:perms:queue;..." or RUN_ALL_DATASETS="ds1 ds2".
export MISSING_PLAN=${MISSING_PLAN:-"maven:0 1 2 3 4:both;rams:0 1 2 3 4:both"}

# Base model: the local copy download.txt puts in models/Qwen3-0.6B when it is there, else the
# hub name. A host without HF access hangs on the hub name, it never errors out. With the local
# copy also default to offline: transformers 4.57 still calls the hub API on tokenizer load
# when online, which hangs the same way.
if [ -f models/Qwen3-0.6B/config.json ]; then
    MODEL_PATH=${MODEL_PATH:-models/Qwen3-0.6B}
    HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}; TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
fi
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}

# GPUs per family, comma-separated. Both modes use these; the single-dataset form can still
# override them with arguments 3 and 4. An id may repeat within a list (one sub-queue per entry,
# sharing that card; each sub-queue gets its own torchrun port); the two lists must not share ids.
# Repeat an id only when the card fits that many runs: the memory guards in the runners are
# snapshots, not reservations.
export GPU_DIST_ALL=${GPU_DIST_ALL:-0,1,2,3}
export GPU_CLLORA_ALL=${GPU_CLLORA_ALL:-4,5,6,7}

# Host-specific knobs, forwarded only when set so each runner keeps its own default. On a host
# that is not A40_3/A40_4 you will usually want at least PY/ENV_BIN (conda env paths) and
# DISK_PATH (the runners' free-space guard, /mnt here). The detached queues inherit the
# environment, not this shell's unexported variables, hence the export.
#   PY         python for the CL-LoRA engine        (default: envs/nuquant, CED / envs/mta, CRE)
#   ENV_BIN    bin/ of the env for the KD runners   (default: envs/mta)
#   DISK_PATH  filesystem the disk guard watches    (default: . after the fix, /mnt before)
#   HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE           (CED CL-LoRA defaults to offline: set 0 on
#              a host whose HF cache does not already hold Qwen3-0.6B)
for _v in PY ENV_BIN CUDA_HOME DISK_PATH NEED_DISK_GB HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; do
    [ -n "${!_v:-}" ] && export "${_v}"
done
unset _v

family_of () {  # $1=dataset -> echoes cre|ced
    case "$1" in
        tacred|fewrel) echo cre ;;
        maven|rams|geneva|ace) echo ced ;;
        *) return 1 ;;
    esac
}

queue_log () {  # $1=family $2=dataset $3=dist|cllora -> that queue's log file
    if [ "$1" = "cre" ]; then echo "logs/$2_cre_$3_queue.log"; else echo "logs/$2_$3_queue.log"; fi
}

protocol_of () {  # $1=dataset -> the CED run-name tag (PROTOCOL) for that dataset
    # CED run names are built from method+perm+protocol+seed and carry NO dataset of their own
    # (cllora_olora_perm0_v2_s42, dist_rkl_perm0_v2_s42), so a second CED dataset at the same
    # perm lands on the exact same run dir and log file: run_cllora.sh then bails with
    # "refusing to overwrite existing run", or with RESUME=1 quietly continues the other
    # dataset's run. ACE and MAVEN only stayed apart because they ran on different machines.
    # Fold the dataset into the tag the way the CRE runners already do (--protocol
    # <ds>_cre), keeping ace/maven on the bare "v2" so their finished and in-flight runs stay
    # recognized by their existing names.
    case "$1" in
        ace|maven) echo v2 ;;
        *) echo "$1_v2" ;;
    esac
}

# Each queue is its own detached process (setsid nohup at the call site), so it survives
# this shell exiting. Arguments are passed positionally rather than via environment
# variables, since a backgrounded `bash -c` does not inherit unexported shell variables.
split_methods () {  # $1=slot $2=slots $3..=methods -> every $2-th method starting at $1
    local slot=$1 n=$2 i=0 m out=""; shift 2
    for m in "$@"; do
        [ $((i % n)) -eq "${slot}" ] && out+=" ${m}"
        i=$((i + 1))
    done
    echo ${out}
}

run_dist_queue () {   # $1=family $2=dataset $3=gpus (comma list) $4..=perms
    local family=$1 ds=$2; local -a gpus=(${3//,/ }); shift 3
    local p i rc pid pids
    for p in "$@"; do
        if [ "${family}" = "cre" ]; then
            bash scripts/qwen/cre/run_cre_dist.sh "${ds}" "${p}" "${gpus[0]}"
            rc=$?
        else
            # Shared task0 first, on one GPU: every method starts from it, and parallel
            # sub-queues would each try to create it (the loser dies on the partial dir).
            PERM=${p} GPU=${gpus[0]} PROTOCOL="$(protocol_of "${ds}")" DIST_METHODS="" MASTER_PORT=29599 \
                DATA_PREFIX="${ds}_b10_perm" bash scripts/qwen/ced/dist_queue.sh
            rc=$?
            if [ "${rc}" -eq 0 ]; then
                pids=()
                for i in "${!gpus[@]}"; do
                    PERM=${p} GPU=${gpus[i]} PROTOCOL="$(protocol_of "${ds}")" MASTER_PORT=$((29600 + i)) \
                        DIST_METHODS="$(split_methods "${i}" "${#gpus[@]}" ${DIST_METHODS})" \
                        DATA_PREFIX="${ds}_b10_perm" bash scripts/qwen/ced/dist_queue.sh &
                    pids+=($!)
                done
                for pid in "${pids[@]}"; do wait "${pid}" || rc=$?; done
            fi
        fi
        # dist_queue.sh (CED) has no top-level FAILED marker of its own -- it dies silently
        # under set -e -- so log it here regardless of family, or a crashed perm just looks
        # like the queue quietly moved on.
        [ "${rc}" -eq 0 ] || echo "[run.sh] FAILED dist queue: family=${family} ds=${ds} perm=${p} (exit ${rc}), see logs/${ds}_*.log"
    done
}
run_cllora_queue () {  # $1=family $2=dataset $3=gpus (comma list) $4=methods $5..=perms
    local family=$1 ds=$2 methods=$4; local -a gpus=(${3//,/ }); shift 4
    local p i rc pid pids
    for p in "$@"; do
        rc=0
        if [ "${family}" = "cre" ]; then
            bash scripts/qwen/cre/run_cre_cllora.sh "${ds}" "${p}" "${gpus[0]}"
            rc=$?
        else
            pids=()
            for i in "${!gpus[@]}"; do
                DATA_ROOT="data/${ds}_b10_perm${p}" PROTOCOL="$(protocol_of "${ds}")" \
                    bash scripts/qwen/ced/run_all_cllora.sh "${gpus[i]}" \
                    $(split_methods "${i}" "${#gpus[@]}" ${methods}) &
                pids+=($!)
            done
            for pid in "${pids[@]}"; do wait "${pid}" || rc=$?; done
        fi
        [ "${rc}" -eq 0 ] || echo "[run.sh] FAILED cllora queue: family=${family} ds=${ds} perm=${p} (exit ${rc}), see logs/${ds}_*.log"
    done
}

check_data () {  # $1=family $2=dataset $3=perms -- tokenizes both families; CED tokenization
                  # is NOT done by run_ced_v2.sh itself (it only tokenizes PL/balance side-data,
                  # never the base task data -- confirmed the hard way against a real RAMS run),
                  # so this has to run tools/process_data.py per task itself, same as the
                  # hand-written maven_dist_perm14.sh that has been doing this successfully.
    local family=$1 ds=$2; shift 2
    if [ "${family}" = "cre" ]; then
        for p in "$@"; do
            bash scripts/qwen/cre/prep_cre.sh "${ds}" "${p}" || { echo "[run.sh] tokenize failed for ${ds} perm${p}"; return 1; }
        done
    else
        local PY=${PY:-python3}
        mkdir -p logs
        for p in "$@"; do
            [ -s "data/${ds}_b10_perm${p}/streams.json" ] || {
                echo "[run.sh] missing data/${ds}_b10_perm${p}/streams.json -- build it first (tools/build_maven_perms.py --src data/${ds} --out-prefix ${ds}_b10_perm)"
                return 1
            }
            for t in 0 1 2 3 4; do
                local out="processed_data/${ds}_b10_perm${p}/${t}"
                [ -d "${out}/qwen" ] && [ -n "$(ls -A "${out}/qwen" 2>/dev/null)" ] && continue
                PYTHONPATH=. ${PY} tools/process_data.py \
                    --data-dir "data/${ds}_b10_perm${p}/${t}/" --processed-data-dir "${out}" \
                    --model-path "${MODEL_PATH}" --data-process-workers 4 \
                    --max-prompt-length 460 --t-max-prompt-length 640 \
                    --dev-num 1000 --model-type qwen > "logs/${ds}_tokenize_perm${p}_task${t}.log" 2>&1 || {
                    echo "[run.sh] tokenize failed for ${ds} perm${p} task${t}, see logs/${ds}_tokenize_perm${p}_task${t}.log"
                    tail -10 "logs/${ds}_tokenize_perm${p}_task${t}.log"
                    return 1
                }
            done
        done
    fi
}

if [ $# -eq 0 ]; then
    # ---- no-argument mode: everything, one dataset at a time ----
    run_all () {
        local plan=${MISSING_PLAN}
        # RUN_ALL_DATASETS stays supported: a bare dataset list means "all perms, both queues".
        if [ -n "${RUN_ALL_DATASETS:-}" ]; then
            plan=""
            for ds in ${RUN_ALL_DATASETS}; do plan="${plan}${ds}:0 1 2 3 4:both;"; done
        fi
        local old_ifs=$IFS entries entry ds perms queue rest family dpid cpid
        IFS=';'; entries=(${plan}); IFS=$old_ifs
        for entry in "${entries[@]}"; do
            [ -n "${entry}" ] || continue
            ds=${entry%%:*}; rest=${entry#*:}; perms=${rest%%:*}; queue=${rest##*:}
            family=$(family_of "${ds}") || { echo "[run.sh:all] unknown dataset ${ds}, skipping"; continue; }
            echo "[run.sh:all] === ${ds} (${family}) perms='${perms}' queue=${queue} start $(date -Iseconds) ==="
            check_data "${family}" "${ds}" ${perms} || { echo "[run.sh:all] === ${ds} data check failed, skipping ==="; continue; }
            dpid=""; cpid=""
            if [ "${queue}" = "dist" ] || [ "${queue}" = "both" ]; then
                ( run_dist_queue "${family}" "${ds}" "${GPU_DIST_ALL}" ${perms} ) > "$(queue_log "${family}" "${ds}" dist)" 2>&1 &
                dpid=$!
            fi
            if [ "${queue}" = "cllora" ] || [ "${queue}" = "both" ]; then
                ( run_cllora_queue "${family}" "${ds}" "${GPU_CLLORA_ALL}" "${CLLORA_METHODS}" ${perms} ) > "$(queue_log "${family}" "${ds}" cllora)" 2>&1 &
                cpid=$!
            fi
            [ -n "${dpid}${cpid}" ] && wait ${dpid} ${cpid}
            echo "[run.sh:all] === ${ds} done $(date -Iseconds) ==="
        done
        echo "[run.sh:all] ALL DATASETS DONE $(date -Iseconds)"
    }
    if [ "${FOREGROUND:-0}" = "1" ]; then
        # project_commands.sh: stay attached, so the job that launched it lasts as long as training
        mkdir -p logs
        run_all 2>&1 | tee logs/run_all.log
        exit 0
    fi
    mkdir -p logs
    setsid nohup bash -c "$(declare -f family_of protocol_of queue_log split_methods run_dist_queue run_cllora_queue check_data run_all); run_all" \
        > logs/run_all.log 2>&1 < /dev/null &
    echo "[run.sh] running every missing run, one dataset at a time, pid $! -> logs/run_all.log"
    echo "[run.sh] plan: ${RUN_ALL_DATASETS:-${MISSING_PLAN}}"
    echo "[run.sh] tail -f logs/run_all.log"
    exit 0
fi

DS=$1
PERMS=${2:-"0 1 2 3 4"}
GPU_DIST=${3:-${GPU_DIST_ALL}}
GPU_CLLORA=${4:-${GPU_CLLORA_ALL}}
QUEUE=${5:-both}

FAMILY=$(family_of "${DS}") || { echo "unknown dataset '${DS}' (expected tacred|fewrel|maven|rams|geneva)"; exit 1; }
case "${QUEUE}" in
    dist|cllora|both|prep) ;;
    *) echo "unknown queue '${QUEUE}' (expected dist|cllora|both|prep)"; exit 1 ;;
esac
[ -n "${PERMS}" ] || PERMS="0 1 2 3 4"   # so `run.sh geneva "" 0 1 prep` still means all perms

echo "[run.sh] dataset=${DS} family=${FAMILY} perms='${PERMS}' gpu_dist=${GPU_DIST} gpu_cllora=${GPU_CLLORA} queue=${QUEUE} resume=${RESUME}"
check_data "${FAMILY}" "${DS}" ${PERMS} || exit 1
if [ "${QUEUE}" = "prep" ]; then
    echo "[run.sh] data ready for ${DS} perms '${PERMS}', nothing launched (queue=prep)"
    exit 0
fi

if [ "${QUEUE}" = "dist" ] || [ "${QUEUE}" = "both" ]; then
    mkdir -p logs
    setsid nohup bash -c "$(declare -f protocol_of split_methods run_dist_queue); run_dist_queue \"\$@\"" _ \
        "${FAMILY}" "${DS}" "${GPU_DIST}" ${PERMS} \
        > "$(queue_log "${FAMILY}" "${DS}" dist)" 2>&1 < /dev/null &
    DIST_PID=$!
    echo "[run.sh] distillation queue running on gpus ${GPU_DIST}, pid ${DIST_PID} -> $(queue_log "${FAMILY}" "${DS}" dist)"
fi
if [ "${QUEUE}" = "cllora" ] || [ "${QUEUE}" = "both" ]; then
    mkdir -p logs
    setsid nohup bash -c "$(declare -f protocol_of split_methods run_cllora_queue); run_cllora_queue \"\$@\"" _ \
        "${FAMILY}" "${DS}" "${GPU_CLLORA}" "${CLLORA_METHODS}" ${PERMS} \
        > "$(queue_log "${FAMILY}" "${DS}" cllora)" 2>&1 < /dev/null &
    CLLORA_PID=$!
    echo "[run.sh] CL-LoRA queue running on gpus ${GPU_CLLORA}, pid ${CLLORA_PID} -> $(queue_log "${FAMILY}" "${DS}" cllora)"
fi
TAIL=()
case "${QUEUE}" in
    dist|both)   TAIL+=("$(queue_log "${FAMILY}" "${DS}" dist)") ;;
esac
case "${QUEUE}" in
    cllora|both) TAIL+=("$(queue_log "${FAMILY}" "${DS}" cllora)") ;;
esac
echo "[run.sh] tail -f ${TAIL[*]}"
