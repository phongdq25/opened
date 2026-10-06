#!/bin/bash
# Which packing finishes the most Ours updates per hour on one GPU? Run it on GPU 0 of a free card
# before a launch with Ours runs. Writes bench/<tag>/ours.tsv and prints it.
#
#   bash tools/bench_ours.sh h200_ours
#
# tools/bench_gpu.sh's Ours workload (ACE task 1: sfkl + span + PL + boost x5 + SD, 320 rows, 1 epoch,
# loss groups of 2) with COMPILE_GEN=1 (Ours' setting) and CUDA MPS, the runs of a row at the same time:
#   3 x PHYS_BS 8 (the H200 default), 3 x PHYS_BS 16 and 3 x PHYS_BS 32 with GRAD_CKPT=1,
#   1 x PHYS_BS 32 without (one run per card).
# updates_per_h = finished runs x updates per run x 3600 / wall seconds.
#
# Gradient checkpointing leaves Ours' loss and gradients unchanged (tests/test_ced_step.py). Ours
# runs do not repeat exactly on the GPU, with or without it, so this does not compare losses.
set -uo pipefail
cd "$(dirname "$0")/.."
TAG=${1:?usage: bash tools/bench_ours.sh <tag>}
OUT=bench/${TAG}
mkdir -p "${OUT}"
export ENV_BIN=${ENV_BIN:-${PWD}/.venv/bin} PY=${PY:-${PWD}/.venv/bin/python}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID DISK_PATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -d /usr/local/cuda ] && export CUDA_HOME=/usr/local/cuda
R=results/qwen3/ced
SUMMARY=${OUT}/ours.tsv
printf "phase\tphys\truns\tckpt\twall_s\ts_per_update\tpeak_gib\tupdates_per_h\n" > "${SUMMARY}"

sampler_start () {
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0 -lms 500 > "$1" &
    SAMPLER=$!
}
peak_gib () { awk -F', *' '{ if ($1 > m) m = $1 } END { printf "%.1f", m / 1024 }' "$1"; }
train_lines () { grep -h "^train | epoch" "${R}/$1/task1/log.txt" 2>/dev/null; }
step_time () {  # $1=run name -> mean "step time" of its train lines (s per update)
    train_lines "$1" | grep -oE "step time: [0-9.]+" | awk '{ s += $3; n++ } END { printf "%.2f", (n ? s / n : 0) }'
}
updates () {  # $1=run name -> updates in its epoch, from the last "global iter: x/<updates>"
    train_lines "$1" | tail -1 | grep -oE "global iter: +[0-9]+/ *[0-9]+" | awk -F/ '{ print $2 + 0 }'
}
run_v2 () {  # $1=run name $2=port, then run_ced_v2.sh flags
    local name=$1 port=$2
    shift 2
    rm -rf "${R:?}/${name}"
    MASTER_PORT=${port} bash scripts/qwen/ced/run_ced_v2.sh --run-name "${name}" "$@" > "${OUT}/${name}.log" 2>&1
}
BASE=(--rank 16 --alpha 64 --lr 0.0002 --seed 42 --greedy 1 --gpus 0 --epochs 1 --bs 2 --acc 16
      --extra "--log-interval 2")
OURS=(--mode ce_kd --kd-type sfkl --w-span 2.0 --kd-ratio 0.9 --skew 0.1 --span-metric cosine --layers "22 25 28"
      --data-prefix ace_b10_perm --perm 0 --pl 1 --replay-boost 5 --pl-conf percentile --pl-conf-pct 70 --sd 1
      --start-task 1 --end-task 1 --task0-source-run bench_t0_ace --train-num 320)

[ -f "${R}/bench_t0_ace/.complete" ] || PHYS_BS=16 run_v2 bench_t0_ace 29591 --mode sft \
    --data-prefix ace_b10_perm --perm 0 --end-task 0 "${BASE[@]}"

export CUDA_MPS_PIPE_DIRECTORY=${PWD}/.mps/pipe CUDA_MPS_LOG_DIRECTORY=${PWD}/.mps/log
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
nvidia-cuda-mps-control -d || { echo "no usable CUDA MPS in this container"; exit 1; }
for row in "3 8 0" "3 16 1" "3 32 1" "1 32 0"; do
    read -r N P C <<< "${row}"
    t=$(date +%s); sampler_start "${OUT}/pack_${N}x${P}_ckpt${C}.csv"
    pids=()
    for i in $(seq 1 "${N}"); do
        PHYS_BS=${P} GRAD_CKPT=${C} COMPILE_GEN=1 run_v2 "bench_pack_${N}x${P}_${i}" $((29610 + i)) \
            "${OURS[@]}" "${BASE[@]}" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "${p}"; done
    kill "${SAMPLER}"
    wall=$(( $(date +%s) - t ))
    done_runs=$(compgen -G "${R}/bench_pack_${N}x${P}_*/.complete" | wc -l)
    phase=pack
    [ "${done_runs}" = "${N}" ] || phase=pack_failed
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${phase}" "${P}" "${N}" "${C}" "${wall}" \
        "$(step_time "bench_pack_${N}x${P}_1")" "$(peak_gib "${OUT}/pack_${N}x${P}_ckpt${C}.csv")" \
        "$(awk -v n="${done_runs}" -v u="$(updates "bench_pack_${N}x${P}_1")" -v w="${wall}" \
           'BEGIN { printf "%.0f", n * u * 3600 / w }')" >> "${SUMMARY}"
done
echo quit | nvidia-cuda-mps-control
column -t -s $'\t' "${SUMMARY}"
