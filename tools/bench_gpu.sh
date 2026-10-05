#!/bin/bash
# Short timing runs of every workload on GPU 0, to choose PHYS_BS / SLOTS_PER_GPU / USE_MPS
# (perf/h200-throughput). Writes bench/<tag>/summary.tsv, prints it, then prints the defaults
# the rule in docs/superpowers/plans/2026-10-05-h200-throughput.md (Task 13) picks.
#
#   bash tools/bench_gpu.sh h200
#   PHYS_LIST="8 16" SHARE_LIST="2 4" bash tools/bench_gpu.sh h200_quick
#
# Phases (new code, 1 epoch each, loss groups of 2):
#   rkl    FewRel task1 RKL distillation, 1280 rows, physical batch P
#   ours   ACE task1 Ours (sfkl + span + PL + boost x5 + SD), 320 rows, physical batch P
#   cl     CL-LoRA IncLoRA on FewRel tasks 0-1, 640 rows each, physical batch P
#   share  N RKL runs at SHARE_PHYS sharing the card, without and with CUDA MPS
set -uo pipefail
cd "$(dirname "$0")/.."
TAG=${1:?usage: bash tools/bench_gpu.sh <tag>}
OUT=bench/${TAG}
mkdir -p "${OUT}"
PHYS_LIST=${PHYS_LIST:-"2 8 16 32"}
SHARE_LIST=${SHARE_LIST:-"1 2 3 4"}
SHARE_PHYS=${SHARE_PHYS:-16}
export ENV_BIN=${ENV_BIN:-${PWD}/.venv/bin} PY=${PY:-${PWD}/.venv/bin/python}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID DISK_PATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -d /usr/local/cuda ] && export CUDA_HOME=/usr/local/cuda
R=results/qwen3/ced
SUMMARY=${OUT}/summary.tsv
printf "phase\tphys\truns\tmps\twall_s\ts_per_update\tpeak_gib\tmean_util\n" > "${SUMMARY}"

sampler_start () {
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0 -lms 500 > "$1" &
    SAMPLER=$!
}
sampler_stats () {  # $1=csv -> "peak_gib<TAB>mean_util"
    awk -F', *' '{ if ($1 > m) m = $1; u += $2; n++ } END { printf "%.1f\t%.0f", m / 1024, (n ? u / n : 0) }' "$1"
}
step_time () {  # $1=run dir -> mean "step time" of its task1 train lines (s per update)
    find "$1/task1" -name log.txt 2>/dev/null | head -1 | xargs -r grep -h "^train | epoch" \
        | grep -oE "step time: [0-9.]+" | awk '{ s += $3; n++ } END { printf "%.2f", (n ? s / n : 0) }'
}
record () {  # $1=phase $2=phys $3=runs $4=mps $5=start (epoch s) $6=csv $7=run dir or "-"
    local wall=$(( $(date +%s) - $5 )) st="-"
    [ "$7" != "-" ] && st=$(step_time "$7")
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "${wall}" "${st}" "$(sampler_stats "$6")" >> "${SUMMARY}"
}
ok_or_failed () {  # $1=phase $2=file a finished run leaves -> phase, or <phase>_failed (the rule ignores those)
    if [ -e "$2" ]; then echo "$1"; else echo "$1_failed"; fi
}
run_v2 () {  # $1=run name $2=port, then run_ced_v2.sh flags
    local name=$1 port=$2
    shift 2
    rm -rf "${R:?}/${name}"
    MASTER_PORT=${port} bash scripts/qwen/ced/run_ced_v2.sh --run-name "${name}" "$@" > "${OUT}/${name}.log" 2>&1
}
BASE=(--rank 16 --alpha 64 --lr 0.0002 --seed 42 --greedy 1 --gpus 0 --epochs 1 --bs 2 --acc 16
      --extra "--log-interval 5")
RKL=(--mode ce_kd --kd-type rkl --w-span 0 --kd-ratio 0.9 --skew 0.1 --span-metric cosine --layers "22 25 28"
     --data-prefix fewrel_perm --perm 0 --start-task 1 --end-task 1 --task0-source-run bench_t0_fewrel
     --train-num 1280 --dev-num 320)
OURS=(--mode ce_kd --kd-type sfkl --w-span 2.0 --kd-ratio 0.9 --skew 0.1 --span-metric cosine --layers "22 25 28"
      --data-prefix ace_b10_perm --perm 0 --pl 1 --replay-boost 5 --pl-conf percentile --pl-conf-pct 70 --sd 1
      --start-task 1 --end-task 1 --task0-source-run bench_t0_ace --train-num 320)

# the shared task0 teachers (plain SFT), once
[ -f "${R}/bench_t0_fewrel/.complete" ] || PHYS_BS=16 run_v2 bench_t0_fewrel 29590 --mode sft \
    --data-prefix fewrel_perm --perm 0 --end-task 0 --train-num 1280 --dev-num 320 "${BASE[@]}"
[ -f "${R}/bench_t0_ace/.complete" ] || PHYS_BS=16 run_v2 bench_t0_ace 29591 --mode sft \
    --data-prefix ace_b10_perm --perm 0 --end-task 0 "${BASE[@]}"

for P in ${PHYS_LIST}; do
    t=$(date +%s); sampler_start "${OUT}/rkl_p${P}.csv"
    PHYS_BS=${P} run_v2 "bench_rkl_p${P}" 29592 "${RKL[@]}" "${BASE[@]}"
    kill "${SAMPLER}"; record "$(ok_or_failed rkl "${R}/bench_rkl_p${P}/.complete")" "${P}" 1 0 "${t}" \
        "${OUT}/rkl_p${P}.csv" "${R}/bench_rkl_p${P}"

    t=$(date +%s); sampler_start "${OUT}/ours_p${P}.csv"
    PHYS_BS=${P} run_v2 "bench_ours_p${P}" 29593 "${OURS[@]}" "${BASE[@]}"
    kill "${SAMPLER}"; record "$(ok_or_failed ours "${R}/bench_ours_p${P}/.complete")" "${P}" 1 0 "${t}" \
        "${OUT}/ours_p${P}.csv" "${R}/bench_ours_p${P}"

    t=$(date +%s); sampler_start "${OUT}/cl_p${P}.csv"
    rm -rf "${R:?}/cllora_inclora_perm0_bench${P}_s42"
    PHYS_BS=${P} bash scripts/qwen/ced/run_cllora.sh --method inclora --data-root data/fewrel_perm0 \
        --num-tasks 10 --end-task 1 --limit 640 --epochs 1 --batch-size 2 --grad-accum 16 --eval-batch-size 128 \
        --gpu 0 --py "${PY}" --protocol "bench${P}" > "${OUT}/cl_p${P}.log" 2>&1
    kill "${SAMPLER}"; record "$(ok_or_failed cl "${R}/cllora_inclora_perm0_bench${P}_s42/predictions/task1.jsonl")" \
        "${P}" 1 0 "${t}" "${OUT}/cl_p${P}.csv" "-"
done

mps_on () {
    command -v nvidia-cuda-mps-control > /dev/null || return 1
    export CUDA_MPS_PIPE_DIRECTORY=${PWD}/.mps/pipe CUDA_MPS_LOG_DIRECTORY=${PWD}/.mps/log
    mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
    nvidia-cuda-mps-control -d
}
mps_off () { echo quit | nvidia-cuda-mps-control; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; }

for MPS in 0 1; do
    if [ "${MPS}" = "1" ]; then
        mps_on || { echo "# no usable CUDA MPS in this container" >> "${SUMMARY}"; break; }
    fi
    for N in ${SHARE_LIST}; do
        t=$(date +%s); sampler_start "${OUT}/share${N}_mps${MPS}.csv"
        pids=()
        for i in $(seq 1 "${N}"); do
            PHYS_BS=${SHARE_PHYS} run_v2 "bench_share${N}_${i}" $((29600 + i)) "${RKL[@]}" "${BASE[@]}" &
            pids+=($!)
        done
        for p in "${pids[@]}"; do wait "${p}"; done
        kill "${SAMPLER}"
        phase=share
        [ "$(compgen -G "${R}/bench_share${N}_*/.complete" | wc -l)" = "${N}" ] || phase=share_failed
        record "${phase}" "${SHARE_PHYS}" "${N}" "${MPS}" "${t}" "${OUT}/share${N}_mps${MPS}.csv" "${R}/bench_share${N}_1"
    done
    [ "${MPS}" = "1" ] && mps_off
done
column -t -s $'\t' "${SUMMARY}"

# The defaults rule:
#   PHYS_BS        largest P whose RKL step time is within 10% of the fastest P, among the P whose
#                  Ours peak memory leaves room for 2 Ours runs on the card (2 x peak <= 0.92 x total)
#   SLOTS_PER_GPU  the N with the most runs finished per hour (N / wall) at SHARE_PHYS, MPS or not
#   USE_MPS        1 if that best N is at least 10% faster with MPS
#   NEED_*_MB      the single-run peak at the chosen P plus 2 GiB
"${PY}" - "${SUMMARY}" "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1]), delimiter="\t") if not r["phase"].startswith("#")]
total_gib = float(sys.argv[2]) / 1024
get = lambda phase, phys, runs="1", mps="0": next((r for r in rows if (r["phase"], r["phys"], r["runs"], r["mps"]) == (phase, str(phys), str(runs), str(mps))), None)
phys = sorted({int(r["phys"]) for r in rows if r["phase"] == "rkl"})
fits = [p for p in phys if get("ours", p) and 2 * float(get("ours", p)["peak_gib"]) <= 0.92 * total_gib]
fastest = min(float(get("rkl", p)["s_per_update"]) for p in fits)
p_best = max(p for p in fits if float(get("rkl", p)["s_per_update"]) <= 1.10 * fastest)
share = [r for r in rows if r["phase"] == "share"]
rate = lambda r: int(r["runs"]) / float(r["wall_s"])
best = max(share, key=rate)
plain = get("share", best["phys"], best["runs"], "0")
use_mps = int(best["mps"] == "1" and rate(best) >= 1.10 * rate(plain))
need_gpu = int(max(float(get("rkl", p_best)["peak_gib"]), float(get("ours", p_best)["peak_gib"])) * 1024 + 2048)
cl = get("cl", p_best)
need_lora = int(float(cl["peak_gib"]) * 1024 + 2048) if cl else 14000      # runner default when cl failed
print(f"PHYS_BS={p_best} SLOTS_PER_GPU={best['runs']} USE_MPS={use_mps} NEED_GPU_MB={need_gpu} NEED_LORA_MB={need_lora}")
PY
