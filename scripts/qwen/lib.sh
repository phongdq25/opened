# Helpers sourced by the runner scripts (bash), from the repo root.

# phys_split <logical micro-batch g> <logical accumulation G> [<target physical batch>]
#   -> "<physical batch P> <physical accumulation G'>"
# P = k*g for the largest k with k*g <= target and k dividing G, so every loss group is one
# logical micro-batch and P*G' = g*G rows go into each update. A target below g gives P = g.
phys_split () {
    local g=$1 acc=$2 want=${3:-$1} k
    k=$(( want / g ))
    [ "${k}" -ge 1 ] || k=1
    while [ $(( acc % k )) -ne 0 ]; do k=$(( k - 1 )); done
    echo "$(( g * k )) $(( acc / k ))"
}

# run_active <cl method> <data root>: is a CL-LoRA engine already training exactly this run?
# Matches the engine's command line, so the same method on another order does not count.
run_active () {
    ps -eo args | grep -F -- "--cl-method $1 --data-root $2 " | grep -v -F "grep" > /dev/null
}

# gpu_defaults <MiB of the smallest card>
#   -> "<SLOTS_PER_GPU> <PHYS_BS> <USE_MPS> <NEED_GPU_MB> <NEED_LORA_MB> <GEN_BACKEND>" ("-" = leave unset)
# The values tools/bench_gpu.sh measured on 1x H200 NVL (README) go to H200-class cards only;
# smaller cards keep one run per GPU and the runners' own settings (PHYS_BS = --bs, no MPS).
# GEN_BACKEND stays hf until the vLLM end-to-end check passes (plan 2026-10-06, Task 8).
gpu_defaults () {
    if [ "${1:-0}" -ge 130000 ]; then echo "3 8 1 39833 18432 hf"; else echo "1 - 0 - - hf"; fi
}

# apply_card_defaults [<gpu id> ...]: fill in SLOTS_PER_GPU, PHYS_BS, USE_MPS, NEED_GPU_MB, NEED_LORA_MB
# and GEN_BACKEND where the caller left them unset, from the smallest listed card (all by default).
apply_card_defaults () {
    local ids mib d_slots d_phys d_mps d_gpu d_lora d_gen
    ids=$(echo "$@" | tr ' ' ',')
    mib=$( (nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits ${ids:+-i "${ids}"} 2>/dev/null || true) \
        | awk 'NR == 1 || $1 < m { m = $1 } END { print m + 0 }')
    read -r d_slots d_phys d_mps d_gpu d_lora d_gen <<< "$(gpu_defaults "${mib}")"
    SLOTS_PER_GPU=${SLOTS_PER_GPU:-${d_slots}}
    export USE_MPS=${USE_MPS:-${d_mps}}
    [ "${d_phys}" = "-" ] || export PHYS_BS=${PHYS_BS:-${d_phys}}
    [ "${d_gpu}" = "-" ] || export NEED_GPU_MB=${NEED_GPU_MB:-${d_gpu}}
    [ "${d_lora}" = "-" ] || export NEED_LORA_MB=${NEED_LORA_MB:-${d_lora}}
    export GEN_BACKEND=${GEN_BACKEND:-${d_gen}}
}

# mps_start <python> <gpu id>: start a CUDA MPS daemon under ./.mps, then check that a CUDA
# client can use it (a started daemon does not prove that, e.g. in some containers).
#   0 = running, CUDA_MPS_* exported; 1 = not usable here, daemon stopped, environment clean.
mps_start () {
    command -v nvidia-cuda-mps-control > /dev/null || return 1
    export CUDA_MPS_PIPE_DIRECTORY=${PWD}/.mps/pipe CUDA_MPS_LOG_DIRECTORY=${PWD}/.mps/log
    mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
    if nvidia-cuda-mps-control -d && CUDA_VISIBLE_DEVICES=$2 "$1" -c \
            "import torch; torch.zeros(1, device='cuda'); torch.cuda.synchronize()" > /dev/null 2>&1; then
        return 0
    fi
    echo quit | nvidia-cuda-mps-control > /dev/null 2>&1 || true
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
    return 1
}

# vllm_check <python>: with GEN_BACKEND=vllm, stop unless gen_backend.find_vllm_python() finds a vLLM
# environment (VLLM_PY, ./.venv-vllm or /venv/main). Prints the interpreter and the vLLM version.
vllm_check () {
    [ "${GEN_BACKEND:-hf}" = "vllm" ] || return 0
    "$1" -c "from gen_backend import find_vllm_python; print('vLLM', *find_vllm_python())"
}
