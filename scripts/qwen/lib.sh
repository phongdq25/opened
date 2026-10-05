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
