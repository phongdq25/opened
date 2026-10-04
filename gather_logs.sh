#!/bin/bash
# Collect only the files needed to read per-task F1 of every run into one folder.
# No checkpoints, no model outputs, no training stdout.
#
#   bash gather_logs.sh                  # -> collected_logs/<host>_<date>/
#   bash gather_logs.sh a100_run1        # -> collected_logs/a100_run1/
#
# Copied from results/qwen3/ced/<run>/, repo layout kept:
#   task*/**/log.txt     dist runs: one "dev | ..." / "test | ..." line per eval, F1 inside
#   cl_results.json      CL-LoRA runs: {"task<t>": {"trigger": {"f1": ...}}, ...}
#   run_config.txt, run_manifest.json, .complete   which config, how far it got, finished or not
set -euo pipefail
cd "$(dirname "$0")"

LABEL=${1:-$(hostname -s)_$(date +%Y%m%d_%H%M)}
OUT=${OUT_ROOT:-collected_logs}/${LABEL}   # OUT_ROOT=logs: project_commands.sh keeps everything in logs/
[ -d results ] || { echo "no results/ here"; exit 1; }
[ ! -e "${OUT}" ] || { echo "${OUT} already exists, pick another label"; exit 1; }
mkdir -p "${OUT}"

n=0
while IFS= read -r -d '' f; do
    mkdir -p "${OUT}/$(dirname "$f")"
    cp -p "$f" "${OUT}/$f"
    n=$((n + 1))
done < <(find results -path '*/merged' -prune -o -type f \( -name log.txt -o -name cl_results.json \
    -o -name run_config.txt -o -name run_manifest.json -o -name .complete \) -print0)

echo "files:  ${n}"
echo "folder: ${OUT} ($(du -sh "${OUT}" | cut -f1))"
