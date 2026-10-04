#!/bin/bash
# Tokenise one CRE dataset+permutation into processed_data/ so the CED runners can train on it.
# CRE splits ship as 10 tasks (TACRED 40 relations, FewRel 80 relations, 4 or 8 per task),
# unlike the 5-task ACE/MAVEN streams.
#
#   bash scripts/qwen/cre/prep_cre.sh tacred 0
#   for p in 0 1 2 3 4; do bash scripts/qwen/cre/prep_cre.sh fewrel $p; done
set -euo pipefail

DS=${1:?dataset required: tacred|fewrel}
PERM=${2:?perm required: 0..4}
NTASK=${NTASK:-10}
ENV_BIN=${ENV_BIN:-$HOME/miniconda3/envs/mta/bin}
export CUDA_HOME=${CUDA_HOME:-$HOME/miniconda3/envs/mta}

cd "$(dirname "$0")/../../.."
SRC=data/${DS}_perm${PERM}
[ -s "${SRC}/streams.json" ] || { echo "khong thay ${SRC}/streams.json"; exit 1; }

echo "PREP ${DS} perm${PERM} ($(date))"
for T in $(seq 0 $((NTASK - 1))); do
    OUT=processed_data/${DS}_perm${PERM}/${T}
    if [ -d "${OUT}/qwen" ] && [ -n "$(ls -A "${OUT}/qwen" 2>/dev/null)" ]; then
        echo "  skip task${T}"; continue
    fi
    [ -s "${SRC}/${T}/train.jsonl" ] || { echo "  thieu ${SRC}/${T}/train.jsonl"; exit 1; }
    PYTHONPATH=. ${ENV_BIN}/python tools/process_data.py \
        --data-dir ${SRC}/${T}/ --processed-data-dir ${OUT} \
        --model-path Qwen/Qwen3-0.6B --data-process-workers 4 \
        --max-prompt-length 460 --t-max-prompt-length 640 \
        --dev-num 1000 --model-type qwen > /tmp/tok_${DS}_p${PERM}_t${T}.log 2>&1 \
        && echo "  tokenized task${T}" \
        || { echo "  TOKENIZE FAILED task${T}"; tail -5 /tmp/tok_${DS}_p${PERM}_t${T}.log; exit 1; }
done
du -sh processed_data/${DS}_perm${PERM}
echo "PREP DONE ${DS} perm${PERM} ($(date))"
