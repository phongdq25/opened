#!/usr/bin/env bash
set -e


mkdir -p "./models"

for n in ace maven rams geneva tacred fewrel; do
    wget -c https://huggingface.co/datasets/datht/processed-cl-$n/resolve/main/${n}_all.tar.gz
done

hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B