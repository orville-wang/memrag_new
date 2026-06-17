#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/hf-cache/datasets
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
source /root/autodl-tmp/envs/memgen/bin/activate
cd /root/autodl-tmp/memrag_new/MemGen
python smoke_memgen_tiny.py
