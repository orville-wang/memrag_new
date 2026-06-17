#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/hf-cache/datasets
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
export OPENAI_API_KEY=dummy
source /root/autodl-tmp/envs/memgen/bin/activate
/root/autodl-tmp/memrag_new/start_searchr1_bm25_server.sh
python /root/autodl-tmp/memrag_new/run_memgen_searchr1_real_smoke.py
