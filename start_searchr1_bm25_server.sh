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
cd /root/autodl-tmp/memrag_new/Search-R1
if curl -fsS http://127.0.0.1:8000/docs >/dev/null 2>&1; then
  echo "Search-R1 retrieval server already ready at http://127.0.0.1:8000"
  exit 0
fi
mkdir -p /root/autodl-tmp/searchr1_logs
CORPUS_JSONL=/root/autodl-tmp/searchr1_data/wiki18_corpus/extracted/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl
nohup python search_r1/search/retrieval_server.py \
  --index_path /root/autodl-tmp/searchr1_data/wiki18_bm25/bm25 \
  --corpus_path "$CORPUS_JSONL" \
  --topk 3 \
  --retriever_name bm25 \
  > /root/autodl-tmp/searchr1_logs/retrieval_server.log 2>&1 &
echo $! > /root/autodl-tmp/searchr1_logs/retrieval_server.pid
echo "Started Search-R1 retrieval server PID $(cat /root/autodl-tmp/searchr1_logs/retrieval_server.pid)"
echo "Log: /root/autodl-tmp/searchr1_logs/retrieval_server.log"
