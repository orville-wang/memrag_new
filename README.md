# memrag_new

这个仓库是 MemGen-style latent memory / retrieval-weaver / policy-fidelity 实验工作区。它不是一个干净的上游库镜像，而是把本轮实验真正用到的代码、脚本、报告、图和可复查 raw result 放在一个顶层 git 里，方便下次直接接着跑。

## 一句话定位

我们在验证：

1. 显式 retrieval 仍然作为 evidence path。
2. MemGen-style latent input experience 更适合作为 policy / search-control path。
3. 对同一个问题状态和同一个外部经验，比较不同 carrier 对模型 policy 的影响：
   - 原版 MemGen latent。
   - Appendix-E 风格 `Retrieval + trained weaver`。
   - Appendix-E 风格 `Retrieval + untrained weaver`。
   - frozen-theta policy fidelity 实验里的 text / latent / random / soft prompt carriers。

最重要的实验报告：

- `docs/mechanistic_paths_appendix_e_large_n100_report.md`
- `docs/policy_fidelity_experiment_report.md`

## 远端机器和仓库

当前主要远端工作目录：

```bash
cd /root/autodl-tmp/memrag_new
```

GitHub remote：

```bash
git remote -v
# origin  git@github.com:orville-wang/memrag_new.git
```

当前约定：

- branch: `main`
- 顶层 git 管理整个工作区。
- `MemGen/` 和 `Search-R1/` 已经去掉自己的嵌套 `.git`，现在作为普通子目录被顶层 git 跟踪。
- 不要重新在 `MemGen/` 或 `Search-R1/` 里单独做 git 操作。

常用 git 操作：

```bash
cd /root/autodl-tmp/memrag_new
git status --short
git pull --ff-only origin main
git add <files>
git commit -m "..."
git push origin main
```

## 不提交什么

`.gitignore` 会排除运行环境和大文件：

- `MemGen_checkpoints/`
- `flashrag_data/`
- `__pycache__/`
- `*.pt`, `*.pth`, `*.safetensors`, `*.bin`, `*.ckpt`
- `*.pkl`, `*.pickle`, `*.faiss`, `*.index`
- `*.tgz`, `*.tar.gz`
- run log，例如 `results/mechanistic_paths/*.run.log`
- 中间训练缓存目录，例如 `latent_controller_debug/`, `latent_controller_hotpot2wiki_n300/`

可以提交的东西：

- 源码脚本。
- Markdown 报告。
- 小/中等 raw JSON/JSONL 结果。
- 论文图 PNG。
- 小规模构造好的 JSONL 数据集。

提交前检查：

```bash
git status --short
git diff --cached --name-only | grep -E '(MemGen_checkpoints|flashrag_data|__pycache__|\\.tgz$|\\.pt$|\\.pkl$)' || true
```

如果第二条有输出，通常不要提交。

## 目录结构

```text
MemGen/
  MemGen 上游代码和本轮 smoke/sanity 用到的补充脚本。

Search-R1/
  Search-R1 上游代码。作为普通子目录跟踪，不保留上游 .git。

experiments/policy_fidelity/
  frozen-theta policy fidelity 实验：
  同一个 base model 参数不变，只替换 memory carrier，抓 logits / KL / overlap / margin。

experiments/mechanistic_paths/
  MemGen、Retrieval + trained weaver、Retrieval + untrained weaver 的路径对比实验。
  里面包括 Appendix-E 风格 retrieval-weaver 注入、attention/logit-lens/hidden drift 分析。

docs/
  实验计划、运行报告、结论文档。

results/policy_fidelity/
  policy fidelity raw result、summary 和 PNG 图。

results/mechanistic_paths/
  mechanistic paths raw result、summary 和 PNG 图。

*.py, *.sh at repo root
  早期 Search-R1 / MemGen smoke test、trajectory builder、controller/reranker 等脚本。
```

## 远端 Python 环境

默认使用远端已有环境：

```bash
source /root/autodl-tmp/envs/memgen/bin/activate
```

也可以显式调用：

```bash
/root/autodl-tmp/envs/memgen/bin/python ...
```

每次跑实验前先设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export PYTHONPATH=/root/autodl-tmp/memrag_new/MemGen:$PYTHONPATH
```

原因：

- 远端机器没有稳定 VPN，Hugging Face 需要走 `hf-mirror.com`。
- MemGen 不是 pip install 的包，脚本需要通过 `PYTHONPATH` 找到 `MemGen/`。

快速检查：

```bash
cd /root/autodl-tmp/memrag_new
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export PYTHONPATH=/root/autodl-tmp/memrag_new/MemGen:$PYTHONPATH

/root/autodl-tmp/envs/memgen/bin/python -m py_compile \
  experiments/mechanistic_paths/run_mechanistic_paths.py \
  experiments/mechanistic_paths/analyze_mechanistic_paths.py \
  experiments/mechanistic_paths/analyze_appendix_e_large.py \
  experiments/policy_fidelity/run_policy_fidelity.py
```

## 外部资产位置

这些不进 git，但实验会用到。

Qwen2.5-1.5B MemGen checkpoint：

```text
/root/autodl-tmp/memrag_new/MemGen_checkpoints/Kana-s_MemGen/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model
```

SmolLM3-3B base model：

```text
/root/autodl-tmp/models/HuggingFaceTB/SmolLM3-3B
```

SmolLM3-3B MemGen checkpoint：

```text
/root/autodl-tmp/memrag_new/MemGen_checkpoints/Kana-s_MemGen/SmolLM3-3B/triviaqa/weaver-sft/pn=8_pl=4_in=0_il=4
```

注意 SmolLM3 这个 checkpoint 的结构：

- reasoner: SmolLM3-3B
- weaver: Qwen2.5-1.5B-Instruct hidden dim
- trigger: Qwen2.5-1.5B-Instruct hidden dim
- `latent_len=4`

所以跑 SmolLM3 时必须显式传：

```bash
--model /root/autodl-tmp/models/HuggingFaceTB/SmolLM3-3B
--weaver-model Qwen/Qwen2.5-1.5B-Instruct
--trigger-model Qwen/Qwen2.5-1.5B-Instruct
--latent-len 4
```

Qwen2.5-1.5B MemGen 跑法：

```bash
--model Qwen/Qwen2.5-1.5B-Instruct
--latent-len 8
```

## 主要实验 1：Appendix-E retrieval-weaver mechanistic paths

目标：比较三条路径。

1. `MemGen`
   - 用 checkpoint 里的 trained weaver 根据原始 prompt/state 生成 latent。
2. `Retrieval + trained weaver`
   - 从当前 state 派生 retrieval query。
   - 从外部 memory bank 检索经验文本。
   - 把当前 state embedding 和 retrieved memory embedding 拼接后送入 trained weaver。
   - reasoner 只看到生成出来的 latent，不直接看到检索文本。
3. `Retrieval + untrained weaver`
   - retrieval 一样。
   - 但 weaver 使用未训练初始化，用来区分“检索内容本身”和“训练过的 latent transformer”。

核心脚本：

```text
experiments/mechanistic_paths/run_mechanistic_paths.py
experiments/mechanistic_paths/analyze_mechanistic_paths.py
experiments/mechanistic_paths/analyze_appendix_e_large.py
```

### Qwen1.5B n=100

```bash
cd /root/autodl-tmp/memrag_new
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export PYTHONPATH=/root/autodl-tmp/memrag_new/MemGen:$PYTHONPATH

/root/autodl-tmp/envs/memgen/bin/python experiments/mechanistic_paths/run_mechanistic_paths.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --memgen-load-model-path /root/autodl-tmp/memrag_new/MemGen_checkpoints/Kana-s_MemGen/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model \
  --latent-len 8 \
  --retrieval-fusion appendix_e_embedding \
  --num-samples 100 \
  --memory-bank-size 2000 \
  --collect-attentions \
  --max-new-tokens 32 \
  --max-retrieval-memory-tokens 384 \
  --output results/mechanistic_paths/qwen15_appendix_e_memgen_expel_paths_n100.jsonl
```

如果担心 SSH 断开，用后台方式：

```bash
nohup /root/autodl-tmp/envs/memgen/bin/python experiments/mechanistic_paths/run_mechanistic_paths.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --memgen-load-model-path /root/autodl-tmp/memrag_new/MemGen_checkpoints/Kana-s_MemGen/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model \
  --latent-len 8 \
  --retrieval-fusion appendix_e_embedding \
  --num-samples 100 \
  --memory-bank-size 2000 \
  --collect-attentions \
  --max-new-tokens 32 \
  --max-retrieval-memory-tokens 384 \
  --output results/mechanistic_paths/qwen15_appendix_e_memgen_expel_paths_n100.jsonl \
  > results/mechanistic_paths/qwen15_appendix_e_n100.run.log 2>&1 < /dev/null &

tail -f results/mechanistic_paths/qwen15_appendix_e_n100.run.log
```

### SmolLM3-3B n=100

```bash
cd /root/autodl-tmp/memrag_new
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export PYTHONPATH=/root/autodl-tmp/memrag_new/MemGen:$PYTHONPATH

/root/autodl-tmp/envs/memgen/bin/python experiments/mechanistic_paths/run_mechanistic_paths.py \
  --model /root/autodl-tmp/models/HuggingFaceTB/SmolLM3-3B \
  --weaver-model Qwen/Qwen2.5-1.5B-Instruct \
  --trigger-model Qwen/Qwen2.5-1.5B-Instruct \
  --memgen-load-model-path /root/autodl-tmp/memrag_new/MemGen_checkpoints/Kana-s_MemGen/SmolLM3-3B/triviaqa/weaver-sft/pn=8_pl=4_in=0_il=4 \
  --latent-len 4 \
  --retrieval-fusion appendix_e_embedding \
  --num-samples 100 \
  --memory-bank-size 2000 \
  --collect-attentions \
  --max-new-tokens 32 \
  --max-retrieval-memory-tokens 384 \
  --output results/mechanistic_paths/smollm3_appendix_e_memgen_expel_paths_n100.jsonl
```

后台方式：

```bash
nohup /root/autodl-tmp/envs/memgen/bin/python experiments/mechanistic_paths/run_mechanistic_paths.py \
  --model /root/autodl-tmp/models/HuggingFaceTB/SmolLM3-3B \
  --weaver-model Qwen/Qwen2.5-1.5B-Instruct \
  --trigger-model Qwen/Qwen2.5-1.5B-Instruct \
  --memgen-load-model-path /root/autodl-tmp/memrag_new/MemGen_checkpoints/Kana-s_MemGen/SmolLM3-3B/triviaqa/weaver-sft/pn=8_pl=4_in=0_il=4 \
  --latent-len 4 \
  --retrieval-fusion appendix_e_embedding \
  --num-samples 100 \
  --memory-bank-size 2000 \
  --collect-attentions \
  --max-new-tokens 32 \
  --max-retrieval-memory-tokens 384 \
  --output results/mechanistic_paths/smollm3_appendix_e_memgen_expel_paths_n100.jsonl \
  > results/mechanistic_paths/smollm3_appendix_e_n100.run.log 2>&1 < /dev/null &

tail -f results/mechanistic_paths/smollm3_appendix_e_n100.run.log
```

### 分析和出图

单模型分析：

```bash
/root/autodl-tmp/envs/memgen/bin/python experiments/mechanistic_paths/analyze_mechanistic_paths.py \
  --input results/mechanistic_paths/qwen15_appendix_e_memgen_expel_paths_n100.jsonl \
  --out-dir results/mechanistic_paths/figures_qwen15_appendix_e_n100 \
  --summary-output results/mechanistic_paths/mechanistic_summary_qwen15_appendix_e_n100.json \
  --report-output docs/mechanistic_paths_qwen15_appendix_e_n100_report.md

/root/autodl-tmp/envs/memgen/bin/python experiments/mechanistic_paths/analyze_mechanistic_paths.py \
  --input results/mechanistic_paths/smollm3_appendix_e_memgen_expel_paths_n100.jsonl \
  --out-dir results/mechanistic_paths/figures_smollm3_appendix_e_n100 \
  --summary-output results/mechanistic_paths/mechanistic_summary_smollm3_appendix_e_n100.json \
  --report-output docs/mechanistic_paths_smollm3_appendix_e_n100_report.md
```

大样本配对统计：

```bash
/root/autodl-tmp/envs/memgen/bin/python experiments/mechanistic_paths/analyze_appendix_e_large.py \
  --input Qwen1.5B=results/mechanistic_paths/qwen15_appendix_e_memgen_expel_paths_n100.jsonl \
  --input SmolLM3-3B=results/mechanistic_paths/smollm3_appendix_e_memgen_expel_paths_n100.jsonl \
  --out-dir results/mechanistic_paths/figures_appendix_e_large_n100 \
  --summary-output results/mechanistic_paths/appendix_e_large_n100_summary.json \
  --report-output docs/mechanistic_paths_appendix_e_large_n100_report.md
```

当前已经跑完的关键结果：

```text
results/mechanistic_paths/qwen15_appendix_e_memgen_expel_paths_n100.jsonl
results/mechanistic_paths/smollm3_appendix_e_memgen_expel_paths_n100.jsonl
results/mechanistic_paths/appendix_e_large_n100_summary.json
docs/mechanistic_paths_appendix_e_large_n100_report.md
```

当前 n=100 + n=100 的核心结论：

- `Retrieval + trained weaver` 相对 `MemGen` 的 gold prob / gold margin 在两个 base model 上都提高。
- Pooled-200:
  - gold prob delta = `+0.0294`, 95% CI `[0.0197, 0.0391]`
  - gold margin delta = `+0.2156`, 95% CI `[0.1360, 0.2964]`
- `Retrieval + untrained weaver` 有时提高 gold prob，但更像强扰动；trained weaver 的稳定信号是更低 entropy、更可控的 policy modulation。

## 主要实验 2：Policy Fidelity / Memory Compression

目标：冻结 base model 参数，只改变 memory carrier，观察首 token policy 分布如何漂移。

核心问题：

1. 压缩是否改变策略。
2. latent 是否只是随机扰动。
3. 显式压缩文本和 latent 压缩谁更保留 memory policy effect。
4. memory 评估指标不只看重建质量，而看 policy KL / top-k overlap / margin / recovery ratio。

核心脚本：

```text
experiments/policy_fidelity/run_policy_fidelity.py
experiments/policy_fidelity/memory_carriers.py
experiments/policy_fidelity/memgen_injector.py
experiments/policy_fidelity/analyze_policy_fidelity.py
experiments/policy_fidelity/render_conclusion_figures.py
```

七臂设置：

```text
no_memory
full_text
extractive_text
summary_text
latent_compressed
random_latent
fixed_soft_prompt
```

已经生成的结果：

```text
results/policy_fidelity/m0_tiny_policy.jsonl
results/policy_fidelity/m1_text_qwen15.jsonl
results/policy_fidelity/m2_latent_qwen15.jsonl
results/policy_fidelity/policy_fidelity_summary.json
docs/policy_fidelity_experiment_report.md
```

典型运行环境：

```bash
cd /root/autodl-tmp/memrag_new
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export PYTHONPATH=/root/autodl-tmp/memrag_new/MemGen:$PYTHONPATH
```

M0 tiny sanity：

```bash
/root/autodl-tmp/envs/memgen/bin/python experiments/policy_fidelity/run_policy_fidelity.py \
  --model llamafactory/tiny-random-qwen2.5 \
  --limit 5 \
  --output results/policy_fidelity/m0_tiny_policy.jsonl
```

M1/M2 用 Qwen2.5-1.5B：

```bash
/root/autodl-tmp/envs/memgen/bin/python experiments/policy_fidelity/run_policy_fidelity.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --limit 100 \
  --output results/policy_fidelity/m1_text_qwen15.jsonl
```

具体参数以后以脚本 `--help` 为准：

```bash
/root/autodl-tmp/envs/memgen/bin/python experiments/policy_fidelity/run_policy_fidelity.py --help
```

## Search-R1 / FlashRAG smoke 相关脚本

早期多轮检索 smoke test 用到了这些脚本：

```text
start_searchr1_bm25_server.sh
run_memgen_searchr1_real_smoke.py
flashrag_memgen_multiturn_smoke.py
memgen_experience_latent_smoke.py
memgen_text_vs_latent_large_eval.py
```

如果要重启 Search-R1 BM25 server：

```bash
cd /root/autodl-tmp/memrag_new
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export PYTHONPATH=/root/autodl-tmp/memrag_new/MemGen:$PYTHONPATH

bash start_searchr1_bm25_server.sh
```

注意：

- `flashrag_data/` 不进 git。
- BM25 corpus/index 如果本地不存在，需要按 Search-R1/FlashRAG 的数据流程重新准备。
- 当前最终 mechanistic paths 实验不依赖在线 Search-R1 loop，它使用脚本内 memory bank / retrieval-weaver 路径。

## 数据和中间结果

本仓库提交了能支撑报告复核的 raw/summary/figure 文件。

没有提交：

- 大模型权重。
- Hugging Face cache。
- FlashRAG 原始数据。
- BM25 index。
- controller 的 `.pt/.pkl` 训练缓存。

如果要完整复现从数据下载开始，需要额外准备这些外部资源。

## 常见问题

### 1. Hugging Face 下载超时

先确认：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
```

如果第一次 `datasets` metadata 访问 timeout，通常会自动 retry。不要马上中断。

### 2. SmolLM3 维度报错

SmolLM3 checkpoint 不是纯 SmolLM3 全套组件，而是：

```text
SmolLM3 reasoner + Qwen2.5-1.5B weaver/trigger
```

必须带：

```bash
--weaver-model Qwen/Qwen2.5-1.5B-Instruct
--trigger-model Qwen/Qwen2.5-1.5B-Instruct
--latent-len 4
```

### 3. SSH 断开导致实验中断

长任务用 `nohup`：

```bash
nohup /root/autodl-tmp/envs/memgen/bin/python ... \
  > results/mechanistic_paths/my_run.run.log 2>&1 < /dev/null &

tail -f results/mechanistic_paths/my_run.run.log
```

当前 `run_mechanistic_paths.py` 是跑完后统一写 JSONL，所以中途断进程可能没有完整 raw output。

### 4. GitHub repo not found

正确 remote 是：

```bash
git@github.com:orville-wang/memrag_new.git
```

不是：

```bash
git@github.com:ao-wang-orville/memrag_new.git
```

### 5. 提交前如何确认没有大文件

```bash
git diff --cached --name-only | grep -E '(MemGen_checkpoints|flashrag_data|__pycache__|\\.tgz$|\\.tar\\.gz$|\\.pt$|\\.pth$|\\.pkl$|\\.safetensors$|\\.bin$|\\.ckpt$)' || true
```

正常情况下不应该输出需要提交的大运行环境文件。

## 下次接手建议顺序

1. 登录远端。
2. 进入仓库并拉最新代码。

```bash
cd /root/autodl-tmp/memrag_new
git pull --ff-only origin main
```

3. 设置环境变量。

```bash
source /root/autodl-tmp/envs/memgen/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-cache
export PYTHONPATH=/root/autodl-tmp/memrag_new/MemGen:$PYTHONPATH
```

4. 先读报告。

```bash
sed -n '1,220p' docs/mechanistic_paths_appendix_e_large_n100_report.md
sed -n '1,220p' docs/policy_fidelity_experiment_report.md
```

5. 如果只是复核图和数字，直接看：

```bash
results/mechanistic_paths/appendix_e_large_n100_summary.json
results/mechanistic_paths/figures_appendix_e_large_n100/
results/policy_fidelity/policy_fidelity_summary.json
results/policy_fidelity/figures/
```

6. 如果要继续跑 mechanistic paths，先做 n=2 dry run，再跑 n=100。

7. 跑完后重新分析、提交 report/summary/figures/raw。

```bash
git status --short
git add docs experiments results README.md
git commit -m "..."
git push origin main
```

## 当前最重要的结论

这个仓库当前支持的最稳结论是：

`Retrieval + trained weaver` 相比原版 MemGen，在 Qwen1.5B 和 SmolLM3-3B 两个 base model 上都能稳定提高 candidate gold probability 和 gold-vs-distractor margin；但是它不总是压过 untrained weaver 的 gold probability。真正更稳定的区别在于 trained weaver 让 policy 分布更可控、entropy 更低，因此更像一种可学习的 latent policy modulation，而不是单纯靠检索文本或随机 latent 扰动。

