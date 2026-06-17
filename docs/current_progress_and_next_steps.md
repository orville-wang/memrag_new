# 当前进展与下一步计划

更新时间：2026-06-17

本文档用于记录 `memrag_new` 当前已经完成的工作、已有实验结论、代码/结果位置，以及接下来最值得继续推进的 TODO。  
如果下次重新接手，建议先读顶层 `README.md`，再读本文档。

## 0. 项目当前目标

我们目前不是在做一个通用 RAG benchmark，而是在验证一个更具体的问题：

> 给定相同输入 `x` 和相同外部经验/记忆 `m`，不同 memory carrier 会如何改变 frozen 或近似 frozen LLM policy？

当前重点包括：

- 显式 retrieval 继续作为 evidence path。
- latent memory / MemGen-style latent input experience 不直接替代 evidence，而是影响 search-control / policy prior。
- 对比显式文本、压缩文本、latent 压缩、随机 latent、untrained weaver、trained weaver 等 carrier 的 policy effect。
- 尽量从 token distribution、KL、margin、entropy、attention、hidden drift 等角度解释“为什么策略发生变化”。

## 1. 仓库与远端状态

远端主目录：

```bash
cd /root/autodl-tmp/memrag_new
```

GitHub remote：

```bash
git@github.com:orville-wang/memrag_new.git
```

当前顶层 git 已经管理整个工作区：

- `MemGen/` 已去掉自己的 `.git`，作为普通子目录被顶层 git 跟踪。
- `Search-R1/` 已去掉自己的 `.git`，作为普通子目录被顶层 git 跟踪。
- 大模型权重、Hugging Face cache、FlashRAG data、tar 包、run log、pycache 不提交。

最近关键 commits：

```text
d8ee58a Document experiment startup workflow
fbfb720 Include remaining MemGen and Search-R1 source assets
3e66394 Track MemGen Search-R1 and experiment artifacts
291b080 Add MemGen retrieval-weaver mechanistic experiments
```

## 2. 已完成工作概览

### 2.1 基础工程整理

已经完成：

- 克隆并配置 MemGen。
- 克隆并配置 Search-R1。
- 把两个子仓库改成顶层 git 的普通子目录。
- 配置远端 5090 环境：
  - `HF_ENDPOINT=https://hf-mirror.com`
  - `HF_HOME=/root/autodl-tmp/hf-cache`
  - `PYTHONPATH=/root/autodl-tmp/memrag_new/MemGen:$PYTHONPATH`
- 顶层 README 已写成详细启动手册。

重要文档：

```text
README.md
docs/repo_reuse_and_bench_guide.md
docs/experiment_design_v0.1.md
```

### 2.2 Search-R1 / FlashRAG 多轮 smoke 探索

早期我们尝试用 Search-R1 + FlashRAG 跑多轮检索 smoke test，目标是观察 latent experience 是否能减少重复搜索、提升 action-format 稳定性、让多轮检索更少走弯路。

相关脚本：

```text
flashrag_memgen_multiturn_smoke.py
memgen_experience_latent_smoke.py
memgen_text_vs_latent_large_eval.py
run_memgen_searchr1_real_smoke.py
start_searchr1_bm25_server.sh
```

主要结论：

- latent experience 对 search-control 行为有影响。
- 比较稳定的收益是减少重复检索、减少 malformed action、降低跑到 max-turn 的概率。
- 但 supporting-fact coverage / final answer 暂时没有稳定提升。
- 显式 experience prompt 对小模型会明显破坏 action grammar，不能简单当作好 baseline。

当前定位：

- Search-R1 在线多轮 smoke 已经不是当前主证据。
- 它作为背景结果保留。
- 当前主线已经转向 policy fidelity 和 mechanistic paths。

### 2.3 Policy Fidelity / Memory Compression 实验

目标：

在 frozen-theta setting 下，固定模型参数，只替换 memory carrier，观察 policy distribution 如何变化。

已实现目录：

```text
experiments/policy_fidelity/
```

核心脚本：

```text
experiments/policy_fidelity/run_policy_fidelity.py
experiments/policy_fidelity/memory_carriers.py
experiments/policy_fidelity/memgen_injector.py
experiments/policy_fidelity/analyze_policy_fidelity.py
experiments/policy_fidelity/render_conclusion_figures.py
```

实验设定：

- 模型：Qwen2.5-1.5B-Instruct。
- 数据：TriviaQA 50 + GSM8K 50，总计 n=100。
- 对照 arms：
  - `no_memory`
  - `full_text`
  - `extractive_text`
  - `summary_text`
  - `latent_compressed`
  - `random_latent`
  - `fixed_soft_prompt`

主要结果文件：

```text
results/policy_fidelity/m0_tiny_policy.jsonl
results/policy_fidelity/m1_text_qwen15.jsonl
results/policy_fidelity/m2_latent_qwen15.jsonl
results/policy_fidelity/policy_fidelity_summary.json
results/policy_fidelity/figures/
docs/policy_fidelity_experiment_report.md
```

当前关键数值：

```text
full_text:
  gold_prob_mean = 0.7210
  gold_margin_mean = 1.3807
  top10_overlap_with_full = 1.0000

summary_text:
  gold_prob_mean = 0.4189
  gold_margin_mean = -0.7692
  top10_overlap_with_full = 0.7190

extractive_text:
  gold_prob_mean = 0.3875
  gold_margin_mean = -1.1058
  top10_overlap_with_full = 0.7830

no_memory:
  gold_prob_mean = 0.3702
  gold_margin_mean = -1.3045

latent_compressed:
  gold_prob_mean = 0.1112
  gold_margin_mean = -3.2233
  top10_overlap_with_full = 0.0000

random_latent:
  gold_prob_mean = 0.2035
  gold_margin_mean = -2.3666
  top10_overlap_with_full = 0.0030
```

当前解释：

- 显式 full text 对 policy 的影响最大。
- summary / extractive text 能保留一部分 policy effect。
- 当前 naive latent compression 没有保留 full text policy effect，甚至比 no-memory 更差。
- 这支持一个重要判断：memory compression 不能只看重建质量或压缩率，而要看 policy fidelity。
- 当前 latent_compressed 还不是 MemGen-style trained latent experience，只是 frozen-carrier baseline；不能把它当作最终 latent memory 方法效果。

### 2.4 Appendix-E Retrieval-Weaver Mechanistic Paths 实验

这是当前最重要的一组实验。

目标：

对比三条路径：

1. `MemGen`
   - 原版 MemGen trained weaver，根据 prompt/state 生成 latent。
2. `Retrieval + trained weaver`
   - 从当前 state 派生 retrieval query。
   - 检索外部经验文本。
   - 把 state embedding 和 retrieved memory embedding 一起送入 trained weaver。
   - reasoner 只看到 latent，不直接看到检索文本。
3. `Retrieval + untrained weaver`
   - retrieval 相同。
   - weaver 是未训练初始化，用来区分“检索带来的信息”和“训练过的 latent 变换器”。

核心代码：

```text
experiments/mechanistic_paths/run_mechanistic_paths.py
experiments/mechanistic_paths/analyze_mechanistic_paths.py
experiments/mechanistic_paths/analyze_appendix_e_large.py
```

已完成 n=100 + n=100：

```text
Qwen2.5-1.5B-Instruct:
  results/mechanistic_paths/qwen15_appendix_e_memgen_expel_paths_n100.jsonl

SmolLM3-3B:
  results/mechanistic_paths/smollm3_appendix_e_memgen_expel_paths_n100.jsonl
```

主要报告：

```text
docs/mechanistic_paths_appendix_e_large_n100_report.md
docs/mechanistic_paths_qwen15_appendix_e_n100_report.md
docs/mechanistic_paths_smollm3_appendix_e_n100_report.md
```

大样本配对统计：

```text
results/mechanistic_paths/appendix_e_large_n100_summary.json
results/mechanistic_paths/figures_appendix_e_large_n100/
```

Pooled-200 结果：

```text
MemGen:
  gold = 0.1486
  margin = 0.8188
  entropy = 4.5990

Retrieval + trained weaver:
  gold = 0.1779
  margin = 1.0344
  entropy = 4.0543

Retrieval + untrained weaver:
  gold = 0.1808
  margin = 0.9795
  entropy = 5.4235
```

关键 paired delta：

```text
Trained - MemGen:
  gold delta = +0.0294
  95% CI = [0.0197, 0.0391]

Trained - MemGen:
  margin delta = +0.2156
  95% CI = [0.1360, 0.2964]

Trained - Untrained:
  entropy delta = -1.3692
  95% CI = [-1.5810, -1.1584]
```

当前结论：

- `Retrieval + trained weaver` 相对 `MemGen` 的正向信号比较稳定。
- 它在 Qwen1.5B 和 SmolLM3-3B 两个 base model 上都提高 gold probability 和 gold-vs-distractor margin。
- `Retrieval + untrained weaver` 在 SmolLM3 上有时 gold probability 更高，但 entropy 明显更高。
- 因此不能只看 gold probability；untrained weaver 更像强扰动或偶然命中。
- trained weaver 更稳定的作用路径是降低 entropy、让 policy 更可控，符合 latent policy modulation 的解释。

## 3. 目前已有图表

Policy fidelity 图：

```text
results/policy_fidelity/figures/policy_kl_bar.png
results/policy_fidelity/figures/overlap_recovery.png
results/policy_fidelity/figures/gold_margin_forest.png
results/policy_fidelity/figures/policy_drift_distribution.png
results/policy_fidelity/figures/conclusion_latent_controls.png
results/policy_fidelity/figures/conclusion_policy_fidelity_map.png
results/policy_fidelity/figures/conclusion_recovery_ladder.png
```

Mechanistic paths 图：

```text
results/mechanistic_paths/figures_appendix_e_large_n100/appendix_e_large_metric_bars.png
results/mechanistic_paths/figures_appendix_e_large_n100/appendix_e_large_paired_delta_forest.png

results/mechanistic_paths/figures_qwen15_appendix_e_n100/
results/mechanistic_paths/figures_smollm3_appendix_e_n100/
```

图表当前足够支持内部讨论和草稿写作，但还不是最终论文图版本。后续还需要统一：

- 字体。
- 颜色。
- figure caption。
- 统计标注。
- 图的主次关系。

## 4. 当前已经解决的问题

### 4.1 “显式文本注入是不是好替代？”

目前看不是。

在早期 Search-R1 smoke 中，显式 experience text 会破坏 action grammar，尤其对小模型影响很大。它虽然可能降低搜索次数，但这不是有效检索控制，而是提前乱停或格式损坏。

### 4.2 “latent memory 是不是 hidden evidence？”

当前结果更支持：

> latent memory 更像 policy prior / control signal，而不是可直接审计的 hidden evidence。

原因：

- supporting-fact coverage / final answer 没有稳定提升。
- action-format、entropy、重复搜索等 control 指标更容易被影响。
- retrieval + trained weaver 的主要稳定信号是 margin 和 entropy，而不是直接给出 evidence。

### 4.3 “untrained weaver 为什么有时也好？”

当前解释：

- untrained weaver 不是有效 memory utilization 的证据。
- 它可能通过高范数 latent 或随机扰动改变模型分布。
- 在某些样本上会偶然提高候选答案概率。
- 但它的 entropy 更高、稳定性更差，不能只用 gold prob 判断。

### 4.4 “为什么要看 policy fidelity，而不是重建质量？”

因为 memory 的目标不是还原文本，而是让模型在给定 `x` 时产生更接近目标策略的行为。

更应该比较：

```text
pi(. | x)
pi(. | x, m)
pi(. | x, compress(m))
oracle_pi(. | x)
```

而不是只比较：

```text
m vs reconstruct(compress(m))
```

## 5. 剩余 TODO

下面按优先级排序。

### P0. 复现实验环境固化

目标：

让下一次启动不会依赖记忆。

TODO：

- 写一个 `scripts/setup_remote_env.sh`，统一 export：
  - `HF_ENDPOINT`
  - `HF_HOME`
  - `PYTHONPATH`
- 写一个 `scripts/check_assets.sh`，检查：
  - Qwen checkpoint 是否存在。
  - SmolLM3 base model 是否存在。
  - SmolLM3 MemGen checkpoint 是否存在。
  - `memgen` Python env 是否可用。
- 给 `run_mechanistic_paths.py` 增加一个 `--quick-check` 或独立 sanity 脚本，跑 n=1 检查 shape/device/dtype。

验收：

```bash
bash scripts/check_assets.sh
bash scripts/run_mechanistic_paths_dryrun.sh
```

### P0. 明确实验主线和 paper story

当前 story 应该收敛为：

1. memory compression 的评价目标应该是 policy fidelity，而不是 reconstruction fidelity。
2. naive latent compression 不能自动保留 policy effect。
3. retrieval-conditioned trained weaver 可以把 retrieved experience 转成更可控的 latent policy modulation。
4. untrained weaver 可作为强扰动对照，说明“有变化”不等于“有用的 memory utilization”。

TODO：

- 把 `docs/mechanistic_paths_appendix_e_large_n100_report.md` 和 `docs/policy_fidelity_experiment_report.md` 合并成一份论文草稿式 narrative。
- 明确每张图服务哪个 claim。
- 删除或弱化早期 Search-R1 smoke 中不稳定的 claim。

### P1. 扩大 Appendix-E mechanistic paths 样本量

当前 n=100 + n=100 已经比 n=30 稳定，但如果要写论文，建议扩大。

TODO：

- Qwen1.5B: n=300 或 n=500。
- SmolLM3-3B: n=300 或 n=500。
- 如果时间允许，加第三个模型或另一套 checkpoint。

重点看：

- `Trained - MemGen` 的 gold / margin CI 是否继续不跨 0。
- `Trained - Untrained` 的 entropy advantage 是否继续稳定。
- untrained 的 gold prob 是否仍然不稳定或模型依赖强。

验收：

```text
results/mechanistic_paths/*_n300.jsonl
results/mechanistic_paths/appendix_e_large_n300_summary.json
docs/mechanistic_paths_appendix_e_large_n300_report.md
```

### P1. 加 seeds / bootstrap / permutation test

当前已有 bootstrap CI，但还可以更严谨。

TODO：

- 对 retrieval memory bank sampling 加不同 seed。
- 对 untrained weaver 初始化加不同 seed。
- 报告 paired bootstrap + paired permutation test。
- 统计每个样本 delta 的 win rate / lose rate / tie rate。

目的：

- 排除单一 untrained 初始化带来的偶然性。
- 排除 memory bank 构造偶然性。

### P1. 加 retrieval quality 分层分析

现在只看总体均值，还没区分 retrieval 是否真的命中有用经验。

TODO：

- 给每条样本记录 retrieval score、retrieved memory 文本、query。
- 标注 retrieved memory 是否含 gold answer / supporting entity / relevant clue。
- 按 retrieval quality 分桶：
  - high-quality retrieval
  - medium retrieval
  - noisy retrieval
  - no-hit retrieval
- 分别计算 trained vs MemGen / trained vs untrained。

想回答的问题：

- trained weaver 的收益是否主要出现在 high-quality retrieval？
- noisy retrieval 下 trained weaver 是否比 untrained 更能抑制错误扰动？

### P1. 做更强的机制分析

当前已经有：

- next-token KL。
- candidate probability。
- gold-vs-distractor margin。
- entropy。
- latent attention mass by layer。
- layer hidden L2 drift。
- logit lens margin。

但还缺 causal evidence。

TODO：

- activation patching：
  - 把 `Retrieval + trained` 的某层 hidden patch 到 `MemGen`。
  - 看 gold prob / margin 是否恢复。
- layer ablation：
  - mask latent tokens at selected layer。
  - 比较 early/mid/late layer 对 policy effect 的贡献。
- latent norm control：
  - 把 trained/untrained latent norm 对齐。
  - 检查 untrained 的收益是否只是 norm 或 distribution shift。
- attention mediation：
  - 看 latent attention mass 是否解释 entropy/margin delta。

目标：

找到更接近“中介变量”的解释，而不是只报告外部指标。

### P1. 重新设计 MemGen-style latent input experience 训练

当前 Appendix-E 实验使用已有 trained weaver checkpoint，不是专门为 HotpotQA/2Wiki/Search-R1 trajectory 训练的 retrieval-conditioned weaver。

TODO：

- 构造 state-level trajectory samples：
  - `state`
  - `retrieved experience`
  - `next useful action`
  - `candidate action scores`
- 训练 composer/weaver 直接服务 Search-R1 action grammar：
  - `Search[...]`
  - `Answer[...]`
  - stop policy
- 不追加第二段 latent，保持 single prompt latent blend。
- 训练目标先用 SFT / ranking，不急着做 GRPO。

验收：

- 同一 Search-R1 logger 下：
  - avg searches 不升。
  - first support hit 更早。
  - duplicate/no-hit 更少。
  - malformed action 更少。
  - final answer 不掉。

### P2. 改善 policy fidelity latent baseline

当前 `latent_compressed` 是 naive mean-pool latent，效果很差。这个结果有用，但不是最终方法。

TODO：

- 加 trained compressor baseline。
- 加 autoencoder-style latent compressor，但评价仍然看 policy fidelity。
- 加 BGE/Contriever dense embedding + projector。
- 加 MemGen weaver latent carrier。
- 把 `full_text -> compressed` 的 policy drift 做 per-token / per-candidate 分析。

核心问题：

> 什么样的 compressed memory 能保留 `pi(.|x,m)` 对 `pi(.|x)` 的有用改变？

### P2. 做任务级 final answer 评估

当前 mechanistic paths 更多是首 token / candidate action / margin 指标，还不是完整 answer accuracy。

TODO：

- 对每个 arm 做完整 generation。
- 对 TriviaQA / HotpotQA / 2Wiki 计算 EM / F1。
- 对 Search-R1 loop 计算：
  - answer accuracy
  - support coverage
  - avg searches
  - duplicate searches
  - malformed actions
  - max-turn rate

注意：

- final answer 不一定比 policy metric 更敏感。
- 但如果要写论文，至少要证明 policy improvement 不会损害任务结果。

### P2. 图表论文化

当前图能说明结论，但还需要统一论文风格。

TODO：

- 统一 Times New Roman。
- 统一配色。
- 图中减少解释性文字。
- 每个 figure 只服务一个 claim。
- 增加 caption 草稿。
- 产出 PDF/PNG 双格式。

建议最终图：

1. Policy fidelity framework diagram。
2. Carrier comparison bar/forest plot。
3. Appendix-E three-path comparison。
4. Trained vs untrained entropy/margin paired delta。
5. Layer-wise mediation / patching figure。

## 6. 当前不建议继续投入的方向

### 6.1 单纯扩大 Search-R1 smoke

原因：

- 在线 loop 噪声大。
- 当前 checkpoint 并不是为 HotpotQA/2Wiki action grammar 训练。
- 很容易混淆 retrieval quality、format stability、model size、latent effect。

建议：

- 先用 mechanistic / policy fidelity 把作用路径说清楚。
- 再回到 Search-R1 做任务级验证。

### 6.2 把 explicit experience prompt 当主要 baseline

原因：

- 对小模型 action grammar 破坏太强。
- avg searches 下降不代表更好，可能是提前乱停。

建议：

- explicit text 作为 diagnostic baseline。
- 不作为主要方法对照。

### 6.3 只用 reconstruction loss 评价 memory compression

原因：

- 我们关心的是 policy behavior，不是文本还原。
- 一个重建很好的 compressor 可能不保留对 action choice 有用的信息。

建议：

- reconstruction 只能作为辅助指标。
- 主指标仍然是 KL / overlap / margin / entropy / action recovery。

## 7. 下一次最合理的执行顺序

建议按这个顺序继续：

1. 固化 remote scripts：
   - `scripts/check_assets.sh`
   - `scripts/run_appendix_e_qwen_n100.sh`
   - `scripts/run_appendix_e_smollm3_n100.sh`
   - `scripts/analyze_appendix_e_large.sh`
2. 把 Appendix-E n=100 结果整理成论文 claim table。
3. 跑 n=300：
   - Qwen1.5B n=300。
   - SmolLM3-3B n=300。
4. 加 untrained weaver 多 seed。
5. 做 latent norm control。
6. 做 layer ablation / activation patching。
7. 如果机制结果稳定，再回到 Search-R1 在线 loop 做任务级验证。

## 8. 当前一句话总结

目前最稳的发现是：

> 检索经验文本本身不是关键；把检索经验通过 trained weaver 转成 latent 后，模型 policy 会出现更稳定、更低熵、更可控的改变。相反，untrained weaver 也能扰动 policy，甚至偶然提高 gold probability，但它缺少稳定的控制特征。因此，MemGen-style latent input experience 更像一种可学习的 policy modulation 机制，而不是 hidden evidence 或简单文本压缩。

