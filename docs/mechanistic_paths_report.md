# MemGen / Retrieval-Conditioned Weaver 作用路径对比实验

样本数：30。三条主路径共享同一个可见 reasoner prompt、同一批 retrieved experience、同一组候选动作；差别只在 latent memory 的生成方式。

## 三条路径

- **MemGen**：训练后的 MemGen weaver 只读取当前 prompt，生成 latent memory。
- **Retrieval + trained weaver**：先从 ExpeL-like memory bank 检索显式经验文本，再把 `prompt + retrieved experience` 输入训练后的 weaver，reasoner 只看到 latent。
- **Retrieval + untrained weaver**：检索文本完全相同，但 weaver/projection/latent query 是未训练初始化。

## 主要数字

| Path | Gold prob | Search prob | Gold margin | Search-vs-answer margin | Entropy |
|---|---:|---:|---:|---:|---:|
| MemGen | 0.1928 | 0.7229 | 1.0207 | 1.5200 | 3.0577 |
| Retrieval + trained weaver | 0.2365 | 0.6779 | 1.3447 | 1.2007 | 3.8276 |
| Retrieval + untrained weaver | 0.1811 | 0.7307 | 1.0699 | 1.6096 | 5.2070 |

## 层级作用路径读数

| Path | Final latent attention | Final hidden L2 | Final search-answer logit margin | Final gold-distractor logit margin |
|---|---:|---:|---:|---:|
| MemGen | 0.5672 | 184.5274 | 1.1417 | 1.7303 |
| Retrieval + trained weaver | 0.4337 | 165.8412 | 0.2042 | 3.6297 |
| Retrieval + untrained weaver | 0.1925 | 195.5944 | 1.0500 | 2.4202 |

| Comparison | Next-token KL |
|---|---:|
| expel_trained_to_expel_untrained | 10.6033 |
| memgen_to_expel_trained | 6.6128 |
| memgen_to_expel_untrained | 8.7458 |
| no_latent_to_expel_trained | 5.5110 |
| no_latent_to_expel_untrained | 12.2681 |
| no_latent_to_memgen | 11.1987 |

## 图表

![candidate_policy_probability.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_qwen15_n30_v2/candidate_policy_probability.png)

![next_token_policy_kl.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_qwen15_n30_v2/next_token_policy_kl.png)

![layer_hidden_l2_to_no_latent.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_qwen15_n30_v2/layer_hidden_l2_to_no_latent.png)

![latent_attention_mass_by_layer.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_qwen15_n30_v2/latent_attention_mass_by_layer.png)

![logit_lens_search_answer_margin.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_qwen15_n30_v2/logit_lens_search_answer_margin.png)

![logit_lens_gold_distractor_margin.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_qwen15_n30_v2/logit_lens_gold_distractor_margin.png)

![generation_prefix_mix.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_qwen15_n30_v2/generation_prefix_mix.png)

## 结论

1. **retrieval-conditioned trained weaver 确实产生了不同于原 MemGen 的 policy shift。** 相比 MemGen，Retrieval + trained weaver 的 gold candidate probability 平均提高 +0.0437，gold-vs-distractor margin 也更高；两者 next-token KL 为 6.6128，说明检索经验经过 trained weaver 后不是等价于原始 MemGen latent。
2. **untrained weaver 证明“检索文本进入 weaver”本身不够。** 它读到同一批 retrieved experience，但 gold probability 比 trained weaver 低 0.0554，next-token entropy 比 trained weaver 高 1.3794，trained-vs-untrained KL 为 10.6033。这更像无序扰动，而不是稳定的经验利用。
3. **层级指标比最终答案更能解释作用路径。** latent attention mass 表示 reasoner 哪些层直接读取 latent slots；hidden L2 表示 latent 对决策状态的扰动强度；logit-lens action-key probe 在强制前缀 `<` 后比较 `search` vs `answer`，answer-content probe 在 `<answer> ` 后比较 gold vs distractor。
4. **当前实验仍是 prompt-level augmentation，不等价于完整在线 Search-R1 多轮 agent。** 它回答的是 frozen reasoner 下 memory carrier 如何改变 token policy 和 hidden trajectory，不能单独证明最终多轮检索收益。

## Raw Files

- Raw JSONL: `/root/autodl-tmp/memrag_new/results/mechanistic_paths/qwen15_memgen_expel_paths_n30_v2.jsonl`
- Summary JSON: `/root/autodl-tmp/memrag_new/results/mechanistic_paths/mechanistic_summary_qwen15_n30_v2.json`
