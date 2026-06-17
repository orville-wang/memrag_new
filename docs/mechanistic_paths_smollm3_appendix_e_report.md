# MemGen / Retrieval-Conditioned Weaver 作用路径对比实验

样本数：30。三条主路径共享同一个可见 reasoner prompt、同一批 retrieved experience、同一组候选动作；差别只在 latent memory 的生成方式。

## 三条路径

- **MemGen**：训练后的 MemGen weaver 只读取当前 prompt，生成 latent memory。
- **Retrieval + trained weaver**：先从 ExpeL-like memory bank 检索显式经验文本，再把 `prompt + retrieved experience` 输入训练后的 weaver，reasoner 只看到 latent。
- **Retrieval + untrained weaver**：检索文本完全相同，但 weaver/projection/latent query 是未训练初始化。

## 主要数字

| Path | Gold prob | Search prob | Gold margin | Search-vs-answer margin | Entropy |
|---|---:|---:|---:|---:|---:|
| MemGen | 0.1042 | 0.7079 | 1.1067 | 2.2573 | 5.9394 |
| Retrieval + trained weaver | 0.1410 | 0.6500 | 1.1147 | 1.9228 | 4.7529 |
| Retrieval + untrained weaver | 0.1851 | 0.5872 | 1.2045 | 1.3464 | 5.3911 |

## 层级作用路径读数

| Path | Final latent attention | Final hidden L2 | Final search-answer logit margin | Final gold-distractor logit margin |
|---|---:|---:|---:|---:|
| MemGen | 0.5502 | 47.4625 | 8.6708 | -0.0090 |
| Retrieval + trained weaver | 0.5143 | 47.4651 | 6.6542 | -0.0694 |
| Retrieval + untrained weaver | 0.1988 | 53.9482 | 4.1583 | 1.9805 |

| Comparison | Next-token KL |
|---|---:|
| expel_trained_to_expel_untrained | 16.4257 |
| memgen_to_expel_trained | 0.9767 |
| memgen_to_expel_untrained | 11.1277 |
| no_latent_to_expel_trained | 3.1437 |
| no_latent_to_expel_untrained | 22.6344 |
| no_latent_to_memgen | 2.9738 |

## 图表

![candidate_policy_probability.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_smollm3_appendix_e_n30/candidate_policy_probability.png)

![next_token_policy_kl.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_smollm3_appendix_e_n30/next_token_policy_kl.png)

![layer_hidden_l2_to_no_latent.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_smollm3_appendix_e_n30/layer_hidden_l2_to_no_latent.png)

![latent_attention_mass_by_layer.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_smollm3_appendix_e_n30/latent_attention_mass_by_layer.png)

![logit_lens_search_answer_margin.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_smollm3_appendix_e_n30/logit_lens_search_answer_margin.png)

![logit_lens_gold_distractor_margin.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_smollm3_appendix_e_n30/logit_lens_gold_distractor_margin.png)

![generation_prefix_mix.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_smollm3_appendix_e_n30/generation_prefix_mix.png)

## 结论

1. **retrieval-conditioned trained weaver 确实产生了不同于原 MemGen 的 policy shift。** 相比 MemGen，Retrieval + trained weaver 的 gold candidate probability 平均变化 +0.0367，gold-vs-distractor margin 同步提高 +0.0080；两者 next-token KL 为 0.9767，说明检索经验经过 trained weaver 后不是等价于原始 MemGen latent。
2. **untrained weaver 是关键负控，但不能简单用 gold probability 判定。** 它读到同一批 retrieved experience，gold probability 反而比 trained weaver 高 0.0441；这说明未训练 latent 可以偶然推高候选概率，不能只看单一概率指标。它的 next-token entropy 比 trained weaver 高 0.6382，trained-vs-untrained KL 为 16.4257。这更像无序扰动，而不是稳定的经验利用。
3. **层级指标比最终答案更能解释作用路径。** latent attention mass 表示 reasoner 哪些层直接读取 latent slots；hidden L2 表示 latent 对决策状态的扰动强度；logit-lens action-key probe 在强制前缀 `<` 后比较 `search` vs `answer`，answer-content probe 在 `<answer> ` 后比较 gold vs distractor。
4. **当前实验仍是 prompt-level augmentation，不等价于完整在线 Search-R1 多轮 agent。** 它回答的是 frozen reasoner 下 memory carrier 如何改变 token policy 和 hidden trajectory，不能单独证明最终多轮检索收益。

## Raw Files

- Raw JSONL: `/root/autodl-tmp/memrag_new/results/mechanistic_paths/smollm3_appendix_e_memgen_expel_paths_n30.jsonl`
- Summary JSON: `/root/autodl-tmp/memrag_new/results/mechanistic_paths/mechanistic_summary_smollm3_appendix_e_n30.json`
