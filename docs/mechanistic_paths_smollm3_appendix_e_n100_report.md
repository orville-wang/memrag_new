# MemGen / Retrieval-Conditioned Weaver 作用路径对比实验

样本数：100。三条主路径共享同一个可见 reasoner prompt、同一批 retrieved experience、同一组候选动作；差别只在 latent memory 的生成方式。

## 三条路径

- **MemGen**：训练后的 MemGen weaver 只读取当前 prompt，生成 latent memory。
- **Retrieval + trained weaver**：先从 ExpeL-like memory bank 检索显式经验文本，再把 `prompt + retrieved experience` 输入训练后的 weaver，reasoner 只看到 latent。
- **Retrieval + untrained weaver**：检索文本完全相同，但 weaver/projection/latent query 是未训练初始化。

## 主要数字

| Path | Gold prob | Search prob | Gold margin | Search-vs-answer margin | Entropy |
|---|---:|---:|---:|---:|---:|
| MemGen | 0.1072 | 0.6955 | 0.5846 | 2.2773 | 5.9824 |
| Retrieval + trained weaver | 0.1403 | 0.6617 | 0.8033 | 1.9973 | 4.3785 |
| Retrieval + untrained weaver | 0.1825 | 0.5988 | 1.0052 | 1.3858 | 5.4578 |

## 层级作用路径读数

| Path | Final latent attention | Final hidden L2 | Final search-answer logit margin | Final gold-distractor logit margin |
|---|---:|---:|---:|---:|
| MemGen | 0.5497 | 47.4418 | 8.4694 | 0.0241 |
| Retrieval + trained weaver | 0.5153 | 47.4764 | 6.4550 | 0.3134 |
| Retrieval + untrained weaver | 0.1974 | 53.9008 | 4.1375 | 2.0181 |

| Comparison | Next-token KL |
|---|---:|
| expel_trained_to_expel_untrained | 17.6054 |
| memgen_to_expel_trained | 1.2451 |
| memgen_to_expel_untrained | 10.9538 |
| no_latent_to_expel_trained | 3.3334 |
| no_latent_to_expel_untrained | 22.7967 |
| no_latent_to_memgen | 3.0195 |

## 图表

![candidate_policy_probability.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_smollm3_appendix_e_n100/candidate_policy_probability.png)

![next_token_policy_kl.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_smollm3_appendix_e_n100/next_token_policy_kl.png)

![layer_hidden_l2_to_no_latent.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_smollm3_appendix_e_n100/layer_hidden_l2_to_no_latent.png)

![latent_attention_mass_by_layer.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_smollm3_appendix_e_n100/latent_attention_mass_by_layer.png)

![logit_lens_search_answer_margin.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_smollm3_appendix_e_n100/logit_lens_search_answer_margin.png)

![logit_lens_gold_distractor_margin.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_smollm3_appendix_e_n100/logit_lens_gold_distractor_margin.png)

![generation_prefix_mix.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_smollm3_appendix_e_n100/generation_prefix_mix.png)

## 结论

1. **retrieval-conditioned trained weaver 确实产生了不同于原 MemGen 的 policy shift。** 相比 MemGen，Retrieval + trained weaver 的 gold candidate probability 平均变化 +0.0331，gold-vs-distractor margin 同步提高 +0.2187；两者 next-token KL 为 1.2451，说明检索经验经过 trained weaver 后不是等价于原始 MemGen latent。
2. **untrained weaver 是关键负控，但不能简单用 gold probability 判定。** 它读到同一批 retrieved experience，gold probability 反而比 trained weaver 高 0.0422；这说明未训练 latent 可以偶然推高候选概率，不能只看单一概率指标。它的 next-token entropy 比 trained weaver 高 1.0793，trained-vs-untrained KL 为 17.6054。这更像无序扰动，而不是稳定的经验利用。
3. **层级指标比最终答案更能解释作用路径。** latent attention mass 表示 reasoner 哪些层直接读取 latent slots；hidden L2 表示 latent 对决策状态的扰动强度；logit-lens action-key probe 在强制前缀 `<` 后比较 `search` vs `answer`，answer-content probe 在 `<answer> ` 后比较 gold vs distractor。
4. **当前实验仍是 prompt-level augmentation，不等价于完整在线 Search-R1 多轮 agent。** 它回答的是 frozen reasoner 下 memory carrier 如何改变 token policy 和 hidden trajectory，不能单独证明最终多轮检索收益。

## Raw Files

- Raw JSONL: `/root/autodl-tmp/memrag_new/results/mechanistic_paths/smollm3_appendix_e_memgen_expel_paths_n100.jsonl`
- Summary JSON: `/root/autodl-tmp/memrag_new/results/mechanistic_paths/mechanistic_summary_smollm3_appendix_e_n100.json`
