# MemGen / Retrieval-Conditioned Weaver 作用路径对比实验

样本数：2。三条主路径共享同一个可见 reasoner prompt、同一批 retrieved experience、同一组候选动作；差别只在 latent memory 的生成方式。

## 三条路径

- **MemGen**：训练后的 MemGen weaver 只读取当前 prompt，生成 latent memory。
- **Retrieval + trained weaver**：先从 ExpeL-like memory bank 检索显式经验文本，再把 `prompt + retrieved experience` 输入训练后的 weaver，reasoner 只看到 latent。
- **Retrieval + untrained weaver**：检索文本完全相同，但 weaver/projection/latent query 是未训练初始化。

## 主要数字

| Path | Gold prob | Search prob | Gold margin | Search-vs-answer margin | Entropy |
|---|---:|---:|---:|---:|---:|
| MemGen | 0.1005 | 0.8597 | 0.9833 | 2.2255 | 2.0140 |
| Retrieval + trained weaver | 0.1135 | 0.8424 | 1.0959 | 2.0619 | 3.5024 |
| Retrieval + untrained weaver | 0.0824 | 0.8751 | 0.8540 | 2.4171 | 5.3170 |

| Comparison | Next-token KL |
|---|---:|
| expel_trained_to_expel_untrained | 11.0836 |
| memgen_to_expel_trained | 5.1906 |
| memgen_to_expel_untrained | 9.3558 |
| no_latent_to_expel_trained | 5.5130 |
| no_latent_to_expel_untrained | 11.2845 |
| no_latent_to_memgen | 9.8471 |

## 图表

![candidate_policy_probability.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/dryrun_figures/candidate_policy_probability.png)

![next_token_policy_kl.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/dryrun_figures/next_token_policy_kl.png)

![layer_hidden_l2_to_no_latent.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/dryrun_figures/layer_hidden_l2_to_no_latent.png)

![latent_attention_mass_by_layer.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/dryrun_figures/latent_attention_mass_by_layer.png)

![logit_lens_search_answer_margin.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/dryrun_figures/logit_lens_search_answer_margin.png)

![logit_lens_gold_distractor_margin.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/dryrun_figures/logit_lens_gold_distractor_margin.png)

![generation_prefix_mix.png](/root/autodl-tmp/memrag_new/results/mechanistic_paths/dryrun_figures/generation_prefix_mix.png)

## 结论

1. **retrieval memory 是否起作用，要看 trained weaver 能不能把显式经验转换成有方向的 latent policy shift。** 如果 retrieval + trained weaver 相比 MemGen 的 KL 和 layer hidden drift 明显非零，同时候选动作概率朝 search/gold 方向变化，说明 retrieved experience 不是简单增加噪声，而是在改变后续 policy。
2. **untrained weaver 是关键负控。** 它读取相同 retrieval text，但如果 next-token KL、hidden drift 或 entropy 很大而 candidate margin 不改善，就说明“有检索文本”本身不够，训练后的 weaver/projection 才是把经验变成可用 latent 的路径。
3. **层级指标比最终答案更能解释作用路径。** latent attention mass 表示 reasoner 哪些层在直接读取 latent slots；hidden L2 表示 latent 对决策状态的扰动强度；logit-lens margin 表示这种扰动从哪些层开始转化成 action token 偏好。
4. **当前实验仍是 prompt-level augmentation，不等价于完整在线 Search-R1 多轮 agent。** 它回答的是 frozen reasoner 下 memory carrier 如何改变 token policy 和 hidden trajectory，不能单独证明最终多轮检索收益。

## Raw Files

- Raw JSONL: `/root/autodl-tmp/memrag_new/results/mechanistic_paths/dryrun_qwen15_n2.jsonl`
- Summary JSON: `/root/autodl-tmp/memrag_new/results/mechanistic_paths/dryrun_figures/mechanistic_summary.json`
