# Appendix-E Retrieval-Weaver 大样本统计

本报告使用 paired 统计检查大样本下的结论是否稳定。每个样本同时跑 MemGen、Retrieval + trained weaver、Retrieval + untrained weaver，因此 paired delta 比单纯均值更能排除样本难度差异。

## 均值

| Model | n | Path | Gold prob | Gold margin | Entropy |
|---|---:|---|---:|---:|---:|
| Qwen1.5B | 100 | MemGen | 0.1900 | 1.0529 | 3.2157 |
| Qwen1.5B | 100 | Retrieval + trained | 0.2156 | 1.2654 | 3.7301 |
| Qwen1.5B | 100 | Retrieval + untrained | 0.1790 | 0.9538 | 5.3891 |
| SmolLM3-3B | 100 | MemGen | 0.1072 | 0.5846 | 5.9824 |
| SmolLM3-3B | 100 | Retrieval + trained | 0.1403 | 0.8033 | 4.3785 |
| SmolLM3-3B | 100 | Retrieval + untrained | 0.1825 | 1.0052 | 5.4578 |
| Pooled-200 | 200 | MemGen | 0.1486 | 0.8188 | 4.5990 |
| Pooled-200 | 200 | Retrieval + trained | 0.1779 | 1.0344 | 4.0543 |
| Pooled-200 | 200 | Retrieval + untrained | 0.1808 | 0.9795 | 5.4235 |

## Paired Delta

| Model | Comparison | Metric | Mean delta | 95% bootstrap CI | Positive rate |
|---|---|---|---:|---:|---:|
| Qwen1.5B | Trained - MemGen | Gold prob | 0.0256 | [0.0118, 0.0404] | 0.580 |
| Qwen1.5B | Trained - MemGen | Gold margin | 0.2125 | [0.1018, 0.3298] | 0.630 |
| Qwen1.5B | Trained - MemGen | Entropy | 0.5145 | [0.0966, 0.9362] | 0.560 |
| Qwen1.5B | Trained - Untrained | Gold prob | 0.0366 | [0.0200, 0.0539] | 0.650 |
| Qwen1.5B | Trained - Untrained | Gold margin | 0.3116 | [0.1905, 0.4390] | 0.710 |
| Qwen1.5B | Trained - Untrained | Entropy | -1.6590 | [-2.0041, -1.3032] | 0.160 |
| SmolLM3-3B | Trained - MemGen | Gold prob | 0.0331 | [0.0204, 0.0464] | 0.700 |
| SmolLM3-3B | Trained - MemGen | Gold margin | 0.2187 | [0.0983, 0.3303] | 0.640 |
| SmolLM3-3B | Trained - MemGen | Entropy | -1.6039 | [-1.8151, -1.4039] | 0.050 |
| SmolLM3-3B | Trained - Untrained | Gold prob | -0.0422 | [-0.0622, -0.0228] | 0.290 |
| SmolLM3-3B | Trained - Untrained | Gold margin | -0.2018 | [-0.4243, 0.0187] | 0.430 |
| SmolLM3-3B | Trained - Untrained | Entropy | -1.0793 | [-1.3074, -0.8566] | 0.200 |
| Pooled-200 | Trained - MemGen | Gold prob | 0.0294 | [0.0197, 0.0391] | 0.640 |
| Pooled-200 | Trained - MemGen | Gold margin | 0.2156 | [0.1360, 0.2964] | 0.635 |
| Pooled-200 | Trained - MemGen | Entropy | -0.5447 | [-0.8313, -0.2515] | 0.305 |
| Pooled-200 | Trained - Untrained | Gold prob | -0.0028 | [-0.0169, 0.0114] | 0.470 |
| Pooled-200 | Trained - Untrained | Gold margin | 0.0549 | [-0.0863, 0.1886] | 0.570 |
| Pooled-200 | Trained - Untrained | Entropy | -1.3692 | [-1.5810, -1.1584] | 0.180 |

## 图表

![metric bars](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_appendix_e_large_n100/appendix_e_large_metric_bars.png)

![paired delta](/root/autodl-tmp/memrag_new/results/mechanistic_paths/figures_appendix_e_large_n100/appendix_e_large_paired_delta_forest.png)

## 结论读法

- 本轮最稳定的结论是：Retrieval + trained weaver 相对 MemGen 在两个 base model 上都提高 gold prob 和 gold margin，且 bootstrap CI 不跨 0；Pooled-200 也保持同向。
- Trained - Untrained 不是单调胜出：Qwen1.5B 上 trained 明显优于 untrained；SmolLM3-3B 上 untrained 的 gold prob 更高，但 trained 明显降低 entropy。
- 因此不能只看候选答案概率。untrained weaver 有时能通过强扰动偶然提高 gold prob，但 trained weaver 的作用路径更像可控的 latent policy modulation。

## Raw Inputs

- Qwen1.5B: `/root/autodl-tmp/memrag_new/results/mechanistic_paths/qwen15_appendix_e_memgen_expel_paths_n100.jsonl`
- SmolLM3-3B: `/root/autodl-tmp/memrag_new/results/mechanistic_paths/smollm3_appendix_e_memgen_expel_paths_n100.jsonl`
