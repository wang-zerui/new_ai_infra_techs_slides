# Story — deck 60: Deep-and-narrow + more attention heads + partial RoPE (GLM-4.5)

- **Where it lives** (prosperity): GLM-4.5 (Zhipu AI & Tsinghua), 355B-total / 32B-active MoE, 23T tokens, Muon.
  Its peers at the same ~32B active budget, DeepSeek-V3 (671B) and Kimi K2 (1043B), share one template:
  61 layers × 7168 hidden, MLA.
- **Pain number** (betrayal): Kimi K2 carries **1,043B** total parameters to activate 32B per token, 2.9× GLM-4.5's 355B
  ([GLM45] Table 1). Total weights are what training and serving must hold in memory.
- **Failed suitors**:
  1. Spend the budget on width and sparsity: K2's sparsity law says sparsity 48 reaches the same loss with 1.69× fewer
     FLOPs than sparsity 8, so experts go 256 → 384 and total weights balloon.
  2. Judge heads by loss: doubling heads lowers validation loss only 0.5–1.2% but costs +83% inference FLOPs at 128K, so K2
     halves heads 128 → 64.
- **Villain**: the shape is chosen by validation loss at fixed FLOPs (plus hardware rooflines). Loss rewards width and
  sparsity and cannot see what depth and heads do for reasoning.
- **Turn / insight**: GLM-4.5's ablations: deeper models reason better; 2.5× more heads leave training loss unchanged yet
  consistently raise MMLU and BBH. (Qualitative; no numbers published.)
- **Mechanisms**:
  ❶ Depth for width: 92 layers × 5120 hidden, 160 experts (vs 61 × 7168, 256/384). Answers: same ~32B active budget.
  ❷ 96 query heads on GQA-8: query width 12,288 = 2.4× hidden while only 8 KV heads are cached. Answers: more heads
     without multiplying the KV cache.
  ❸ Partial RoPE: rotate 64 of 128 head dims, leave 64 position-free; plus QK-Norm (deck 66) to bound logits. Answers:
     a narrow 128-dim head still needs a content-only channel (same split as MLA's 128 nope + 64 RoPE).
- **Release**: 355B total, 34% of K2 and 53% of DSV3; GLM-4.5-Base within 1.7 (MMLU) / 2.5 (BBH) points of K2-Base;
  post-trained GLM-4.5 ranks 3rd of 12 models on the ARC suite.
- **Limits**: no ablation numbers published; 23T vs 14.8T/15.5T tokens confound model-level comparisons; KV cache 5.4× MLA
  per token (analytic, GQA-8 × 92 layers); 178 vs 116 EP all-to-alls per forward pass; GLM-5 walked back to 78 layers
  (EP communication), MLA (memory) and 64 heads.

## Slide plan (13 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Three ~32B-active MoEs, three very different shapes | prosperity | fig_shapes (SVG silhouettes) |
| 3 | Width costs weights: K2 holds 1,043B | pain | `%% big: 1,043B` |
| 4 | Loss-only search rewards width, punishes heads | suitors + villain | two-col text, callout |
| 5 | Insight: depth and heads buy reasoning, not loss | turn | boxed sentence |
| 6 | GLM-4.5 reshapes the model at three levels | overview | fig_overview (SVG zoom: model → layer → head) |
| 7 | ❶ Same active budget, 31 more narrower layers | mechanism | fig_table (SVG shape table) |
| 8 | ❷ 96 query heads share just 8 KV heads | mechanism | fig_gqa (SVG) |
| 9 | ❸ Partial RoPE rotates only half of each head | mechanism | fig_prope (SVG) |
| 10 | A third of K2's weights, within 2.5 points | result | chart_base (params + MMLU/BBH) |
| 11 | The bill: 5.4× KV cache, 53% more all-to-alls | cost | chart_cost (analytic) |
| 12 | GLM-5 cut depth to 78 and moved to MLA | limits / maturity | two-col text |
| 13 | Summary | release | bullets + QR |
