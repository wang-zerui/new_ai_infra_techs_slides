# Story — deck 64: Distributed engineering of Muon at scale

- **Where it lives** (prosperity): Muon now trains the largest open MoEs — DeepSeek-V4-Pro (1.6T) and V4-Flash (284B)
  use it for every module except embedding, head, RMSNorm and mHC biases/gates; Kimi K2 (MuonClip) and GLM-5 use it
  too. Moonlight measured ~2× compute efficiency over AdamW (~52% of the FLOPs).
- **Pain number** (betrayal): Muon's update needs the *whole* matrix (Newton-Schulz multiplies X·Xᵀ). The first ZeRO-1
  port (Moonlight's Distributed Muon) gathers the full matrix on every DP rank: DP traffic up to **1.25×** that of ZeRO-1
  AdamW, and every rank re-runs Newton-Schulz on the full matrix. GLM-5: naive versions all-gather full params on every
  rank → transient memory spikes.
- **Failed suitors**:
  1. Element-wise ZeRO (AdamW style): cuts matrices across ranks — wrong for Muon.
  2. ZeRO-1 + gather (Distributed Muon): correct, but extra gather + DP-fold redundant Newton-Schulz.
- **Villain**: granularity mismatch — ZeRO shards a flat vector of elements; Muon's unit of work is a matrix.
- **Turn / insight**: shard by whole matrices instead of elements. A fine-grained MoE is made of tens of thousands of
  equal-size independent matrices (V4-Pro: 70,455 expert matrices of 22M weights, derived), so whole-matrix assignment
  balances almost perfectly; the few, uneven dense matrices need packing and a smaller ZeRO group.
- **Mechanisms**:
  ❶ Dense: knapsack-pack whole matrices into per-rank buckets, cap the ZeRO degree (≤ 5 matrices per rank), pad to the
     largest bucket (< 10% memory); extra DP groups recompute (answers: few uneven matrices).
  ❷ MoE: flatten down | up | gate of all experts across layers, split across all ranks at matrix boundaries, merge
     same-shape runs into batched Newton-Schulz (answers: tens of thousands of small matrices, GPU efficiency).
  ❸ MoE gradient sync in BF16 with stochastic rounding via all-to-all + local FP32 sum (answers: traffic, without
     low-precision accumulation error).
- **Release**: gather step gone (owner receives whole matrices); MoE gradient-path bytes halved; analytic gradient-path
  traffic per weight 6 B (Distributed Muon) → 4 B (V4 dense) → 2 B (V4 MoE). GLM-5 reaches the same principle from the
  all-gather side.
- **Limits**: dense NS recomputed across extra DP groups; padding < 10% only with few matrices per rank; BF16 safety is an
  empirical observation; all-to-all replaces ring/tree reduce-scatter; AdamW params still use classic ZeRO; TP adds a gather.

## Slide plan (13 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Muon updates each weight matrix as one unit | prosperity | fig_muon_step (SVG) |
| 3 | Gathering for Muon costs up to 1.25× traffic | pain | `%% big: 1.25×` |
| 4 | ZeRO slices the matrices Muon needs whole | failed suitors / villain | fig_zero (SVG two panels) |
| 5 | Insight: shard by matrix, not by element | turn | text + boxed sentence (build) |
| 6 | V4 routes dense and expert weights differently | overview | fig_overview (SVG) |
| 7 | ❶ Knapsack packs whole dense matrices per rank | mechanism | fig_knapsack (SVG) |
| 8 | ❷ Expert matrices split only at matrix edges | mechanism | fig_moe_flat (SVG) |
| 9 | ❸ BF16 on the wire, FP32 in the sum | mechanism | fig_bf16 (SVG) |
| 10 | Gradient traffic falls from 6 to 2 B/weight | release | chart grad_traffic (analytic) |
| 11 | GLM-5 removes the redundant all-gather instead | adoption | two-col text + callout |
| 12 | The cost: recompute, padding, BF16 trust | limits | two-col text |
| 13 | Summary | | QR |
