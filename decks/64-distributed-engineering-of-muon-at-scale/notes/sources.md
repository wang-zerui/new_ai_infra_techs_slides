# Sources — deck 64: Distributed engineering of Muon at scale

## Sources read

| Tag | Source | Where |
|---|---|---|
| [V4] | DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence", arXiv 2606.19348v1 (26 Apr 2026). https://arxiv.org/abs/2606.19348 (cached `arxiv-2606.19348.txt`) | §1 (p.4: "hybrid ZeRO strategy for the Muon optimizer"); §2.4 + Algorithm 1 (p.14); §3.4.1 Efficient Implementation of Muon (p.19-20); §4.2.1-4.2.2 (p.24-26) |
| [ML] | Liu et al. (Moonshot AI), "Muon is Scalable for LLM Training" (Moonlight), arXiv 2502.16982 (downloaded to `$SCRATCH/research/64/`) | Abstract; §1 contributions (p.2); §2.1 (p.2-3, Muon + Newton-Schulz); §2.3 Distributed Muon + Algorithm 1 + Analysis (p.4-5) |
| [GLM5] | GLM-5 Team, arXiv 2602.15763 (cached) | §2.1 "Muon Split" (p.4-5); §2.4.1 Memory Efficiency (p.9: Pipeline ZeRO2 gradient sharding; zero-redundant communication for the Muon distributed optimizer); App. A (p.36, Muon optimizer) |
| [K2] | Kimi Team, "Kimi K2", arXiv 2507.20534 (cached) | §2.4.2 (p.8: 16-way PP/EP + ZeRO-1 DP; ~30 GB of states per GPU); MuonClip (p.4) |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 1.6T / 49B; 284B / 13B | V4-Pro / V4-Flash size | [V4] Abstract |
| AdamW kept for embedding, prediction head, mHC static biases and gating factors, RMSNorm weights; Muon for all other modules | V4 optimizer split | [V4] §2.4 p.14 |
| momentum 0.95, weight decay 0.1, update RMS rescaled to 0.18 | V4 Muon hyper-parameters | [V4] §4.2.2 p.25 |
| 10 Newton-Schulz iterations: 8 with (3.4445, −4.7750, 2.0315), then 2 with (2, −1.5, 0.5) | hybrid NS | [V4] §2.4 p.14, eq. 28 |
| update rescale √max(n, m) · γ; Nesterov trick | Algorithm 1 | [V4] Algorithm 1 p.14 |
| ~2× compute efficiency / ~52% of training FLOPs vs AdamW | why Muon | [ML] Abstract; §1 p.2 |
| 4 + 2 + 4 = 10 vs 4 + 4 = 8 bytes; up to 1.25× | DP communication of ZeRO-1 Distributed Muon vs ZeRO-1 AdamW (FP32 G reduce-scatter + BF16 gather + FP32 P all-gather) | [ML] §2.3 Analysis p.5 |
| NS on the full gathered matrix on each DP rank, then discard all but the local partition | Distributed Muon algorithm | [ML] §2.3 Algorithm 1 p.5 |
| optimizer latency usually 1%-3% of forward-backward; ~5 NS steps | Moonlight's scale (speaker notes) | [ML] §2.3 p.5; §2.2 p.4 |
| Muon: 1 momentum buffer vs AdamW's 2 | optimizer state (speaker notes) | [ML] §2.3 p.5 |
| "transient memory spikes and redundant communication" | naive Muon all-gathers full params on every DP rank | [GLM5] §2.4.1 p.9 |
| < 10% padding memory; ≤ 5 parameter matrices per rank | dense knapsack buckets | [V4] §3.4.1 p.19 |
| ZeRO degree capped for dense params; redundant Muon update across extra DP groups | dense hybrid | [V4] §3.4.1 p.19-20 |
| MoE: flatten all down-projections of all experts across all layers, then up, then gate; pad so ranks split evenly without splitting a matrix; no ZeRO cap; padding negligible | MoE bucketing | [V4] §3.4.1 p.20 |
| consecutive same-shape parameters merged → batched NS | batching | [V4] §3.4.1 p.20 |
| NS stable with BF16 matmuls → MoE DP gradients quantized to BF16 with stochastic rounding, "halving the communication volume" | BF16 sync | [V4] §3.4.1 p.20 |
| all-to-all exchange, then local FP32 sum (replaces tree/ring reduce-scatter) | two-phase reduction | [V4] §3.4.1 p.20 |
| GLM-5: all-gather restricted to shards owned by each rank, overlapped with local computation | zero-redundant Muon communication | [GLM5] §2.4.1 p.9 |
| GLM-5: gradients sharded 1/dp per pipeline stage; only two full accumulation buffers, double-buffered | Pipeline ZeRO2 | [GLM5] §2.4.1 p.9 |
| GLM-5 Muon Split: orthogonalize per-head pieces of the MLA up-projections | what a "matrix" is | [GLM5] §2.1 p.4-5 |
| V4: QK-Clip not used (RMSNorm on queries and KV entries) | contrast with K2's MuonClip (deck 65) | [V4] §2.4 p.14 |
| TP requires an extra BF16 TP gather | limit (speaker notes) | [ML] footnote 5 p.5 |

## Derived numbers (labelled "derived" / "analytic" on slides)

| Derived | Formula | Value |
|---|---|---|
| V4-Pro expert matrices | (384 routed + 1 shared) × 3 (gate, up, down) × 61 layers | **70,455** |
| weights per V4-Pro expert matrix | 7168 × 3072 | **22.0M** |
| V4-Flash expert matrices (notes only) | 257 × 3 × 43; each 4096 × 2048 = 8.4M | 33,153 |
| gradient-path bytes per weight per step (chart `grad_traffic.py`) | Moonlight's accounting: FP32 = 4 B, BF16 = 2 B; ZeRO-1 AdamW = 4 (reduce-scatter); Distributed Muon = 4 + 2 (BF16 gather); V4 dense = 4 (owner receives whole matrices by reduce-scatter, no gather) + up to 10% padding; V4 MoE = 2 (BF16 all-to-all) | **4, 6, 4 (+≤0.4), 2** |

Assumptions behind the chart (stated on the slide caption): gradient path only (the parameter all-gather is excluded
because V4 does not state its precision); within one ZeRO group (V4's extra DP groups for dense params add a
cross-group reduction that V4 does not quantify); FP32 baseline for non-MoE gradient sync, implied by V4's statement that
BF16 "halv[es] the communication volume". The "no gather" step follows from whole-matrix ownership in V4 §3.4.1 (each
rank's bucket contains whole matrices, so the reduce-scatter already delivers full gradient matrices to their owner).

Illustrative (no real sizes): the matrix lengths in `fig_knapsack` and the cell counts in `fig_moe_flat` (31 matrices + 1
pad over 4 ranks) are schematic; the slides say so.

## Discrepancies

1. **"Halves MoE gradient-sync traffic"** (Notion) matches [V4] ("halving the communication volume") — consistent, but it
   applies only to MoE parameters; dense parameters are not stated to use BF16.
2. **"Knapsack-based hybrid ZeRO bucketing (<10% padding memory)"** — [V4] gives "< 10%" for the dense buckets only
   ("in our setup, where each rank manages no more than five parameter matrices"); MoE padding is "negligible".
3. **"Removes naive Muon's transient memory spikes"** is GLM-5's claim about its own zero-redundant all-gather ([GLM5]
   §2.4.1), not a DeepSeek-V4 result; no number is published. Slides attribute it to GLM-5.
4. **"Every DP rank need not all-gather full params"** — GLM-5's text says naive implementations all-gather full
   *parameters*; Moonlight's Distributed Muon gathers *gradients/momentum* for the matrix. Slides keep the two separate.
