# Story — deck 58: Ultra-sparse MoE (fine-grained routed experts + shared expert)

- **Where it lives** (prosperity): every 2026 open frontier MoE uses the same FFN shape — DeepSeek-V4-Pro
  (1.6T / 49B active, 384 routed + 1 shared, top-6), V4-Flash (284B / 13B), Kimi K2 (1.04T / 32.6B, 384 + 1, top-8),
  GLM-5 (744B / 40B, 256 + 1, top-8).
- **Pain number** (betrayal): a conventional top-2 MoE (GShard 2.9B) needs **1.5×** the expert parameters and
  computation to match a fine-grained + shared-expert MoE of 2B ([DSMoE]).
- **Failed suitors**:
  1. Dense scaling: every parameter is active on every token, so FLOPs grow with model size.
  2. Coarse MoE (8-16 big experts, top-1/2): sparse, but each expert mixes unrelated knowledge and all of them
     relearn the same common knowledge; only 120 expert combinations for top-2 of 16.
- **Villain**: expert granularity — a few large experts couple specialization to size, so the router cannot compose
  knowledge finely and common knowledge is stored many times.
- **Turn / insight**: at fixed activated parameters (constant FLOPs), adding more and smaller experts keeps lowering
  training and validation loss (K2 sparsity law). Quality follows total parameters; cost follows activated ones.
- **Mechanisms**:
  ❶ Fine-grained routed experts — split each FFN into m narrower experts, route to m× more (answers hybridity;
     120 → 4.4 billion combinations at equal FLOPs).
  ❷ One shared expert — always on, holds common knowledge once (answers redundancy; routed top-k shrinks to keep cost).
  ❸ Sparsity dial — total ÷ active experts set by a scaling law and capped by infra cost (answers "how sparse?").
- **Release**: sparsity 48 reaches the same loss with **1.69×** fewer FLOPs than sparsity 8 (K2); frontier models now
  activate **3-5%** of their weights (V4-Pro 3.1%, derived); a V4-Pro token touches 7 of 385 experts per layer.
- **Limits**: all experts stay resident (K2: ~6 TB of params + grad buffers over 256 GPUs); all-to-all per MoE layer
  (GLM-5 cuts layers, K2 keeps EP = 16, V4 needs wave-level overlap); balance and stability (aux-loss-free bias,
  MoE outliers → loss spikes); K2 stops at sparsity 48 for infra complexity.

## Slide plan (13 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Frontier open MoEs share one FFN shape | prosperity | fig_models (SVG table) |
| 3 | Coarse experts cost 1.5× the compute | pain | `%% big: 1.5×` |
| 4 | Dense FFNs and coarse MoEs both waste FLOPs | failed suitors / villain | fig_coarse (SVG two panels) |
| 5 | Insight: at equal FLOPs, more experts win | turn | text + boxed sentence (build) |
| 6 | A V4-Pro token uses 7 of 385 experts | overview | fig_overview (SVG router + 384 grid + shared) |
| 7 | ❶ Narrow experts multiply the combinations | mechanism | fig_segment (SVG) |
| 8 | ❷ One shared expert holds common knowledge | mechanism | fig_shared (SVG) |
| 9 | ❸ Sparsity is a dial capped by infrastructure | mechanism | fig_dial (SVG number line) |
| 10 | Sparsity 48 needs 1.69× fewer FLOPs than 8 | release | chart sparsity_flops |
| 11 | Frontier MoEs activate only 3–5% of weights | release | chart active_share (derived) |
| 12 | The cost: memory, all-to-all, balance | limits | two-col text + callout |
| 13 | Summary | | QR |
