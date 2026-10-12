# Story — deck 61: mHC (manifold-constrained hyper-connections)

- **Where it lives** (prosperity): every Transformer block of DeepSeek-V4-Pro (1.6T) and V4-Flash (284B) uses mHC with
  a 4-lane residual stream (n = 4). Hyper-Connections (HC) widen the residual stream to n × C and learn how the lanes
  mix, at almost no FLOPs; the mixing matrix H_res carries most of HC's gain (−0.022 of −0.027 loss gap).
- **Pain number** (betrayal): in a 27B MoE, HC's composite residual mapping amplifies signals by up to **~3000×**
  (ideal: 1), and the loss surges near step 12k.
- **Failed suitors**:
  1. Plain residual (H_res = I): stable, but the lanes never exchange information — gives up most of HC's gain.
  2. Unconstrained HC: gains, but the product of 60 learned H_res matrices drifts away from identity; plus ~n× residual
     memory traffic (34C vs 3C per token-layer) and n× pipeline traffic.
- **Villain**: nothing constrains the *product* of the mixing matrices across depth; each layer's small gain compounds.
- **Turn / insight**: doubly stochastic matrices have spectral norm ≤ 1 and are closed under multiplication, so a product
  of any depth is still doubly stochastic: mixing is allowed, amplification is not. Measured: composite gain ≤ ~1.6.
- **Mechanisms**:
  ❶ Sinkhorn-Knopp projection of H_res onto the Birkhoff polytope (exp, then alternate column/row normalization,
     20 iterations) — answers: unbounded composite gain.
  ❷ One small projection (nC × 24) generates input-dependent pre/post/res mappings; σ and 2σ keep pre/post
     non-negative — answers: expressivity without signal cancellation.
  ❸ Systems: fused kernels (post/res merge 25C → 9C elements at n = 4), block recomputation aligned to PP stages,
     DualPipe extension — answers: the n-fold memory, activation and pipeline cost.
- **Release**: composite gain ~3000 → ≤ 1.6; 27B final loss −0.021 vs baseline, mHC ≥ baseline on all 8 benchmarks
  (BBH +7.2); 6.7% training-time overhead at n = 4.
- **Limits**: 6.7% time, n× activation/pipeline traffic still there, 20 SK iterations are approximate (backward gain
  deviates), custom kernels needed (mHC-lite, Mega mHC deck 41), results are DeepSeek-internal; AdamW for its static
  biases/gates.

## Slide plan (13 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Hyper-Connections widen the residual into 4 lanes | prosperity | fig_layer_1 (SVG, HC layer) |
| 3 | Unconstrained mixing amplifies signals ~3000× | pain | `%% big: ~3000×` |
| 4 | Identity is stable; free mixing compounds | failed suitors / villain | fig_compose_1 (SVG two rows) |
| 5 | Insight: doubly stochastic products stay bounded | turn | fig_compose_2 (adds the mHC row) |
| 6 | mHC projects each mapping onto a safe set | overview | fig_layer_2 (same canvas + projections) |
| 7 | ❶ Sinkhorn-Knopp balances rows and columns | mechanism | fig_sinkhorn (SVG, toy numbers) |
| 8 | ❷ One 24-wide projection drives all mappings | mechanism | fig_param (SVG) |
| 9 | ❸ Fused kernels cut merge traffic 25C → 9C | mechanism | chart io_merge (analytic) |
| 10 | Composite gain drops from ~3000 to 1.6 | release | chart gain (log) |
| 11 | mHC beats the baseline on all eight benchmarks | release | chart bench27b |
| 12 | The cost: 6.7% step time, custom kernels | limits | two-col text |
| 13 | Summary | | QR |
