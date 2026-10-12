# Story — deck 57: Stable LatentMoE (Kimi K3)

- **Where it lives** (prosperity): Kimi K3 (2.8T total, 104B active) scales MoE width: 896 routed experts with
  16 active per token (sparsity 56) and 2 full-width shared experts, versus Kimi K2's 384 / 8 / 1 at the same
  hidden size d = 7,168.
- **Pain number** (betrayal): if K3's 16 active experts each received the full 7,168-wide token, all-to-all
  dispatch/combine volume per token would be **2×** Kimi K2's (16 × 7,168 vs 8 × 7,168; analytic, from K3
  Table 1 and the LatentMoE volume model ∝ K·d). Routed-expert weights would double too (66M vs 33M params per
  expert).
- **Failed suitors**:
  1. Keep top-k small (K2's 8): caps the routed nonlinear budget K·m and the combinatorial expert space.
  2. Plain LatentMoE (NVIDIA): shrink the routed width with a shared W↓ / W↑ — fixes the bytes, but at 896
     experts and 2.8T parameters the routed path becomes W↓ → gated expert FFN → W↑, "nearly four consecutive
     matrix multiplications": ill-conditioned, exploding internal activations (K3 §2.3). SwiGLU's product of two
     unbounded factors adds outliers and low-precision overflow risk (§2.3.2). And balancing ~10³ experts breaks
     the sign-update bias (deck 56).
- **Villain**: the latent path has no scale control: the aggregated latent u changes scale with which experts
  fire and with their weights, and every factor of SwiGLU can grow without bound — compounded across a
  four-matmul chain at 2.8T scale.
- **Turn / insight**: routed experts do not need the full hidden width — quality holds for compression
  ratios d/ℓ ≤ 4 (LatentMoE, measured on a 16B MoE), and all-to-all volume scales with K·ℓ. So K3 halves the
  routed width (ℓ = 3,584) and doubles top-k at K2's bytes, then makes the narrow path scale-safe.
- **Mechanisms**:
  ❶ Latent routed experts: one shared W↓ (7,168 → 3,584) before dispatch, experts 3,584 → 3,072 → 3,584,
     one shared W↑ after combine (answers: bytes and weight traffic).
  ❷ RMSNorm on the aggregated latent before W↑ (answers: scale varies with the selected experts).
  ❸ SiTU-GLU: smooth tanh caps on both GLU factors, |output| ≤ β1β2 = 100 (answers: activation explosion).
  (Load balance at 896 experts: Quantile Balancing, deck 56.)
- **Release**: K3 trains 16-of-896 routing stably at 2.8T; LatentMoE's measured evidence: +4.6 MMLU-Pro at
  equal active parameters (73B hybrid, 1T tokens) and serving throughput within ~6% at concurrency ≥ 4 (−12% at
  concurrency 1) on H100; K3's overall 2.5× scaling efficiency over K2 is joint with KDA, AttnRes and recipes.
- **Limits**: no isolated K3 ablation numbers for RMSNorm or SiTU; two extra latent GEMMs (51M params per
  layer) that K3 fuses with the router GEMM in serving; more and smaller expert GEMMs stress kernels; latent
  width has a floor (quality drops beyond d/ℓ = 4 in LatentMoE's sweep); needs QB.

## Slide plan (14 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | K3 doubles K2's active experts at equal width | prosperity | chart width.py |
| 3 | Full-width experts would double the all-to-all | pain | `%% big: 2×` |
| 4 | The latent detour chains four unnormalized matmuls | suitors / villain | fig_chain (SVG) |
| 5 | Insight: experts do not need full width | turn | text + callout |
| 6 | Stable LatentMoE: one narrow, scale-safe routed path | overview | fig_overview (SVG) |
| 7 | ❶ Sixteen half-width experts cost K2's bytes | mechanism | fig_bytes (SVG) |
| 8 | ❷ RMSNorm fixes the latent's scale before W↑ | mechanism | fig_norm (SVG) |
| 9 | ❸ SiTU-GLU caps both GLU factors smoothly | mechanism | chart situ.py (analytic) |
| 10 | Latent experts add +4.6 MMLU-Pro at equal cost | release | chart accuracy.py |
| 11 | The latent detour costs ≤ 12% serving throughput | release / cost | chart throughput.py |
| 12 | K3 scales 2.5× more efficiently than K2 | release | `%% big: 2.5×` |
| 13 | The price: extra GEMMs, small experts, no ablation | limits | two-col |
| 14 | Summary | | QR |
