# Story — deck 53: Memory-efficient token permutation

Family: decks 52 / 53 / 54 attack the MoE activation memory wall (recompute, reorder, offload). Shared
visual language: DeepSeek-V3 per-GPU memory stack (Table 3; here as a before/after stacked bar like deck 46's
chart) and the "memory toolbox" savings ledger with this deck's bar highlighted. Cross-references: 52, 54,
46 (FP8 activations), 49 (BF16 optimizer moments), 70 (FSDP).

- **Where it lives** (prosperity): every MoE layer (router → permute → dispatch → experts → combine →
  unpermute); Megatron-Core v0.13+ (README), on in both final DeepSeek-V3 configurations (Table 18).
- **Pain number** (betrayal): ~26.3 GB per GPU of DeepSeek-V3 activations (≈ 20% of 131 GB, derived) are
  held only to differentiate the router weights.
- **Failed suitors**: (1) store E (the standard order) = 26.3 GB; (2) recompute E in backward (Kimi K2
  recomputes MoE down-projections) = an extra expert GEMM. Neither removes the dependency.
- **Villain**: applying p after the expert makes ∂L/∂p = ⟨E, ∂L/∂y⟩ depend on the expert output E, a
  hidden-size row per routed copy that nothing else in backward needs (fc2's backward needs its input a).
- **Turn / insight**: p is a scalar and W2 is a bias-free linear map, so p·W2·φ(z) = W2·(p·φ(z)); then
  ∂L/∂p needs only φ(z), recomputable from z, which SwiGLU saves anyway.
- **Mechanisms**:
  ❶ Probs travel with tokens: permute(tokens, map, probs) → one p per row → second all-to-all → expert;
     unpermute becomes a plain FP32 sum (answers: p must be available where φ(z) lives).
  ❷ Fused weighted SwiGLU: forward saves (z, p) only; backward dz = SwiGLU'(z)(p g), dp = Σ SwiGLU(z)⊙g,
     SwiGLU(z) recomputed on the fly (answers: no new buffer for the new gradient).
- **Release**: 26.3 GB per GPU freed at zero compute cost (Table 6 lists no trade-off); per routed copy the
  saved expert-path values drop from 20,480 to 13,313 (analytic, E was 35%).
- **Limits**: requires bias-free fc2 (bias scaled by p separately in Megatron); only E goes (x, z, a stay →
  pair with 52/46/54); p needs its own small all-to-all; top-1 alternative moe_apply_probs_on_input.

## Slide plan (15 logical slides, 15 PDF pages, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Every MoE output is a router-weighted expert sum | prosperity | fig_moe |
| 3 | One multiply pins a fifth of all activations | pain | `%% big: 26.3 GB` |
| 4 | Weighting after the expert pins its output | villain | fig_before |
| 5 | Storing or recomputing E both cost something | failed suitors | two-col |
| 6 | Insight: a scalar weight commutes with fc2 | turn | fig_algebra |
| 7 | Move the multiply from unpermute into SwiGLU | overview | fig_pipeline |
| 8 | ❶ Router weights travel with their tokens | mechanism | fig_dispatch (worked example) |
| 9 | ❷ One fused kernel weights forward and backward | mechanism | fig_swiglu |
| 10 | Per routed token, the E row disappears | mechanism → result bridge | fig_ledger (analytic) |
| 11 | 26.3 GB freed per GPU at zero compute cost | release | chart mem_compare |
| 12 | The only memory saver with no trade-off | result (family) | chart toolbox |
| 13 | Kimi K3 cuts the same dependency another way | related practice | two-col |
| 14 | The cost: a bias rule and a tiny all-to-all | limits | two-col |
| 15 | Summary | | QR arxiv.org/abs/2603.07685 |
