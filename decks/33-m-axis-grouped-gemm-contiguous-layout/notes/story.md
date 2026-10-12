# Story — deck 33: M-axis grouped GEMM (contiguous layout)

- **Where it lives** (prosperity): DeepGEMM's `m_grouped_fp8_gemm_{nt,nn}_contiguous`, the expert GEMMs of
  MoE training forward and inference prefill. DeepSeek-V3: 58 MoE layers, 256 routed experts, top-8,
  hidden 7168, expert width 2048; prefill runs EP32 with 8 + 1 redundant experts per GPU.
- **Pain number** (betrayal): fixed-shape (batched) expert GEMMs must drop tokens or pad every expert to a
  capacity; avoiding drops this way raised MoE-layer math by **over 2×** (MegaBlocks §3), and some MoEs need
  capacity factors as high as 11.
- **Failed suitors**:
  1. Batched GEMM with a capacity factor: drop tokens (quality) or pad (math, memory).
  2. One GEMM launch per expert: G ragged kernels, each with its own wave tail; the host needs each M_e.
  3. Generic grouped GEMM (CUTLASS-style, per-problem M/N/K): generality the MoE does not need, and per
     precision/shape engineering (MCore §4.3.2).
- **Villain**: GEMM APIs model one shape per call (or arbitrary shapes per problem); an MoE layer is neither:
  every expert has the same weight shape [N, K] and only the token count M_e is ragged.
- **Turn / insight**: only M is ragged, so stack all experts' tokens along M; if no M tile straddles two
  experts, the whole layer is one dense GEMM whose M tiles each pick their expert's weight.
- **Mechanisms**:
  ❶ Contiguous layout aligned to BLOCK_M (128; 256 on SM100), padding rows marked −1 (answers: tiles must
     not mix experts).
  ❷ Tile scheduler reads the expert of each M tile from `m_indices[m·BLOCK_M]`, offsets B by e·N, skips
     padding blocks, multicasts B only between same-expert tiles (answers: one launch, no per-expert sizes
     on the host besides ΣM).
  ❸ Producer writes the layout: DeepEP dispatch emits the aligned, zero-padded layout directly
     (`expert_alignment`), psum layout = one int per expert (answers: permute/pad kernels between dispatch
     and GEMM).
- **Release**: grouped FP8 GEMM at 1214–1346 TFLOPS on H800, 1.2–1.3× over the tuned CUTLASS 3.6 baseline,
  and at or above the dense GEMM of the same per-expert shape (1265 / 1168 TFLOPS). Padding is bounded by
  one tile per expert (≤ 3.1% worst case at 4096 tokens/expert, analytic) instead of > 2×.
- **Limits**: ΣM is a host-side shape (plain layout needs counts on the CPU after dispatch, so no CUDA Graph
  → decode uses masked, deck 34); all experts must share N and K; padding grows at small M; weight gradients
  group along K (deck 32); FP8 on SM90 needs K-major weights; fused Mega MoE (deck 35) supersedes the
  separate-kernel pipeline for FP8×FP4 EP inference.

## Slide plan (13 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Every MoE layer runs one GEMM per expert | prosperity | fig_ragged |
| 3 | Fixed-shape expert GEMMs pay over 2× the math | pain | `%% big: >2×` |
| 4 | Fixed shapes pad, drop, or launch per expert | failed suitors | fig_statusquo (a)(b) |
| 5 | Insight: experts differ only in M | turn | bullets + boxed sentence |
| 6 | One contiguous layout serves all experts | overview | fig_overview |
| 7 | ❶ Align every expert segment to the M tile | mechanism | fig_align_1/2 (build) |
| 8 | ❷ Each M tile looks up its own expert | mechanism | fig_sched |
| 9 | ❸ Dispatch writes the GEMM-ready layout | mechanism | fig_pipeline |
| 10 | Grouped GEMM matches dense: 1346 TFLOPS | result | chart tflops |
| 11 | Alignment padding stays ≤ 3.1% at prefill scale | result/cost | chart padding (analytic) |
| 12 | Contiguous needs host-side counts and one shape | limits | two-col |
| 13 | Summary | | QR |
