# Story — deck 32: K-axis grouped GEMM (MoE weight backward)

- **Where it lives** (prosperity): FP8 MoE training. DeepSeek-V3 runs all three GEMMs of every linear layer
  (Fprop, Dgrad, Wgrad) in FP8 with fine-grained scaling. For the routed experts, Fprop and Dgrad are
  M-grouped (deck 33: `m_grouped_fp8_gemm_{nt,nn}_contiguous`); Wgrad is DeepGEMM's
  `k_grouped_fp8_gemm_tn_contiguous` (SM100) / `k_grouped_fp8_gemm_nt_contiguous` (SM90).
- **Pain number** (betrayal): Wgrad is **1/3** of every expert's training GEMM FLOPs (analytic: each of the three
  GEMMs is 2·M·N·K), and it is the one whose reduction axis is the ragged per-expert token count, so the
  M-grouped kernels cannot run it.
- **Failed suitors**:
  1. One GEMM per expert (DeepGEMM's own first version, May 2025: a Python loop over experts with a host
     list of K sizes): G launches per weight per layer, host-known K.
  2. Pad every expert's tokens to the largest: zero rows in the reduction cost full FLOPs (and FP8 alignment
     already forces token padding, MCore §5.4.1).
- **Villain**: in the weight gradient the token axis becomes the dot-product (K) axis; GEMM APIs that group by
  output rows (M) or need one K per call do not fit an op whose outputs are fixed and whose K is ragged.
- **Turn / insight**: for Wgrad, experts differ only in K: every expert has the same M×N output tile grid, so
  stack the experts' tokens along K, align each segment to the 128-token K block, and let each tile reduce only
  over its own expert's segment.
- **Mechanisms**:
  ❶ K-grouped contiguous layout: `A [ΣK, M]`, `B [ΣK, N]` token-major, each expert padded to 128 tokens with
     zeros (answers: a K block must never mix two experts' tokens).
  ❷ Scheduler: G copies of the same M×N tile grid; tile (g, m, n) walks only expert g's K blocks and accumulates
     into FP32 `D[g] = C[g] + A_gᵀ B_g` (answers: one launch for all experts, gradient accumulation fused).
  ❸ Token-axis scales and device-side offsets: per-channel scales per 128 tokens, packed per expert into UE8M0
     by the K-grouped packing kernel; psum layout = one int per expert on the GPU, host K list optional (SM100);
     SM90 rewrites a TMA descriptor in shared memory per expert (answers: scale groups and offsets follow the
     ragged axis without host knowledge).
- **Release**: FP8 grouped wgrad at 900–1044 TFLOPS for DeepSeek-V3 expert shapes, ≥ 97.8% of a dense wgrad of
  the same per-expert shape (derived), ~20% over the internal CUTLASS implementation (PR #95); Sept 2026 release:
  ~20% faster MoE backward GEMMs (PR #432).
- **Limits**: shared M, N; ≤ 127 padding tokens per expert; column-wise (token-axis) FP8 quantization of
  activations and gradients is a separate pass; SM90 needs K-major per-expert copies, a host K list and FP32
  accumulation; the published per-kernel numbers predate the single-kernel versions (kernel time of the
  May 2025 loop).

## Slide plan (13 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Each expert trains with three FP8 GEMMs | prosperity | fig_backward |
| 3 | Wgrad is a third of expert GEMM work | pain | `%% big: 1/3` |
| 4 | Ragged K defeats per-expert loops and padding | failed suitors | fig_kstatusquo (a)(b) |
| 5 | Insight: in Wgrad, experts differ only in K | turn | bullets + boxed sentence |
| 6 | Stack experts along K, one output per expert | overview | fig_kgroup |
| 7 | ❶ Align each expert to the 128-token K block | mechanism | fig_kalign_1/2 (build) |
| 8 | ❷ Every expert gets the same tile grid | mechanism | fig_ksched |
| 9 | ❸ Scales and offsets follow the token axis | mechanism | fig_kscale |
| 10 | Grouped Wgrad keeps ≥ 97.8% of dense speed | result | chart wgrad_tflops |
| 11 | From a host loop to one GPU-driven kernel | result/evolution | fig_timeline |
| 12 | The cost: token-axis casts and shared shapes | limits | two-col |
| 13 | Summary | | QR |
