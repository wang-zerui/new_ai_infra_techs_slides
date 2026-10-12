# Story — deck 29: FP8 grouped GEMM (HPC-Ops)

- **Where it lives** (prosperity): MoE expert layers in Tencent Hunyuan inference (HPC-Ops, H20/SM90): after
  routing, each expert's tokens form one group; one grouped GEMM computes every expert's projection in FP8
  (per-tensor or DeepSeek-style block-wise scaling) with BF16 output.
- **Pain number** (betrayal): at decode an expert gets a handful of tokens (README example: 8 per group); in a
  token-major grouped GEMM the smallest Hopper tile is 64 rows, so **87.5%** of each tile is padding (analytic).
- **Failed suitors**:
  1. Contiguous layout (DeepGEMM): each expert segment aligned to the M block, 128 rows by default — padding plus
     an aligned producer layout.
  2. Masked layout (DeepGEMM, decode / CUDA graph): no alignment, but BLOCK_M ≥ 64 on SM90 — still tile-sized
     padding per expert.
- **Villain**: Hopper's FP8 WGMMA fixes M at 64; putting tokens on M ties the tile floor to 64 rows regardless of
  how few tokens an expert has.
- **Turn / insight**: the WGMMA N side is flexible (8–256, step 8) and weights always fill 64 rows → swap
  operands (yᵀ = W·xᵀ) so tokens take N and the tile shrinks to 8.
- **Mechanisms**:
  ❶ Device-side per-group TMA descriptors from seqlens + cu_seqlens (answers: unequal groups without padding,
     host sync or repacking; compact and DeepEP strided layouts).
  ❷ Swap-AB with an adaptive token tile 8/16/32/48/64 chosen from the average group size; persistent kernel with a
     flattened (group, m, n) tile list and an optional decode task map (answers: tile waste at small groups).
  ❸ One mainloop for per-tensor and block-wise FP8 scales: per-128-K-block scaling and FP32 promotion (answers:
     both quantization recipes without separate kernels' structure).
- **Release**: up to **1.88×** over DeepGEMM at decode and **1.1×** at prefill on H20 (README maxima); analytic
  support: 8× more useful MMA rows at ≤ 8 tokens per expert.
- **Limits**: SM90 only; same N, K for all groups; one tile size from the average (skew); block-wise needs
  transposed, per-group-padded x_scale (reformat for DeepEP input, K ∈ {2048, 4096, 7168}); maxima only, no
  per-shape table; tile tables must stay consistent across kernels (PR #53 hang).

## Slide plan (13 logical slides, 14 pages, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | MoE layers multiply every expert's tokens at once | prosperity | fig_groups |
| 3 | Decode experts fill one-eighth of a tile | pain (analytic) | big number |
| 4 | Token-major tiles pad every small expert | suitors / villain | fig_padding |
| 5 | Insight: put tokens on the MMA's narrow side | turn | text + callout (build) |
| 6 | Kernel 0 describes groups, kernel 1 streams tiles | overview | fig_overview |
| 7 | ❶ cu_seqlens become per-group TMA descriptors | mechanism 1 | fig_tma |
| 8 | ❷ Swapping operands lets the tile shrink to 8 | mechanism 2 | fig_swap |
| 9 | At decode sizes, 8× more MMA rows do work | analytic support | chart tile_util |
| 10 | ❸ One mainloop handles both FP8 scalings | mechanism 3 | fig_scales |
| 11 | Up to 1.88× over DeepGEMM at decode | release | chart speedup |
| 12 | The price: Hopper only, one shape per call | limits | two-col |
| 13 | Summary | | QR |
