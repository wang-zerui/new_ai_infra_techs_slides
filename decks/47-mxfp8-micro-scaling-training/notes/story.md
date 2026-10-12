# Story: MXFP8 micro-scaling training

Rung 2 of the precision ladder FP8 (deck 46, Hopper) -> **MXFP8 (this deck, Blackwell)** -> NVFP4 (deck 48).

- **Where it lives** (prosperity): Blackwell (B200 / GB200 / GB300 / B300) pretraining. MXFP8 is the default FP8
  recipe on Blackwell in Megatron-Core; DeepSeek-V3 trains at 1,048 TFLOPS/GPU on 256 GB200s and 1,233 on GB300
  in MXFP8. TE exposes it as `MXFP8BlockScaling`.
- **Pain number** (betrayal): on Hopper, the accurate fine-grained FP8 recipe is the slow one: blockwise FP8 GEMMs
  reach **1.41x** over BF16 on H200, below per-tensor current scaling (1.57x) and delayed scaling (1.69x)
  (TE "GEMM Speedups", 5B model). Fine scales cost speed because they are applied in software.
- **Failed suitors**:
  1. Per-tensor FP8 (fast): needs E5M2 gradients and leaves some blocks in BF16 (Nemotron-H keeps the first and
     last transformer blocks in BF16), which costs end-to-end speedup and requires choosing layers.
  2. Software blockwise FP8 (accurate, deck 46): scales applied on CUDA cores after every 128-element MMA
     chunk; mixed 1x128 / 128x128 layouts complicate kernels; block-scaled GEMMs emulated on Hopper.
- **Villain**: the scale lives outside the tensor core; each block's scale must be applied once per block
  dot-product, which is expensive in software (MXFP8 paper App. A.3).
- **Turn / insight**: Blackwell tensor cores consume the per-block scale inside the MMA, so a smaller block no
  longer costs speed; shrink the block to 32 and spend the effort on numerics (encoding, rounding).
- **Mechanisms**:
  1. MX format: 32 E4M3 values share one E8M0 (power-of-two) scale byte; hardware applies it.
     Challenge: granularity without overhead. Engineering consequence: rowwise (1x32) and columnwise (32x1)
     copies quantized separately from BF16.
  2. E4M3 for all tensors including activation gradients. Challenge: precision vs range; within 32 values
     E4M3's 17.8 binades suffice; E5M2 gradients degrade an 8B model.
  3. Round the scale up (ceil) instead of OCP v1's round-down. Challenge: overflow at the block max;
     round-down caused training issues (843M ablations).
- **Release**: < 0.50% validation-perplexity gap vs BF16 throughout an 8B / 15T-token run; 16B MoE within 0.1%
  final loss; on B300 MXFP8 GEMMs reach 1.44x over BF16 (vs 1.41x per-tensor current), and end to end GB200
  DeepSeek-V3 and Qwen3-235B train 1.22x faster than BF16 (tokens/s/GPU, best-tuned configs each).
- **Limits**: the paper's accuracy runs simulated MXFP8 on Hopper; two quantized copies per tensor (memory,
  quantize kernels); parameter all-gather stays BF16; per-tensor delayed is still faster at GEMM level (1.61x
  on B300); dims must be multiples of 32; Blackwell-only (SM 10.x).

## Slide plan (15 logical slides)

| # | Headline | Device |
|---|---|---|
| 1 | Title | |
| 2 | Blackwell moves FP8 scales into the tensor core | fig_ladder + callout |
| 3 | On Hopper, finer FP8 scales cost GEMM speed | big 1.41x |
| 4 | Software block scales detour through CUDA cores | fig_where_1 |
| 5 | Insight: let the tensor core apply the scale | fig_where_2 |
| 6 | MXFP8 quantizes every linear layer of every block | fig_overview |
| 7 | ❶ 32 values share one power-of-two scale byte | fig_mxblock |
| 8 | ❶ Every operand needs a row and a column copy | fig_two_copies |
| 9 | ❷ E4M3 everywhere, gradients included | fig_encoding |
| 10 | ❸ Round the scale up, not down | fig_rounding |
| 11 | MXFP8 tracks BF16 within 0.5% over 15T tokens | big <0.50% |
| 12 | MXFP8 trains DeepSeek-V3 1.22x faster than BF16 | chart_e2e |
| 13 | On B300, MXFP8 GEMMs match per-tensor speed | chart_gemm_b300 |
| 14 | The price: two copies, BF16 all-gather, simulation | bullets |
| 15 | Summary | numbers + QR |
