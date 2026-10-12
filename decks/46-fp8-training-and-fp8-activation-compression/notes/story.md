# Story: FP8 training (per-tensor / blockwise) and FP8 activation compression

Rung 1 of the precision ladder FP8 (this deck, Hopper) -> MXFP8 (deck 47, Blackwell) -> NVFP4 (deck 48, Blackwell).

- **Where it lives** (prosperity): Hopper-generation (H100/H800) pretraining of DeepSeek-V3-class MoE models.
  DeepSeek-V3 (671B MoE, 14.8T tokens on H800) runs all three linear-layer GEMMs in FP8; Megatron-Core trains
  DeepSeek-V3 on 1,024 H100s with blockwise FP8 at 368 TFLOPS/GPU; Kimi K2 (1T MoE, H800) stores selected
  activations in FP8. Transformer Engine (TE) ships the recipes (DelayedScaling, Float8CurrentScaling,
  Float8BlockScaling) behind one `te.autocast` context.
- **Pain number** (betrayal): raw FP8 GEMMs are **1.92x** faster than BF16 on H200 for 5B-model shapes
  (TE "GEMM Speedups Across Precisions", pre-quantized). BF16 training leaves that throughput idle.
- **Failed suitors**:
  1. FP16-style mixed precision with one global loss scale: FP8 range fits any one tensor but not all at once,
     so a single scale is infeasible (TE FP8 primer).
  2. Per-tensor scaling (delayed or current): one amax per tensor, so a single outlier sets the scale and pushes
     small values below E4M3's 2^-9 floor; delayed scaling uses a stale amax history (Megatron-Core does not
     recommend it); gradients must fall back to E5M2 (2 mantissa bits).
- **Villain**: a granularity mismatch. Outliers are local (per token / per channel, DeepSeek-V3 3.3.2) but the
  scale is global (one per tensor). Second, a hardware assumption: H800 FP8 tensor cores accumulate with only
  ~14 bits, giving nearly 2% max relative error at K = 4096.
- **Turn / insight**: give every 128 values their own scale; small groups effectively share exponent bits, so
  E4M3 suffices for activations, weights and gradients (DeepSeek-V3 "Mantissa over exponents").
- **Mechanisms**:
  1. Fine-grained scales: 1x128 tiles for activations and gradients, 128x128 blocks for weights, FP32
     (power-of-2) scales computed online. Challenge answered: outliers + transposes (1D tiles must be quantized
     twice; 2D weight blocks transpose for free; 128x128 for activation gradients diverged).
  2. FP32 promotion every 128 K-elements: partial sums move from tensor cores to FP32 registers on CUDA cores,
     where the per-group scales are applied. Challenge answered: 14-bit accumulation error.
  3. FP8 storage and communication: cache linear-layer inputs in FP8 (Kimi K2 stores them in FP8 while
     computing in BF16), cast FP32 master weights straight to FP8 and all-gather 1 byte/param.
     Challenge answered: activation memory and parameter traffic.
- **Release**: blockwise FP8 keeps relative loss error below 0.25% vs BF16 (DeepSeek-V3, 16B/1.33T and
  230B/0.9T ablations) while TE measures 1.41x (blockwise) to 1.69x (delayed) GEMM speedup over BF16 on H200
  including quantization; Megatron-Core reports ~10-25% end-to-end for large-scale MoE; FP8 caching saves
  ~16 GB/GPU of DeepSeek-V3 activations.
- **Limits**: Hopper emulates block scaling in software (blockwise 1.41x vs per-tensor 1.69x); 1D tensors need
  two FP8 copies; FP8 does not automatically reduce memory (TE example 6.02 MB vs 6.00 MB); dims must be
  multiples of 128 (blockwise) and MoE token counts need padding; router/embedding/head/attention stay
  high precision; Kimi K2 kept compute in BF16 after seeing risk; 16 GB of 131 GB activations is only 12%.
  On Blackwell, MXFP8 (deck 47) replaces the emulated blockwise recipe.

## Slide plan (16 logical slides, 8 minutes)

| # | Headline | Device |
|---|---|---|
| 1 | Title | front matter |
| 2 | Hopper pretraining runs its GEMMs in 8-bit floats | fig_formats (bit layouts) |
| 3 | BF16 leaves the FP8 tensor cores idle | big 1.92x |
| 4 | One scale per tensor lets outliers erase values | fig_outlier_1 |
| 5 | Insight: outliers are local, so scales must be local | fig_outlier_2 + callout |
| 6 | Three FP8 GEMMs per layer; sensitive ops stay high | fig_overview |
| 7 | ❶ Scale each 1x128 tile and 128x128 block | fig_granularity |
| 8 | ❶ Row tiles must be quantized twice | fig_transpose + callout (B.2 divergence) |
| 9 | ❷ Promote partial sums to FP32 every 128 | fig_promotion |
| 10 | ❸ Cache activations in FP8, even for BF16 GEMMs | fig_storage |
| 11 | ❸ FP8 master casts halve parameter all-gather bytes | fig_paramag |
| 12 | Blockwise FP8 stays within 0.25% of BF16 loss | big <0.25% |
| 13 | FP8 GEMMs run 1.4-1.7x faster on H200 | chart_gemm_speedup |
| 14 | FP8 caching trims 16 GB of DeepSeek-V3 activations | chart_memory |
| 15 | The price: emulated scaling, double copies, padding | bullets |
| 16 | Summary | numbers + QR |
