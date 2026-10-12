# Story: NVFP4 pretraining recipe

Rung 3 of the precision ladder FP8 (deck 46, Hopper) -> MXFP8 (deck 47, Blackwell) -> **NVFP4 (this deck)**.

- **Where it lives** (prosperity): Blackwell GB200 / GB300 / B300. FP4 tensor cores run at 4x (GB200) / 6x (GB300)
  the BF16 rate, 2x / 3x FP8. NVIDIA pretrained a 12B hybrid Mamba-Transformer on 10T tokens in NVFP4; TE ships
  `NVFP4BlockScaling`, Megatron-Bridge ships `bf16_with_nvfp4_mixed`; MLPerf Training: Llama 3.1 405B pretrained
  in NVFP4 on 512 GB300 GPUs in 64.6 minutes.
- **Pain number** (betrayal): 4 bits leave no slack. The straightforward MX 4-bit format (MXFP4, power-of-two
  scale per 32) needs **36% more tokens** to reach NVFP4's loss on an 8B model (1T-token budget), even with the full
  recipe; plain NVFP4 with round-to-nearest on every layer diverges early.
- **Failed suitors**:
  1. MXFP4: E8M0 power-of-two scale rounded up; worst case the block max lands at 3, so FP4's +-4 and +-6 samples go
     unused and only 2.58 of 3.58 binades are used. 2.5% loss gap vs BF16 (NVFP4 1.5%).
  2. "Just quantize every linear layer with nearest rounding" (the base method): diverges early; SR alone delays but
     does not prevent divergence.
- **Villain**: E2M1 has only 8 magnitudes (0, 0.5, 1, 1.5, 2, 3, 4, 6), so every error source that FP8 could absorb
  becomes visible: coarse scales, block outliers, forward/backward mismatch, rounding bias, sensitive last layers.
- **Turn / insight**: spend bits on the scale, not on the elements: an E4M3 scale per 16 values puts each block's
  max at near-FP8 precision (>= 6.25% of values), and a per-tensor FP32 scale keeps those E4M3 scales in range.
  Then fix the remaining error sources one by one, each where it hurts (paper Sec. 4: each component is needed at
  12B / 10T).
- **Mechanisms**:
  1. Two-level scaling (FP32 per tensor x E4M3 per 1x16) -> scale precision.
  2. 2D 16x16 weight scales -> forward and backward use the same quantized W (chain rule).
  3. Random Hadamard transform (16x16) on Wgrad inputs -> block outliers spread.
  4. Stochastic rounding on gradients (RNE elsewhere) -> unbiased gradients.
  Plus: ~15-16% of linear layers (mostly the last blocks) stay BF16.
- **Release**: 12B / 10T tokens: relative loss error vs FP8 < 1% during the stable phase, slightly above 1.5% at
  the end; MMLU-Pro 62.58 vs 62.62; GEMM speedup on B300 2.03x over BF16 (1.41x over MXFP8, derived), 3.55x raw.
- **Limits**: residual ~1.5% loss gap and coding -3 points (MBPP+ 55.91 vs 59.11); 16% of linear layers in BF16;
  quantization work (global amax pass, RHT computed twice, two FP4 copies) eats much of the raw 3.55x; evidence on a
  12B hybrid dense model, MoE and larger models are future work; remedy: forward pass in BF16 from 8.2T tokens cuts
  the gap to 0.5% for ~6% of compute.

## Slide plan (15 logical slides)

| # | Headline | Device |
|---|---|---|
| 1 | Title | |
| 2 | NVFP4: 16 four-bit values share an FP8 scale | fig_block + callout |
| 3 | Naive 4-bit formats waste tokens or diverge | big 36% |
| 4 | Power-of-two scales waste FP4's top samples | fig_samples_1 |
| 5 | Insight: spend bits on the scale, not the values | fig_samples_2 |
| 6 | The recipe pairs one fix with each error source | fig_overview |
| 7 | ❶ A tensor scale keeps block scales in E4M3 range | fig_twolevel |
| 8 | ❷ 16x16 weight blocks keep both passes consistent | fig_2d |
| 9 | ❸ Rotate Wgrad inputs to spread outliers | fig_rht |
| 10 | ❹ Round gradients stochastically to remove bias | fig_sr |
| 11 | NVFP4 tracks FP8 loss within 1.5% over 10T tokens | big |
| 12 | Downstream accuracy matches FP8 | chart_downstream |
| 13 | NVFP4 GEMMs run 2.03x BF16 on B300 | chart_gemm_b300 |
| 14 | The price: BF16 layers, quantize work, a residual gap | bullets |
| 15 | Summary | numbers + QR |
