# Sources for deck 48 (NVFP4 pretraining recipe)

Downloaded copies: `$SCRATCH/research/48/` (NVIDIA blogs, Megatron-Bridge `mixed_precision.py`),
`$SCRATCH/research/46/` (TE low-precision guide pages incl. NVFP4 and speedups), shared `$SCRATCH/research/cache/`
(NVFP4 paper text, Megatron-Core MoE report).

## Sources read

| Tag | Source | URL | Sections used |
|---|---|---|---|
| [NV4] | NVIDIA, "Pretraining Large Language Models with NVFP4", arXiv 2509.25149 v2 | https://arxiv.org/abs/2509.25149 | Abstract; Sec. 1-6; Table 1, Table 2; Fig. 2, 4, 5, 6; App. A.1, A.2, B.1-B.4, C, D, E.1-E.5 |
| [TE-nv4] | TE low-precision guide: NVFP4 | https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html | Data format; 2D weight scaling; Stochastic rounding; Random Hadamard Transform; Distributed training |
| [TE-primer] | TE "Using FP8 and FP4 with Transformer Engine" | https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html | NVFP4 format; NVFP4 training recipe; multiple recipes |
| [TE-speed] | TE low-precision guide: GEMM Speedups Across Precisions | https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/speedups.html | "Example: 5B Model on B300" |
| [MCore-MoE] | NVIDIA, "Scalable Training of MoE Models with Megatron Core", arXiv 2603.07685 | https://arxiv.org/abs/2603.07685 | Sec. 4.1.3; Sec. 5.2; Sec. 5.3.4; Sec. 5.4.1; Sec. 5.4.3 |
| [MB] | Megatron-Bridge `src/megatron/bridge/training/mixed_precision.py` (main, fetched 2026-10-13 via gh api) | https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/training/mixed_precision.py | `bf16_with_nvfp4_mixed`, `nemotron_3_super_...`, `nemotron_3_ultra_...` |
| [NV-blog-train] | NVIDIA blog "NVFP4 Trains with Precision of 16-Bit and Speed and Efficiency of 4-Bit" (Aug 2025) | https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit | recipe summary; "research phase" |
| [NV-3ways] | NVIDIA blog "3 Ways NVFP4 Accelerates AI Training and Inference" | https://developer.nvidia.com/blog/3-ways-nvfp4-accelerates-ai-training-and-inference/ | MLPerf Training Llama 3.1 405B paragraph |
| [NV-research] | NVIDIA Research blog "Pushing intelligence to 4-bit" | https://research.nvidia.com/labs/eai/blogs/pushing-intelligence-to-4-bit/ | Fig. 4 (cites [NV-3ways]); read for context only |

## Numbers on slides (with exact location)

### Format (slides 2, 4, 5, 7)
- E2M1: 1 sign, 2 exponent, 1 mantissa bit; magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 — [NV4] Sec. 2; max +-6 — [TE-nv4].
- NVFP4: 16-element blocks, E4M3 block scale, FP32 per-tensor scale; x = x_e2m1 x s_block x s_global —
  [NV4] Sec. 2; [TE-nv4] "Data Format".
- s_global = global_amax / (448 x 6); s_block = (block_amax / 6) / s_global stored as E4M3 — [TE-nv4];
  encode scale s_enc = 6 x 448 / amax_x — [NV4] App. B.1; block decode scale amax_b / 6 — [NV4] App. B.2.
- Tensor Cores apply the block scales to per-block partial dot products and the global scales after the GEMM —
  [NV4] App. B.3, Eq. 5.
- ">= 6.25% of values in a block (the amax values) at near-FP8 precision" — [NV4] Sec. 2 (1 of 16 = 6.25%).
- MXFP4: 32-element blocks with UE8M0 scales; power-of-two rounding can lose up to one binade and the samples +-4
  and +-6; worst case amax = 3 + delta uses 2.58 of 3.58 binades — [NV4] Sec. 2, App. B.4.
- Table 1: NVFP4 (E2M1, E4M3 scale, block 16) 4x on GB200 and 6x on GB300 vs BF16; MXFP8 2x / 2x — [NV4] Table 1.
  FP4 math 2x (GB200) and 3x (GB300) the FP8 rate; memory roughly halved vs FP8 — [NV4] Sec. 2.
- **Derived** bits per value: NVFP4 4 + 8/16 = 4.5 bits (per-tensor FP32 scale ignored); MXFP8 8.25 bits (deck 47).

### Context (slide 2)
- MLPerf Training: 512 Blackwell Ultra GPUs (GB300 NVL72) pretrained Llama 3.1 405B in **64.6 minutes** using
  NVFP4, 1.9x faster than the prior-round FP8 submission on 512 GB200 GPUs — [NV-3ways]. (Different GPU
  generations, so the 1.9x is not shown as an NVFP4-vs-FP8 speedup.)
- TE recipe `NVFP4BlockScaling()` (2D weight quantization and RHT on by default) — [TE-nv4] Examples.
- Megatron-Bridge `bf16_with_nvfp4_mixed`: fp4 = "e2m1", fp4_recipe = "nvfp4", fp4_param_gather = True;
  Nemotron-3 Super variant keeps the last 14 layers in BF16, Ultra the last 16 ("last 15% of layers") — [MB].

### Pain (slide 3)
- **36%**: "MXFP4 matches NVFP4 loss when trained on 36% more tokens (i.e., using 1.36T instead of 1T tokens)";
  8B hybrid model, same recipe, last 8 blocks BF16 — [NV4] Sec. 5, Fig. 6b.
- MXFP4 relative loss error ~2.5% vs NVFP4 1.5% (vs BF16, 8B, 1T tokens) — [NV4] Sec. 5, Fig. 6a.
- Base method (all linear layers NVFP4, 1x16 scales, RNE) diverges early; SR alone eventually diverges —
  [NV4] App. E.1; training diverges when every linear layer is FP4 — [NV4] Sec. 4.1.

### Overview (slide 6)
- Recommendations: ~15% of the network in high precision, mostly at the end; RHT 16x16 on Wgrad inputs; 2D 16x16
  scaling for weights, 1x16 for activations and gradients; SR for gradients, RNE for weights and activations —
  [NV4] Sec. 4.
- Compute flow of a quantized linear layer (Fprop, Dgrad, Wgrad all NVFP4 GEMMs; FP32 weights; BF16 activations and
  gradients) — [NV4] Fig. 5, Sec. 4.1.
- RHT d = 16 "better convergence than d = 4 and similar results as d = 128"; 128x128 gives a minor benefit —
  [NV4] Sec. 4.2, App. E.4.2. Weights "can adapt to the FP4 values" (coarser 2D blocks OK) — [NV4] Sec. 4.3.
- 12B model: 62 blocks; first 2 and last 8 blocks kept BF16 = 16% of linear layers; stable with only the last 4 —
  [NV4] Sec. 4.1, App. A.1, Fig. 4.
- Attention (softmax, QK and score-V BMMs), embeddings, output head, norms, non-linearities stay BF16/FP32; master
  weights, weight gradients, optimizer states FP32; TP reductions BF16 — [NV4] Sec. 4.1.

### Mechanisms (slides 7-10)
- ❶ two-level scaling — see Format above.
- ❷ 2D scaling: backward uses w_bprop != w_fprop with 1D scales (chain-rule violation); 16x16 weight blocks
  (16 input x 16 output channels), replicated per 1x16 for Tensor Cores; activations/gradients keep 1x16; 1D vs 2D
  weight scales compared in a 1.2B ablation — [NV4] Sec. 4.3, App. E.5; [TE-nv4] "2D weight scaling".
- ❸ RHT: x' = q(xH s); H = S_d H_d / normalized, H H^T = I so (AH)(H^T B) = AB; applied only to Wgrad inputs;
  d = 16 better than d = 4, similar to d = 128 (12B); one shared random sign vector — [NV4] Sec. 4.2, App. C,
  E.4; [TE-nv4] "Random Hadamard Transform" (applied to columnwise inputs/gradients for wgrad, BF16 inputs only).
- ❹ SR: rounds to one of the two nearest values with probability inversely proportional to distance; essential for
  gradients at 12B; SR on forward tensors is detrimental (divergence on 1.2B) — [NV4] Sec. 4.4, App. E.3;
  hardware SR on Blackwell, gradients only by default — [TE-nv4] "Stochastic Rounding".
  **Worked example (analytic)**: x = 1.2 between E2M1 samples 1 and 1.5 is 40% of the way up, so SR returns 1 with
  probability 0.6 and 1.5 with 0.4; E[q] = 0.6 x 1 + 0.4 x 1.5 = 1.2; RNE always returns 1 (TE docs use the same
  40%/60% illustration).

### Results (slides 11-13)
- Relative loss error of NVFP4 vs FP8 "consistently below 1%" in the stable phase, "slightly above 1.5%" at the end
  of LR decay; 12B, 10T tokens — [NV4] Sec. 3, Fig. 2.
- "the longest publicly documented training run in 4-bit precision to date" (12B, 10T tokens) — [NV4]
  Abstract (slide 11 bullet, paraphrased as "longest public 4-bit pretraining run at the time").
- **Derived** code gap: 59.52 - 56.67 = 2.85 points (shown as 2.9) — from [NV4] Table 2; the paper attributes the
  MBPP+ drop to noisy final-checkpoint evaluation — [NV4] Sec. 3.
- FP8 baseline = blockwise 1x128 / 128x128 E4M3 with first block and last two blocks in BF16 — [NV4] App. A.1.
- Table 2 (FP8 vs NVFP4): General 68.99 / 69.82; Math 86.20 / 86.88; Multilingual 77.93 / 80.24; Code 59.52 /
  56.67; Commonsense Understanding 77.29 / 76.75; MMLU-Pro 5-shot 62.62 / 62.58; MBPP+ 59.11 / 55.91; MMLU
  77.36 / 76.57 — [NV4] Table 2. Chart `charts/chart_downstream.py`.
- B300, 5B model: NVFP4 ~2.03x over BF16 in autocast, 3.55x pre-quantized; MXFP8 ~1.44x; FP8 delayed 1.61x —
  [TE-speed]. **Derived**: 2.03 / 1.44 = 1.41x NVFP4 over MXFP8. Chart `charts/chart_gemm_b300.py`.

### Limits (slide 14)
- 16% of linear layers in BF16 (12B) — [NV4] App. A.1.
- Global amax needs an extra pass through memory — [NV4] App. B.1; Hadamard computed twice (amax kernel + fused
  quantize kernel); fused kernel emits two FP4 copies (forward and transposed+RHT for Wgrad) — [MCore-MoE]
  Sec. 5.4.3.
- Quantization overhead: 3.55x pre-quantized vs 2.03x autocast — [TE-speed].
- Switching to BF16 at 8.2T tokens (18% of training) matches FP8 loss; forward-only switch recovers from 1.5% to
  0.5% relative error with ~6% of total compute in higher precision — [NV4] App. D, Fig. 7.
- Future work: MoE, larger models, quantizing all linear layers, attention and communication in NVFP4 — [NV4]
  Sec. 6. "still in the research phase" — [NV-blog-train].
- Parameter all-gather in NVFP4 gathers row- and column-wise FP4 weights, so traffic is 50% of BF16 —
  [MCore-MoE] Sec. 5.2.2; FP4 activations save 75% vs BF16 — [MCore-MoE] Sec. 4.1.3.

## Discrepancies (Notion row vs primary sources; the primary source wins on the slides)

1. **"a further 1.31-1.73x over FP8 on GB200/GB300"** — Not found in the NVFP4 paper (which states it is about
   algorithms, not runtime), the TE docs, the Megatron-Core report or the three NVIDIA blogs read. What the sources
   give: peak FP4 math 2x (GB200) / 3x (GB300) the FP8 rate ([NV4] Table 1, Sec. 2); measured GEMM speedup on B300
   NVFP4 2.03x vs BF16, i.e. 1.41x over MXFP8 and 1.26x over FP8 delayed scaling (derived from [TE-speed]); MLPerf
   405B at 1.9x vs a prior FP8 round on older GPUs ([NV-3ways]). Slides use the TE GEMM numbers.
2. **"<1% loss gap vs FP8 over 10T tokens (~1.5% late in training)"** — Consistent with [NV4] Sec. 3 ("below 1%"
   in the stable phase, "slightly above 1.5%" at the end of decay).
3. **"E2M1 with two-dimensional block scaling"** — The NVFP4 *format* uses 1x16 E4M3 block scales plus an FP32
   tensor scale; 2D (16x16) scaling is a recipe choice applied only to weights ([NV4] Sec. 4.3; [TE-nv4]).
4. **"Random Hadamard transforms spread outliers"** — Applied only to Wgrad inputs (16x16); Fprop/Dgrad transforms
   gave no benefit or hurt in ablations ([NV4] Sec. 4.2, App. E.4.1).
5. **"sensitive layers stay high precision (bf16_with_nvfp4_mixed)"** — In Megatron-Bridge, `bf16_with_nvfp4_mixed`
   only switches GEMMs to NVFP4 (fp4 e2m1, FP4 param gather); BF16 layers are set separately
   (`first_last_layers_bf16`, `num_layers_at_end_in_bf16`; the Nemotron-3 Super/Ultra variants keep the last 14/16
   layers) ([MB]). The paper's 12B kept the first 2 and last 8 blocks (16% of linear layers) ([NV4] App. A.1).
6. Notion omits that the FP8 baseline itself is the blockwise DeepSeek-style recipe with BF16 first/last blocks
   ([NV4] App. A.1), and that evidence is a 12B dense hybrid model; MoE is listed as future work ([NV4] Sec. 6).
