# Sources for deck 47 (MXFP8 micro-scaling training)

Downloaded copies: `$SCRATCH/research/47/` (MXFP8 paper v1 HTML, NVIDIA per-block scaling blog) and
`$SCRATCH/research/46/` (TE low-precision guide pages), plus the shared `$SCRATCH/research/cache/`
(MXFP8 paper v2 text, Megatron-Core MoE report, Megatron MoE README).

## Sources read

| Tag | Source | URL | Sections used |
|---|---|---|---|
| [MX] | Mishra, Stosic, Layton, Micikevicius, "Recipes for Pre-training LLMs with MXFP8", arXiv 2506.08027 v2 | https://arxiv.org/abs/2506.08027 | Sec. 2 + Table 1; Sec. 3.1-3.3, Fig. 2-5, Algorithm 1; Sec. 4; Sec. 5; App. A.1-A.5 |
| [MX-v1] | same paper, v1 HTML | https://arxiv.org/html/2506.08027v1 | Sec. 3.1 "MXFP8 versus FP8" (first/last block in BF16) |
| [TE-mx] | TE low-precision guide: MXFP8 | https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/mxfp8/mxfp8.html | Data format; Block size; Scaling factors; Handling transposes; Distributed training; Supported devices; Swizzling |
| [TE-blk] | TE low-precision guide: FP8 Blockwise Scaling | .../fp8_blockwise_scaling/fp8_blockwise_scaling.html | Block size (software emulation on Hopper); Supported devices |
| [TE-speed] | TE low-precision guide: GEMM Speedups Across Precisions | .../low_precision_training/speedups.html | 5B model on H200 and on B300 |
| [TE-primer] | TE "Using FP8 and FP4 with Transformer Engine" | https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html | MXFP8 vs FP8; Handling transposes |
| [TE-perf] | TE low-precision guide: Performance Considerations | .../performance_considerations/performance_considerations.html | Table 2 (GEMM operand usages) |
| [MCore-MoE] | NVIDIA, "Scalable Training of MoE Models with Megatron Core", arXiv 2603.07685 | https://arxiv.org/abs/2603.07685 | Sec. 5.2.2; Sec. 5.3.3; Sec. 5.4.1; Table 11; App. B Table 20 |
| [MCore-README] | Megatron-LM MoE README | https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md | FP8 Recipes table; example flags |
| [NV-blog] | NVIDIA blog "Per-Tensor and Per-Block Scaling Strategies for Effective FP8 Training" (Jul 2025) | https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training | "How does MXFP8 work?" |
| [DSV3] | DeepSeek-V3 Technical Report, arXiv 2412.19437 | https://arxiv.org/abs/2412.19437 | Sec. 3.3.2 (promotion every 128 elements on CUDA cores) |

## Numbers on slides (with exact location)

### Format (slides 2, 7)
- MX block: K = 32 elements share one scale X of type UE8M0 (power of two, 2^-127 .. 2^127) — [MX] Sec. 2.
- MXFP8 E4M3: max normal 1.75 x 2^8 (= 448), min subnormal 2^-9, **17.8 binades**; E5M2: 1.75 x 2^15, 2^-16,
  31.8 binades; relative Tensor Core math speed vs BF16: **2x** for MXFP8 (4x for MXFP4) — [MX] Table 1.
- Decode: value = Q_i x X; quantize Q_i = Quantize_to_fp8(V_i / X) with round-to-nearest-even, saturating —
  [MX] Sec. 2, Sec. 3.3, App. A.2.
- E8M0 = 8 exponent bits, 0 mantissa bits, bias 127; s_block = 2^(e-127) — [TE-mx] "Scaling factors".
- Hardware applies the scale inside the dot product ("Tensor Cores in Blackwell consume X and Q_i") —
  [MX] Sec. 2; "scaling is handled in the hardware" — [MX] Sec. 3.1.
- **Derived** bits per element: MXFP8 = 8 + 8/32 = 8.25 bits; blockwise FP8 1x128 with FP32 scale =
  8 + 32/128 = 8.25 bits; NVFP4 (1x16, E4M3 scale) = 4 + 8/16 = 4.5 bits (per-tensor FP32 scale ignored).
  Format facts from [MX] Sec. 2, [TE-blk], [TE-primer].
- Ladder granularity: per-tensor FP32 scale; blockwise 1x128 / 128x128 FP32 scales (software on Hopper); MXFP8
  1x32 E8M0 (hardware); NVFP4 1x16 E4M3 + FP32 tensor scale — [TE-primer]; [TE-blk]; [TE-mx]; [MCore-README].

- Ladder figure column "scale applied": per-tensor scales factor out of the dot product, so they are applied once
  per GEMM (**derived** from x = x_fp8 x s, [TE-cur] in deck 46); blockwise scales on Hopper are applied on CUDA
  cores during promotion ([DSV3] Sec. 3.3.2; software emulation per [TE-blk]); MX/NVFP4 scales inside the tensor
  core ([MX] Sec. 2; NVFP4 paper App. B.3, deck 48).

### Context (slide 2)
- MXFP8 is the default FP8 recipe on Blackwell in Megatron-Core — [MCore-MoE] Sec. 5.3.3;
  "Native hardware support on GB200" — [MCore-README] "FP8 Recipes".
- DeepSeek-V3 MXFP8 on 256 GB200: **1,048 TFLOPS/GPU**; on 256 GB300: **1,233 TFLOPS/GPU** — [MCore-MoE] Table 11.

### Pain (slide 3)
- H200 NVL, 5B model, autocast: FP8 Block Scaling **~1.41x**, CurrentScaling **~1.57x**, DelayedScaling
  **~1.69x** over BF16 — [TE-speed] "Example: 5B Model on H200". (Same numbers as deck 46.)
- Block-scaled GEMMs are software-emulated on Hopper — [TE-blk] "Block size".

### Status quo / insight (slides 4-5)
- Hopper fine-grained FP8: partial sums promoted from Tensor Cores to FP32 CUDA-core registers every 128
  elements of K, where per-group scales are multiplied — [DSV3] Sec. 3.3.2.
- "scale factors are processed once after each block's dot-product ... expensive to do in software, so hardware
  needs to add support" — [MX] App. A.3 (paraphrased on slide).
- Per-tensor FP8 recipes leave some transformer layers in BF16 — [MX] Sec. 3.2.1; the FP8 baseline in [MX-v1]
  Sec. 3.1 keeps the first and last transformer blocks in BF16 (Nemotron-H guideline), which "affects
  end-to-end speedup".
- "hardware support for scaling avoids any tradeoff between smaller block sizes and hardware speed" —
  [MX] Sec. 4 (paraphrased as the insight).

### Overview (slide 6)
- MXFP8 GEMMs: QKV, attention projection, FFN (FC1, FC2) in all transformer blocks; BMM1/BMM2, softmax,
  activation, residual add, layer norm, input embedding and output projection stay BF16/FP16 — [MX] Sec. 3.2.1,
  Fig. 3.

### Mechanism ❶ (slides 7-8)
- Two copies per tensor, quantized along row and column (the two reduction axes) — [MX] Sec. 3.2.1, Sec. 5.
- Rowwise 1x32 vs columnwise 32x1 blocks are numerically different and both quantized from full precision;
  Blackwell supports TN/NT/NN layouts so no physical transpose — [TE-mx] "Handling transposes";
  "automatic requantization from the high-precision input" — [NV-blog].
- GEMM operand usage: Fprop X row x W row; Dgrad dY row x W col; Wgrad X col x dY col — [TE-perf] Table 2
  (TE Performance Considerations, see deck 46 sources).
- Parameter all-gather in BF16 for MXFP8 (communicating both MXFP8 copies would cost 2 bytes/param) —
  [MCore-MoE] Sec. 5.2.2; flag `--reuse-grad-buf-for-mxfp8-param-ag` — [MCore-README].
- Dims: last dim and leading product divisible by 32 — [TE-mx] "Block size"; MoE alignment 32 — [MCore-MoE]
  Sec. 5.4.1.
- No scale synchronization across ranks — [TE-mx] "Distributed training".

### Mechanism ❷ (slide 9)
- MXFP8 recipe uses E4M3 for weights, activations and activation gradients; coarser-scaled FP8 used E5M2 for
  gradients; 17.8 binades of E4M3 suffice at 32-element blocks — [MX] Sec. 3.2.2.
- 843M / 1T tokens: E5M2 for weights or activations gives worse perplexity; 8B / 1T tokens: E5M2 activation
  gradients lead to "substantial degradation" — [MX] Sec. 3.2.2, Fig. 4 (curves only, no numeric gap given).

### Mechanism ❸ (slide 10)
- Algorithm 1: X = 2^ceil(log2(amax / destmax)), clamped to [-127, 127] in exponent; OCP v1 effectively rounds
  down, so scaled values can overflow; OCP rounding showed training issues for 843M (300B and 1T tokens),
  removed by round-up — [MX] Sec. 3.3, Fig. 5, App. A.1. Blackwell has hardware support for this rounding —
  [MX] App. A.2.
- **Worked example (analytic)**: block amax = 480, destmax = 448: 480/448 = 1.07; round-down -> X = 1 ->
  480 exceeds 448 and saturates to 448 (6.7% error); round-up -> X = 2 -> 240, exactly representable in E4M3
  (step 16 in [128, 256)) -> decodes to 480. Computed with `$SCRATCH/scripts46/e4m3.py` (E4M3 spec).

### Results (slides 11-13)
- **< 0.50%** difference between MXFP8 and BF16 validation perplexity throughout pre-training of an 8B model on
  15T tokens; downstream MMLU and 9-task reasoning scores match — [MX] Sec. 3.1, Fig. 2.
- 8B model: 32 blocks, hidden 4096; trained with Megatron-LM on 3,072 Hopper GPUs with MX operations
  **simulated** (BF16 -> MXFP8 -> BF16 before each GEMM) — [MX] Sec. 3.1.
- MoE: 16B total / ~2.5B active, 1T tokens: final loss **within 0.1%** of BF16 — [MX] App. A.4.
- GB200, 256 GPUs, 4k seqlen (Table 11): DeepSeek-V3 BF16 857 TF, 3,298 tokens/s/GPU; MXFP8 1,048 TF, 4,020
  tokens/s/GPU. Qwen3-235B BF16 750 TF, 5,100 tokens/s/GPU; MXFP8 919 TF, 6,212 tokens/s/GPU —
  [MCore-MoE] Table 11. **Derived** speedups (tokens/s): 4,020/3,298 = 1.22x; 6,212/5,100 = 1.22x.
  BF16 and MXFP8 rows use different best-found parallel configs (e.g. DeepSeek-V3 BF16 PP8 EP32 GBS4096 vs
  MXFP8 PP4 EP64 GBS8192) — [MCore-MoE] App. B, Table 20.
- B300, 5B model, autocast: MXFP8 **~1.44x**, FP8 CurrentScaling **1.41x**, FP8 DelayedScaling **1.61x**,
  NVFP4 **~2.03x** over BF16 — [TE-speed] "Example: 5B Model on B300".

- Delayed scaling not recommended for precision reasons — [MCore-MoE] Sec. 5.3.1 (slide 13 callout).

### Limits (slide 14)
- Simulation on Hopper for the accuracy runs — [MX] Sec. 3.1.
- Two copies -> storage overhead; future work on scaling methods that lower it — [MX] Sec. 5; extra
  column-wise FP8 data stored for activations and weights — [MCore-MoE] Sec. 5.3.3.
- BF16 parameter all-gather — [MCore-MoE] Sec. 5.2.2.
- Supported devices SM 10.0, SM 10.3 — [TE-mx].
- Ongoing MoE work: grouped quantization, fusing activation+quantization into GEMM, CUDA-graphable MXFP8
  pipeline — [MCore-MoE] Sec. 5.3.3.

## Discrepancies (Notion row vs primary sources; the primary source wins on the slides)

1. **"1.28-1.37x over BF16 on B200"** — Not found in the MXFP8 paper (v1 or v2), the TE docs, the Megatron-Core
   report or README, or the NVIDIA scaling blog. The paper reports no measured speedup at all (its accuracy runs
   were simulated on Hopper). Measured numbers available: end-to-end 1.22x tokens/s on GB200 for DeepSeek-V3
   and Qwen3-235B (derived from [MCore-MoE] Table 11) and GEMM-level 1.44x on B300 ([TE-speed]). Slides use
   these.
2. **"up to 2x on GEMMs"** — The 2x is the peak Tensor Core math rate of MXFP8 vs BF16 ([MX] Table 1), not a
   measured GEMM speedup. Slides call it "peak math rate".
3. **"loss curves indistinguishable from BF16"** — The paper quantifies "< 0.50% validation-perplexity difference"
   (8B, 15T tokens) and "within 0.1%" final loss (16B MoE, 1T tokens), with MXFP8 numerics simulated on Hopper.
4. **"requires keeping both transposed and non-transposed copies"** — Correct, but they are not transposes of each
   other: rowwise (1x32) and columnwise (32x1) copies are quantized separately from high precision; Blackwell
   needs no physical transpose ([TE-mx]).
5. **"covers more layers than per-tensor FP8"** — Consistent: MXFP8 quantizes all transformer blocks, whereas the
   per-tensor FP8 baseline keeps the first and last blocks in BF16 ([MX-v1], [MX] Sec. 3.2.1). Embedding, output
   layer and attention BMMs remain BF16 in MXFP8 too.
6. Notion lists only TransformerEngine as source; Megatron-Core is where MXFP8 is the Blackwell default and where
   the end-to-end numbers come from.
