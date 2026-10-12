# Sources for deck 46 (FP8 training per-tensor / blockwise + FP8 activation compression)

Downloaded copies live in `$SCRATCH/research/46/` (TE docs as .html + .txt, FP8-formats paper) and the shared
`$SCRATCH/research/cache/` (DeepSeek-V3, Kimi K2, Megatron-Core MoE report, Megatron MoE README).

## Sources read

| Tag | Source | URL | Sections used |
|---|---|---|---|
| [TE-README] | Transformer Engine README (main, fetched 2026-10-13) | https://github.com/NVIDIA/TransformerEngine/blob/main/README.rst | Highlights (FP8 on Hopper/Ada/Blackwell/Rubin; MXFP8/NVFP4 on Blackwell) |
| [TE-primer] | TE "Using FP8 and FP4 with Transformer Engine" | https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html | Structure; Mixed precision training with FP8; recipes list; autocast |
| [TE-cur] | TE low-precision guide: FP8 Current Scaling | https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html | FP8 data type; Scaling factors; Performance analysis; Transpose handling; Distributed training |
| [TE-del] | TE low-precision guide: FP8 Delayed Scaling | .../fp8_delayed_scaling/fp8_delayed_scaling.html | Quantization with delayed scaling factors; Amax history; example defaults |
| [TE-blk] | TE low-precision guide: FP8 Blockwise Scaling | https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html | Data format; Block size; Scaling factors; Handling transposes; Distributed training; Supported devices |
| [TE-perf] | TE low-precision guide: Performance Considerations | .../performance_considerations/performance_considerations.html | Table 2 (GEMM operand usages); Architecture differences; Memory usage examples 1-2 |
| [TE-speed] | TE low-precision guide: GEMM Speedups Across Precisions | .../low_precision_training/speedups.html | "Example: 5B Model on H200 (Hopper)" and "on B300" |
| [FP8-fmt] | Micikevicius et al., "FP8 Formats for Deep Learning", arXiv 2209.05433 | https://arxiv.org/abs/2209.05433 | Table 1; Sec. 3.1; Sec. 4.1; Table 4 |
| [DSV3] | DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv 2412.19437 | https://arxiv.org/abs/2412.19437 | Sec. 3.3 (FP8 training), Fig. 6, Fig. 7; App. B.1, B.2 |
| [K2] | Kimi Team, "Kimi K2: Open Agentic Intelligence", arXiv 2507.20534 | https://arxiv.org/abs/2507.20534 | Sec. 2.4.1 (cluster), Sec. 2.4.3 (Activation Reduction) |
| [MCore-MoE] | NVIDIA, "Scalable Training of MoE Models with Megatron Core", arXiv 2603.07685 | https://arxiv.org/abs/2603.07685 | Table 3; Sec. 4.1.3; Sec. 4.3.5; Sec. 5.1-5.3; Sec. 5.4.1; Table 11 |
| [MCore-README] | Megatron-LM MoE README | https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md | "FP8 Training" tables; Communication optimization table; MCore v0.14 news |

## Numbers on slides (with exact location)

### Formats (slide 2 figure, slide 4-5)
- E4M3: 1 sign, 4 exponent, 3 mantissa bits; max +-448; min normal 2^-6; min subnormal 2^-9; exponent bias 7 —
  [FP8-fmt] Table 1; [TE-cur] "FP8 data type".
- E5M2: 1 sign, 5 exponent, 2 mantissa bits; max +-57,344; min subnormal 2^-16 — [FP8-fmt] Table 1.
- E4M3 covers 18 binades (17 without reclaiming special values); E5M2 covers 32 binades incl. subnormals —
  [FP8-fmt] Sec. 3.1.
- BF16: 1 sign, 8 exponent, 7 mantissa bits; same exponent range as FP32 — [TE intro] "Training in BF16/FP16"
  (lpt_introduction).
- Recommended use: E4M3 for weights/activations, E5M2 for gradients (hybrid) — [FP8-fmt] Sec. 3; [TE-cur].

### Pain (slide 3)
- **1.92x**: "In pre-quantized mode, raw FP8 reaches 1.92x over BF16" on NVIDIA H200 NVL, 5B-parameter model
  config (hidden 4096, intermediate 16384, 24 layers) — [TE-speed] "Example: 5B Model on H200 (Hopper)".
- Supporting: FP8 GEMMs "theoretically double" compute speed vs BF16 — [DSV3] Sec. 3.3.1.

### Status quo (slides 4-5)
- Single FP16-style loss scale is infeasible for FP8; each FP8 tensor needs its own scale — [TE-primer]
  "Mixed precision training with FP8".
- Delayed scaling: scale from amax history (default `amax_history_len=1024`), 1 tensor read; current scaling:
  2 tensor reads — [TE-del], [TE-cur] "Performance analysis".
- Megatron-Core does not recommend delayed scaling "due to precision issues" — [MCore-MoE] Sec. 5.3.1.
- Per-tensor scaling "makes low-precision training highly sensitive to activation outliers" — [DSV3] Sec. 3.3.2.
- **Worked example (analytic, slides 4-5 figure)**: values {0.012, -0.004, 0.021, 0.0007 | 1.5, -0.9, 2400,
  0.6}, quantized to E4M3 with round-to-nearest-even and an exact FP32 scale s = 448/amax.
  Per-tensor (amax 2400): tile 1 dequantizes to {0.0105, 0, 0.0209, 0}; tile 2 to {1.51, -0.92, 2400, 0.59}.
  Per-tile scales (amax 0.021 and 2400): tile 1 {0.012, -0.0041, 0.021, 0.0007}; tile 2 unchanged.
  Computed by `$SCRATCH/scripts46/e4m3.py` from the E4M3 spec in [FP8-fmt] Table 1. Illustrative, not measured.

### Mechanism ❶ (slides 7-8)
- Blockwise recipe: block size 128; 1D (1x128) for activations and gradients, 2D (128x128) for weights;
  E4M3 for forward and backward by default; FP32 scales, power of 2 by default; s_block = max_fp8 / amax_block
  rounded down to a power of 2 — [TE-blk] "Data format", "Block size", "Scaling factors".
- DeepSeek-V3: 1x128 tiles "per token per 128 channels" for activations, 128x128 blocks "per 128 input
  channels per 128 output channels" for weights; E4M3 on all tensors; amax computed online — [DSV3] Sec. 3.3.2.
- 2D x 2D GEMMs not supported; 1D rowwise and columnwise blocks differ, both quantized from high precision;
  2D blocks transpose data+scales — [TE-blk] "Handling transposes".
- Hopper FP8 GEMMs need physically transposed columnwise operands (TN-only) — [TE-perf] "Architecture
  differences"; [MCore-MoE] Sec. 5.3.1.
- GEMM operands: Fprop Y = X W^T (X rowwise, W rowwise); Wgrad dW = dY^T X (both columnwise); Dgrad dX = dY W
  (dY rowwise, W columnwise) — [TE-perf] Table 2.
- 128x128 block quantization of activation gradients led to divergence on a ~16B-parameter MoE trained
  ~300B tokens — [DSV3] App. B.2.
- No scale synchronization needed across ranks for blockwise (scales are local) — [TE-blk] "Distributed training".

### Mechanism ❷ (slide 9)
- H800 FP8 GEMM accumulation retains "around 14 bits"; for K = 4096, max relative error "nearly 2%";
  promotion to FP32 CUDA-core registers every N_C = 128 elements (= 4 WGMMAs); per-group scales multiplied on
  CUDA cores during dequantization; two warpgroups overlap promotion and MMA — [DSV3] Sec. 3.3.2
  "Increasing Accumulation Precision", Fig. 7(b).

### Overview (slide 6)
- All three GEMMs (Fprop, Dgrad, Wgrad) in FP8; inputs FP8, outputs BF16 or FP32 — [DSV3] Sec. 3.3.1, Fig. 6.
- Kept in original precision: embedding, output head, MoE gating, normalization, attention; master weights,
  weight gradients, optimizer states in higher precision — [DSV3] Sec. 3.3.1; router in FP32, SDPA in BF16 —
  [MCore-MoE] Sec. 5.1, Sec. 5.3.
- 12 GEMMs per transformer layer (4 Fprop + 4 Dgrad + 4 Wgrad) — [TE-speed].

### Mechanism ❸ (slides 10-11)
- Kimi K2: inputs of MoE up-projections and SwiGLU compressed to FP8-E4M3 in 1x128 tiles with FP32 scales;
  "no measurable loss increase" in small-scale experiments; FP8 not applied in computation — [K2] Sec. 2.4.3.
  K2: 1T total / 32B activated, H800 cluster — [K2] abstract, Sec. 2.4.1.
- DeepSeek-V3: activations cached in FP8 for Wgrad; SwiGLU inputs cached FP8 and output recomputed; inputs of
  the linear after attention in a custom E5M6 format with power-of-2 scales; dispatch in FP8, combine in BF16 —
  [DSV3] Sec. 3.3.3.
- FP8/FP4 storage of linear-layer inputs reduces each tensor's footprint by 50%/75% — [MCore-MoE] Sec. 4.1.3.
- Byte cost of a 1x128 tile with an FP32 scale: (128 x 1 B + 4 B) / 128 = 1.03 B/element vs 2 B for BF16
  — **derived** from the tile format above (not shown as a measured number).
- FP8 primary weights: FP32 master shards cast directly to FP8, eliminating the BF16 copy; parameter AllGather
  1 byte vs 2 bytes per parameter (-50%); per-tensor recipes all-reduce the amax first; blockwise needs a
  2D-aware amax kernel; MXFP8 still all-gathers BF16 — [MCore-MoE] Sec. 5.2.2, Sec. 5.3.5; flag
  `--fp8-param-gather` — [MCore-README] "FP8 Training", "Communication" table ("reducing overhead by 50%").
- EP dispatch in FP8 halves dispatch volume — [MCore-README] "FP8 Training" table. (Deep dive: deck 18.)

### Results (slides 12-14)
- **< 0.25%** relative loss error of FP8 vs BF16 training; ablations on ~16B MoE / 1.33T tokens and ~230B MoE /
  ~0.9T tokens — [DSV3] Sec. 3.3 and App. B.1.
- Per-tensor FP8 (simulated: values clipped to FP8, arithmetic in 16-bit): GPT 175B perplexity 6.65 (16-bit
  baseline) vs 6.68 (FP8), reported at 75% of training — [FP8-fmt] Table 4, Sec. 4.1.
- GEMM speedup over BF16, H200 NVL, 5B model, autocast mode (includes quantization): FP8 DelayedScaling ~1.69x,
  FP8 CurrentScaling ~1.57x, FP8 Block Scaling ~1.41x; pre-quantized raw FP8 1.92x — [TE-speed] H200 example.
  Chart `charts/chart_gemm_speedup.py`. Slide 13 headline rounds 1.41-1.69x to "1.4-1.7x".
- End-to-end: FP8 training "approximately 10% to 25%" improvement for large-scale MoE — [MCore-MoE] Sec. 4.3.5.
- DeepSeek-V3 on 1,024 H100 with blockwise FP8 (FP8-BLK): 368 TFLOPS/GPU, 1,412 tokens/s/GPU — [MCore-MoE]
  Table 11.
- DeepSeek-V3 BF16 memory per GPU (PP4 x VPP4 x EP64, 256 GPUs): weights & grads 36.4 GB, main weights &
  optimizer states 32.1 GB, activations 131.0 GB, total 199.5 GB; H100 has 80 GB — [MCore-MoE] Table 3.
- FP8 saves ~16 GB of activation memory (~12% of 131 GB) by halving ~32 GB of eligible linear-layer inputs —
  [MCore-MoE] Sec. 4.1.3.
- **Derived** for chart_memory: activations with FP8 = 131.0 - 16 = 115 GB; total = 199.5 - 16 = 183.5 GB.

### Limits (slide 15)
- On Hopper, block-scaled GEMMs are software-emulated; mixed 1D/2D has lower overhead than 1D x 1D — [TE-blk]
  "Block size".
- Blockwise 1.41x vs delayed 1.69x on H200 — [TE-speed].
- FP8 does not always reduce memory: TE Linear 1024x1024 forward = 6.00 MB in BF16 vs 6.02 MB in FP8 with BF16
  weights (extra FP8 weight copy offsets the FP8 input) — [TE-perf] "Memory usage" examples 1-2.
- Alignment: dims divisible by 16 (TE Linear), last dim and leading product divisible by 128 (blockwise);
  MoE: 16 for per-tensor/blockwise, 32 for MXFP8/NVFP4, tokens-per-expert padding to 128 — [TE-primer] Note;
  [TE-blk] "Block size"; [MCore-MoE] Sec. 5.4.1.
- On Blackwell, Float8BlockScaling is emulated with MXFP8; "MXFP8 is the preferred recipe on Blackwell" —
  [TE-blk] "Supported devices". (Deck 47.)

### Context (slide 2, summary)
- DeepSeek-V3: 671B total / 37B activated parameters, 14.8T tokens, H800 — [DSV3] abstract, Sec. 3.
- Blockwise FP8 "production-proven" (DeepSeek-V3, Minimax-M2) and recommended on Hopper — [MCore-README]
  "FP8 Recipes"; also Ant Ling-2.0 — [MCore-MoE] Sec. 5.3.2.

## Discrepancies (Notion row vs primary sources; the primary source wins on the slides)

1. **"activation memory halved"** — The sources halve only the FP8-eligible tensors (linear-layer inputs saved
   for backward). For DeepSeek-V3 the whole activation budget drops by ~16 GB of 131 GB, i.e. ~12%
   ([MCore-MoE] Sec. 4.1.3). Kimi K2 reports no memory number at all, only "no measurable loss increase"
   ([K2] Sec. 2.4.3). Slides say "50% per cached tensor, ~12% of DeepSeek-V3's activations".
2. **"FP8 parameter all-gather cuts roughly 50% of that communication"** — Correct as a byte count (1 vs 2
   bytes/param) for per-tensor and blockwise FP8 ([MCore-MoE] Sec. 5.2.2), but it is a design property, not a
   measured result, and it does not hold for MXFP8, which all-gathers in BF16 because both quantization
   directions would be needed.
3. **"the default recipe for DeepSeek-V3-class models on H100"** — Megatron-Core recommends blockwise FP8 on
   Hopper and validated it on DeepSeek-V3 ([MCore-README]); DeepSeek-V3 itself was trained on H800, not H100
   ([DSV3]). Slides say "Hopper (H100/H800)".
4. **"K2 compresses activations in FP8-E4M3"** — True, but storage only: K2 explicitly does *not* compute in
   FP8 because of risks seen in preliminary studies ([K2] Sec. 2.4.3). The Notion row's framing under "FP8
   training" could suggest K2 trains with FP8 GEMMs; it does not.
5. **"TE provides DelayedScaling / blockwise FP8 recipes"** — TE also provides Float8CurrentScaling; Megatron-Core
   does not recommend DelayedScaling for precision reasons ([MCore-MoE] Sec. 5.3.1). Not a contradiction, an
   omission.
