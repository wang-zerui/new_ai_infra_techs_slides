# Story — deck 30, FP8/FP4/BF16 dense GEMM + UE8M0 scaling layout

- **Where it lives**: every dense projection of a DeepSeek-class model. V3 pre-trained 14.8T tokens
  with FP8 linear layers; V4 adds FP4 expert weights. E4M3 gives 3 mantissa bits and a maximum normal
  of 448 (S5.1).
- **Pain number**: FP8 accumulation inside the H800 tensor core keeps only ~14 bits, which costs up to
  **~2% maximum relative error** at K = 4096 (S4.3) — before any quantization error from outliers.
- **Failed suitors**:
  1. Per-tensor (delayed) scaling, the FP8-formats recipe (S5.3): one scale per tensor, so a single
     activation outlier squashes everything else; V3 calls FP8 training "highly sensitive to
     activation outliers" (S4.1).
  2. Scale per group and hope the library takes it: "per-group scaling factors along the inner
     dimension ... is not directly supported in the standard FP8 GEMM" (S4.2). Accumulating in FP32
     registers does not help either, because the truncation happens inside the MMA.
- **Villain**: a mismatch of altitude. Quantization wants to act every 128 elements along K, inside the
  GEMM's inner loop; a BLAS interface only lets you scale outside it, and the tensor core quietly
  truncates partial products on the way.
- **Turn / insight**: if every scale factor is an exact power of two, applying it is a cheap exponent
  add and the factor itself fits in 8 bits (UE8M0). It can then become a *streamed operand* of the
  inner loop: TMA-copied next to the data tile and consumed either by CUDA cores (SM90) or by the
  block-scaled tensor core (SM100).
- **Mechanisms**:
  ❶ **The recipe**: SFA covers `gran_m x gran_k` of A, SFB `gran_n x gran_k` of B; V3 uses 1x128 for
     activations and 128x128 for weights; SM100 also accepts 1x32 MX granularity (S2.1, S4.1).
  ❷ **Applying the scale inside the K loop**: SM90 promotes to CUDA cores every 128 elements (4
     WGMMAs) and multiplies by `scale_a * scale_b` there (S4.3, S3.2); SM100 hands the packed UE8M0
     factors to the block-scaled UMMA through tensor memory (S3.3).
  ❸ **The layout contract**: values are exact powers of two, 4 exponents packed per int32, MN-major
     with `stride(-1) = align(mn, 4)` so TMA sees 16-byte alignment; transform once and cache, or emit
     the packed layout straight from the cast kernel (S2.2-S2.8).
- **Release**: 206-1426 TFLOPS across V3 inference shapes, 1.1-2.7x over DeepSeek's tuned CUTLASS 3.6
  implementation, 1550 TFLOPS peak on H800 (S6, S1.4); the training-quality side: FP8 relative loss
  error below 0.25% against BF16 (S4.6).
- **Limits**: SM90 is NT-only with FP32 SFs; SM100 needs packed UE8M0 with gran_k 32 or 128; SF values
  must be exact powers of two; unpacked SFs cost a transform kernel per call; casting, transposition
  and SF production are the caller's job; FP4 forces BLOCK_K = 256 and a 1x32 recipe; BF16 usually
  lands in cuBLASLt unless determinism is on.

## Beat → slide map

| Beat | Slide |
|---|---|
| Prosperity | 2 — FP8 is the training format, E4M3 is narrow |
| Betrayal (pain number) | 3 — ~2% error from 14-bit accumulation |
| Failed suitors / villain | 4 — per-tensor scaling vs the outlier |
| Turn | 5 — power-of-two scales, 8 bits, streamed with the tile |
| Overview | 6 — the FP8 GEMM inner loop with SF operands |
| Mechanisms ❶❷❸ | 7, 8, 9 |
| Coverage | 10 — FP4 and BF16 on the same path |
| Release | 11, 12 |
| Humility | 13 — cost and limits |
| Close | 14 — summary + QR |
