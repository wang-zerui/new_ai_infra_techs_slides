# Story — deck 35: Mega MoE fused MegaKernel (DeepGEMM)

- **Where it lives** (prosperity): DeepSeek-V4 MoE layers under expert parallelism inside one NVLink domain
  (DeepGEMM PR #316 benchmarks EP8 on SM100). V4-Flash: 256 routed experts + 1 shared, top-6, d = 4096,
  I = 2048. V4-Pro: 384 routed + 1 shared, top-6, d = 7168, I = 3072. Each layer: dispatch → linear-1 →
  SwiGLU (+ FP8 cast) → linear-2 → combine. Used for inference, RL rollout and agent serving (V4 §3.1).
- **Pain number** (betrayal): at 1 token per rank, a V4-Flash MoE layer run as five kernels (DeepEP dispatch,
  grouped GEMM, SwiGLU kernel, grouped GEMM, DeepEP combine) takes **≈ 111 µs** (derived: 56.5 µs fused ×
  1.96 speedup vs legacy, PR #316).
- **Failed suitors**:
  1. Batch-level overlap (two streams / two micro-batches, DualPipe-style; deck 11): needs a second
     independent batch; a latency-bound decode or RL-rollout step has one small batch.
  2. Per-stage overlap (Comet): dispatch ∥ linear-1, linear-2 ∥ combine: theoretical ceiling 1.42× vs 1.92×
     for expert-wave pipelining on V4-Flash (V4 Fig. 5; deck 14).
- **Villain**: the kernel boundary. Each of the five kernels is a layer-wide barrier (no expert's GEMM starts
  until every token of every expert has landed; no result returns until every GEMM is done) and every
  hand-off is a full tensor round trip through HBM (BF16 [n, 2I] linear-1 output written then re-read by
  the SwiGLU kernel; BF16 [n, d] linear-2 output written then re-read by combine).
- **Turn / insight**: over NVLink symmetric memory, a peer's buffer is the local address plus a fixed
  offset, so dispatch and combine are just loads and stores that a GEMM kernel's own warps can issue; and
  V4 profiling shows a MoE layer communicates for less time than it computes. So one persistent kernel can
  run all five stages as a tile-level dataflow, and compute is the only bottleneck left.
- **Mechanisms**:
  ❶ Symmetric buffer + pull dispatch (answers: getting tokens into expert-major GEMM tiles without a
     separate kernel and without fine-grained push notification latency).
  ❷ Tile-level dependencies through counters (answers: kernel boundaries as layer-wide barriers): a
     persistent scheduler hands out linear-1 tiles first, then interleaves linear-2 tiles; an L1 tile waits
     only for its M-block's tokens, an L2 K-block only for the L1 N-blocks it reads; ring buffers bound
     memory.
  ❸ Epilogues absorb the glue (answers: HBM round trips of intermediates): L1 epilogue does SwiGLU × routing
     weight → per-32 UE8M0 FP8 cast straight from TMEM; L2 epilogue stores BF16 rows into the source rank's
     combine slot over NVLink; a final local FP32 reduction in fixed order writes y.
- **Release**: 56.5 µs instead of ≈ 111 µs at 1 token/rank; **1.50–1.96×** faster than the five-kernel path
  across 1–32768 tokens/rank on V4-Flash and V4-Pro (PR #316); at 32768 tokens/rank it sustains 2038 / 2438
  TFLOPS while moving 529 / 417 GB/s over NVLink. Output is bitwise equal to the non-fused path (test).
  Later releases fold in the shared expert ("11 kernels into 1", +20%), V4.1 tuning (+15% low latency),
  locality-aware scheduling (−100 W).
- **Limits**: SM100 only; multi-process launch with PyTorch ≥ 2.9 symmetric memory; one NVLink domain
  (≤ 72 ranks in code); SwiGLU only; takes every SM; symmetric buffer sized for the worst case; inputs
  must be copied into the buffer; full concurrency of compute, HBM and NVLink makes power a limiter;
  first pull and last combine stay exposed (grid syncs + NVLink barriers).

## Relation to sibling decks
- Deck 14 (wave-based fine-grained overlap) owns the *scheduling idea* (expert waves, theoretical 1.42× vs
  1.92×, the C/B ≤ 2d balance point). Deck 35 owns the *kernel*: what is fused, how symmetric memory and
  NVLink live inside it, how stages synchronize without kernel boundaries, what memory traffic disappears.
- Deck 25 (DeepGEMM) is the parent library; deck 18 (FP8 dispatch + BF16 combine) explains the dispatch/
  combine precisions; deck 39 the determinism switch; deck 31 (HPC-Ops fused FP8 MoE) is a contrasting
  single-GPU fusion.

## Slide plan (16 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Each V4 MoE layer crosses GPUs twice | prosperity | fig_layer (SVG) |
| 3 | One token per rank still costs about 111 µs | pain | `%% big: ≈ 111 µs` |
| 4 | Every kernel boundary is a layer-wide barrier | villain | fig_baseline (SVG timeline) |
| 5 | Earlier overlap fixes fall short inside one batch | failed suitors | two-col text |
| 6 | Insight: NVLink all-to-all is just loads and stores | turn | boxed sentence |
| 7 | One persistent kernel runs all five stages | overview | fig_fused (SVG timeline, same canvas as fig_baseline) |
| 8 | ❶ Dispatch warps pull tokens into expert tiles | mechanism | fig_dispatch (SVG, 2 builds) |
| 9 | ❷ Counters, not kernel ends, order the tiles | mechanism | fig_deps (SVG) |
| 10 | ❸ Epilogues fuse SwiGLU, FP8 cast, combine | mechanism | fig_epilogue (SVG) |
| 11 | One kernel runs 1.50–1.96× faster than five | result | chart speedup |
| 12 | Big batches run 2+ PFLOPS while NVLink streams | result | chart rates (TFLOPS + NVLink GB/s) |
| 13 | Fused output matches five kernels bit for bit | result | fig_bitwise (SVG) |
| 14 | Later releases fused more and ran faster | result | fig_releases (SVG timeline) |
| 15 | The cost: whole GPU, one NVLink domain, power | limits | two-col text |
| 16 | Summary | | QR |
