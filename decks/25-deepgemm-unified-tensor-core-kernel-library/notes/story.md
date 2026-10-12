# Story — deck 25, DeepGEMM (container entry)

Container deck: library overview, one slide per component family, each pointing at the sibling deck.

- **Where it lives**: DeepSeek-class MoE LLMs — V4-Pro 1.6T params (49B activated), V4-Flash 284B
  (13B activated), 1M context, 256–384 experts with top-6 routing — trained and served on H800
  (SM90) and SM100 GPUs. Every step is a handful of tensor-core primitives: dense projections,
  per-expert MoE GEMMs, indexer scoring, hyper-connection mixing. (S4.1, S3.3)
- **Pain number**: at the decode-side shapes of a V3-class model (M=64, N=2112, K=7168) a carefully
  tuned CUTLASS 3.6 FP8 implementation left **2.7x** on the table against DeepGEMM on the same H800.
  (S2.3, S2.4)
- **Failed suitors**:
  1. Vendor/library GEMM (cuBLASLt, CUTLASS templates) — no per-group scaling along K, which is what
     fine-grained FP8 needs (S5.1); and no batch invariance, so DeepSeek replaced cuBLAS end-to-end
     (S4.3).
  2. One hand-written kernel per operator — the model changes (FP4 weights, mHC, sparse indexer) and
     each new operator restarts the tiling, pipelining and scheduling work.
- **Villain**: the mismatch between what a BLAS interface assumes (one dtype, one scale, static
  shapes, N and K fixed per call) and what an MoE LLM step actually is (mixed FP8/FP4/BF16 operands,
  scales per 1x128 tile, per-expert token counts known only on the GPU, communication in the middle
  of the GEMM).
- **Turn / insight**: these primitives differ in scheduling, not in their inner loop. Write one
  tile-level kernel per family — a few hundred lines each, 9,593 lines of device code for all 19
  (derived, S6.2-S6.3) — give them a shared scheduler, layout and epilogue, and let a runtime JIT
  burn every shape, block size and switch into the kernel as a compile-time constant.
- **Mechanisms**:
  1. ❶ **One kernel per family, specialised per shape by DeepJIT** — the GEMM family covers
     FP8/FP4/BF16 and all four majors on SM100; the JIT emits a template instantiation with ~25
     compile-time constants (S6.4, S2.7) → deep dive: deck 30 (dense + UE8M0) and deck 39 (JIT/PDL).
  2. ❷ **Grouped layouts instead of per-expert calls** — M-axis contiguous for prefill/training,
     masked for CUDA-graph decode, K-axis for the MoE weight backward (S1.7) → decks 33, 34, 32.
  3. ❸ **Mega kernels that swallow the surrounding ops** — Mega MoE fuses dispatch, both linears,
     SwiGLU and combine and overlaps NVLink traffic with tensor-core work; Mega Gate and Mega mHC do
     the same for routing and hyper-connections; MQA-logit kernels serve the lightning/sparse indexer
     (S1.8, S3.4) → decks 35, 41, 10.
- **Release**: 1,550 TFLOPS peak FP8 on H800 (S1.3); 1,426 TFLOPS at M=4096 and 336 TFLOPS at M=64
  for the same N=7168, K=16384 shape, 1.3–1.4x over the tuned CUTLASS implementation (S2.4);
  Mega MoE 1.50–1.96x over the non-fused pipeline on V4-Flash/V4-Pro at EP8 (S3.3, S4.2);
  Mega Gate 30–75%, Mega mHC 45–85%, MoE backward ~20% (S3.4).
- **Limits**: SM90/SM100 only (plus a separate Ascend port), CUDA ≥ 12.9, CUTLASS ≥ 4.0; casting,
  transposition and scale-factor production are the caller's job; Mega MoE needs symmetric memory and
  PyTorch ≥ 2.9; kernels compile at first use, so serving stacks ship a warm-up pass; determinism is
  off by default since the 2026-09 release.

## Beat → slide map

| Beat | Slide |
|---|---|
| Prosperity | 2 — an LLM step is four tensor-core primitives |
| Betrayal (pain number) | 3 — 2.7x left on the table |
| Villain / failed suitors | 4 — what a BLAS call cannot express |
| Turn | 5 — insight slide |
| Overview | 6 — layered library figure |
| Mechanisms ❶❷❸ | 7, 8, 9, 10 |
| Release | 11, 12, 13 |
| Humility | 14 — cost and limits |
| Close | 15 — summary + QR |
