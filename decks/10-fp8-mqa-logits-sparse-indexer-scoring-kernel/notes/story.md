# Story — deck 10: FP8 MQA-logits kernel (DeepGEMM's lightning-indexer scoring kernel)

- **Where it lives** (prosperity): DSA's lightning indexer (deck 02) runs in every layer of DeepSeek-V3.2 (and GLM-5,
  DeepSeek-V4); DeepGEMM shipped its scoring kernels on V3.2-Exp's release day (README 2025-09-28; PR #200 "indexer logit
  kernels for H800 (SM90)", merged 2025-09-29); vLLM and others call them (`fp8_mqa_logits`, now `fp8_fp4_mqa_logits`).
- **Pain number** (betrayal / stakes): the indexer is **81%** of prefill time at 200K tokens on a 30B DSA model (IndexCache
  §1) — the scoring kernel is the hot loop of long-context DSA (deck 04 removes passes; this deck makes each pass fast).
- **Failed suitors**:
  1. Stock GEMM + elementwise kernel: ReLU sits between the dot product and the head sum, so the GEMM must write
     per-head FP32 scores and a second kernel reads them back: **64×** the bytes of the logits (analytic, H = 64);
     128 GiB vs 2 GiB for one 4K-query chunk against 128K keys.
  2. Attention kernels (FlashAttention/FlashMLA dense): their epilogue is softmax + P·V and they never emit per-key
     scores, which is exactly what top-k needs.
- **Villain**: the operator looks like attention but ends like a reduction: K = 128 is short, the nonlinearity is per
  (query, head, key), and the output is one number per (query, key).
- **Turn / insight**: it is MQA: one 128-d FP8 key serves every head and every query token, so put (token × head) on the
  MMA's N side (a 128-row Q tile: 2 tokens × 64 heads), keys on M, and finish ReLU · w · Σheads in registers; only the
  [query × key] logits ever reach HBM.
- **Mechanisms**:
  ❶ Tile mapping + warp-specialized pipeline: Q tile of 128 (token, head) rows loaded once; 256-key KV tiles (SM90:
     4 math warpgroups × 64 keys; SM100: 3 × 128 = 384) streamed by a TMA warp through a 3-stage ring; one persistent
     CTA per SM (answers: K = 128 gives little work per byte; keep tensor cores fed).
  ❷ Fused epilogue: ReLU → × head weight → Σ heads (in-thread adds + 2 shuffles) → × per-key scale once → write only
     columns in [ks, ke), rows compressed; SM100 fuses ReLU into `cvt.rn.relu.bf16x2.f32` and weights 2 heads per
     `hfma2` (answers: the per-element epilogue is as hot as the MMA).
  ❸ Paged decode with split-KV and no combine: a metadata kernel cuts each request into 256-key splits (4 pages of 64)
     and gives every SM an equal share; logits per key are independent, so splits need no reduction; next_n ≤ 2
     tokens share one key read (answers: decode is HBM-bound, 124 FLOP/B vs a 412 FLOP/B ridge on H200).
- **Release**: community measurements on H200 (PR #452): **941–1,042 TFLOPS** FP8 on production indexer shapes, about half
  of H200's 1,979 TFLOPS dense FP8 peak (derived); swapping only the ReLU instruction adds 4.8–8.1% → the epilogue is on
  the critical path. Evolution: FP4 indexer (2026-04), Sparse Indexer for V4.1 (2026-09), SM100 MXFP8/MXFP4 with BF16
  logits.
- **Limits**: logits are still materialized [n_q × span] before a separate top-k (vLLM blog); SM90/SM100 only (SM120
  community ports), SM90 heads 32/64; callers must mask past each row's span; since 2026-09 the default is
  non-deterministic (opt back in via `use_deterministic_algorithms`), which matters for RL (GLM-5); no official perf
  table from DeepSeek.

## Slide plan (14 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | The indexer is one MQA-shaped kernel call | prosperity | fig_contract (SVG) |
| 3 | At 200K, scoring is 81% of prefill time | pain | `%% big: 81%` |
| 4 | GEMM + epilogue kernels move 64× the bytes | failed suitors / villain | fig_unfused (SVG) |
| 5 | Insight: one key serves every head and token | turn | fig_fanout (SVG) + boxed sentence (build) |
| 6 | One persistent CTA per SM streams keys | overview | fig_kernel (SVG) |
| 7 | ❶ 128 (token, head) rows meet 256 keys | mechanism | fig_tile (SVG) |
| 8 | ❷ ReLU, weights and head sum stay in registers | mechanism | fig_epilogue (SVG) |
| 9 | ❸ Decode splits keys evenly, with no combine | mechanism | fig_decode (SVG) |
| 10 | ~1 PFLOPS FP8 on H200 at production shapes | release | chart_tflops (bars) |
| 11 | The kernel family grew FP4 and sparse variants | evolution | fig_timeline (SVG) |
| 12 | The cost: dense logits, masks, determinism | limits | two-col text |
| 13 | Summary | | QR |
