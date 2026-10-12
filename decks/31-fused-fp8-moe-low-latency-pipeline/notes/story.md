# Story — deck 31: Fused FP8 MoE low-latency pipeline (HPC-Ops)

- **Where it lives** (prosperity): MoE layers in Tencent Hunyuan serving with HPC-Ops on H20 (SGLang
  `--moe-runner-backend hpc_ops`); Hunyuan-V3 shape 192 experts, top-8, hidden 4096, intermediate 1536; at TP8
  and batch 16 only 0.67 tokens per expert.
- **Pain** (betrayal): a gather-then-GEMM MoE copies every token **8×** (top-8) into a gathered buffer before its
  first expert GEMM, inside a 6-kernel chain (route, gather, Gate-Up, act + quant, Down, top-k reduce); at a few
  tokens per expert this plumbing rivals the expert GEMMs (LMSYS). No measured per-stage breakdown is published,
  so the pain slide carries the structural (analytic) number.
- **Failed suitors**:
  1. Gather-then-GEMM: global atomic per (token, expert) pair, an 8× gathered copy (top-8) before the first GEMM.
  2. TMA + warp-specialized persistent GEMMs: one deep pipeline per SM that small tiles cannot keep busy.
- **Villain**: the MoE plumbing was designed for large per-expert batches; at decode every stage is latency-bound.
- **Turn / insight**: routing already fixes every token's slot, and an 8-row tile leaves room for many CTAs →
  pass indices instead of copies, hide latency across CTAs, and butt the stages together with PDL.
- **Mechanisms**:
  ❶ Shared-memory counting + one global atomic per expert per block → row_indices / topk_pos / task maps; Gate-Up
     reads x rows in place through row_indices with cp.async (answers: atomics and the gather copy).
  ❷ cp.async GEMMs without warp specialization: one 128-thread warpgroup, 2 stages, ≥ 5 resident CTAs, grid of
     5–9 CTAs per SM (answers: idle deep pipelines at small tiles).
  ❸ PDL on every kernel of the five stages (6–7 kernels; answers: launch gaps).
- **Release**: vs the best of SGLang, vLLM Triton, vLLM CUTLASS on Hy3 shapes (H20): mean **1.08×** at TP8 (cp.async
  path, max 1.18×, never slower) and **1.21×** at EP8 (TMA path, max 1.60×); with HPC-Ops attention, Hy3-FP8 TPOT
  **−15.1% to −48.8%** at batch 4–64.
- **Limits**: fast path only for per-tensor FP8 with per-rank intermediate ≤ 512 (multiple of 64), ≤ 512 local
  experts, top-k ≤ 128; wider experts and block-wise use the TMA path with a gather; intermediates still in HBM
  (megakernel on the roadmap; cf. deck 35); not uniformly faster (0.98× EP8 batch 512; H200 Qwen3 0.89–4.21×;
  H200 Hy3 serving −4.2% to +6.3%).

## Slide plan (14 logical slides, 16 pages, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Decode spreads a few tokens over many experts | prosperity | fig_layer |
| 3 | A top-8 gather copies every token 8 times | pain | big number (8×) |
| 4 | Six kernels, a copy and five launch gaps | status quo / villain | fig_baseline |
| 5 | Insight: pass indices, not copies | turn | text + callout (build) |
| 6 | Five stages, chained with PDL | overview | fig_pipeline |
| 7 | ❶ Count in shared memory, reserve once | mechanism 1 | fig_count_1 → fig_count_2 |
| 8 | ❶ Gate-Up reads tokens in place | mechanism 1 | fig_rowidx |
| 9 | ❷ Many small CTAs beat one deep pipeline | mechanism 2 | fig_occupancy |
| 10 | ❸ PDL hides the gaps between stages | mechanism 3 | fig_pdl |
| 11 | 1.08× mean over the best baseline at TP8 | release (kernel) | chart speedup |
| 12 | Hy3 TPOT drops up to 48.8% end-to-end | release (E2E) | chart tpot |
| 13 | The price: narrow fast path, honest ceilings | limits | two-col |
| 14 | Summary | | QR |
