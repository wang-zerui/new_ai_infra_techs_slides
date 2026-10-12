# Story — deck 41: Mega Gate / Mega mHC fused kernels (DeepGEMM)

- **Where it lives** (prosperity): DeepSeek-V4 (Flash d = 4096, Pro d = 7168). Every sub-layer is wrapped
  in mHC with n_hc = 4, so each token carries a 4 × d residual stream and 24 mixing coefficients per layer
  (H_pre 4, H_post 4, H_res 4×4 via 20 Sinkhorn iterations). Every MoE layer starts with a router: Sqrt(Softplus)
  scores over 256 / 384 experts, aux-loss-free bias, top-6. DeepGEMM ships Mega mHC and Mega Gate (PR #432,
  2026-09) as single SM100 kernels for both.
- **Pain number** (betrayal): op by op, 4-stream hyper-connections move **≈ 11×** the memory traffic of a
  plain residual per token per layer (34d vs 3d elements; mHC paper Table 2 at n = 4, where its C = d; derived ratio).
- **Failed suitors**:
  1. Op-by-op HC (eager PyTorch): every coefficient, mapping and merge is a kernel that re-reads the n × d
     stream → the 34d above.
  2. Hand-fused TileLang kernels (mHC paper): merge fused into F_post,res ((3n+1)C → (n+1)C reads), 6.7%
     training overhead with recomputation and DualPipe changes — but still 4 kernels (normal) or 7 (shifted)
     at each layer boundary; the new residual is written, then re-read by the projection GEMM and by the
     pre-mix; split-K partials go through HBM to a second kernel. Router: cuBLASLt GEMM writes FP32 logits
     [T, E], a separate top-k kernel reads them back.
- **Villain**: the work between two layers is a chain of tiny, memory-bound steps (24 outputs, a 4×4
  Sinkhorn, one norm) that each touch the wide stream; as separate kernels, every step pays a launch, a full
  pass over 4d per token and a global barrier.
- **Turn / insight**: everything between two layers is row-local: one token's x and 4 × d residual in, its
  new residual and next-layer input out; only the 24-wide projection's K = 4d reduction needs other SMs
  (split-K). Same for the router: top-k needs only one token's E scores. So one kernel can stream each row
  through once.
- **Mechanisms**:
  ❶ One pass, three consumers (answers: the stream is re-read per step): Post workers compute every new-residual
     element once and hand it to the HBM store, to tensor memory as the MMA's A operand for the 24-wide
     projection, and to a running sum of squares for the RMS.
  ❷ Split-K that finishes in the same kernel (answers: tiny GEMM needs many SMs, reduction used to need a
     second kernel): partials + split barriers, fixed-order reduction, Mix warps (scale, bias, sigmoid,
     Sinkhorn), Norm warps (H_pre mix → RMSNorm → BF16 / FP8 + scales in Mega MoE's layout). Deterministic
     mode = 16 fixed splits; bitwise-stable across runs, batch sizes, graphs and streams.
  ❸ Mega Gate (answers: router logits round trip + 2-3 kernels): BF16 GEMM with experts on TMEM lanes and
     tokens on columns, sqrt(softplus) + bias, per-token top-k, normalised weights, replica mapping → int64
     indices + FP32 weights that Mega MoE takes directly.
- **Release**: Mega mHC 45–85% faster and Mega Gate 30–75% faster than their multi-kernel baselines
  (PR #432); analytic traffic per token at a layer boundary 34d (op by op) → 10d logical (16d including
  scratch) — while also emitting the normalised next-layer input. Context: with the earlier fused kernels
  mHC already cost only 6.7% in training (V4 §3.4.2).
- **Limits**: SM100 only; n_hc fixed at 4; Mega mHC needs d % 1024 = 0; forward only; Mega Gate: ≤ 512
  experts, top-k ≤ 32, sqrtsoftplus only; streams must be warmed up before CUDA-graph capture; the new
  residual is still an HBM output and normal mode re-reads it (L2 hint).

## Relation to sibling decks
- Deck 61 owns mHC as an architecture (why doubly stochastic, stability, quality). Deck 41 owns the kernels.
- Deck 35 (Mega MoE) consumes Mega Gate's outputs and Mega mHC's FP8 output layout; deck 25 is the parent
  library; deck 39 the determinism switch; deck 55 the aux-loss-free bias the gate adds.

## Slide plan (15 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | V4 widens every residual stream to four rows | prosperity | fig_hc (SVG) |
| 3 | Hyper-connections move about 11× the bytes | pain | `%% big: ≈ 11×` |
| 4 | Four kernels re-read the stream at every boundary | villain | fig_mhc_baseline (SVG) |
| 5 | Earlier fusion cut bytes but kept kernel boundaries | failed suitors | two-col text |
| 6 | Insight: everything between two layers is row-local | turn | boxed sentence |
| 7 | Both kernels sit around every MoE layer | overview | fig_chain (SVG) |
| 8 | ❶ One pass feeds store, MMA and RMS | mechanism | fig_post (SVG) |
| 9 | ❷ Split-K partials finish inside the kernel | mechanism | fig_splitk (SVG) |
| 10 | ❸ Mega Gate: GEMM to top-6 in one kernel | mechanism | fig_gate (SVG) |
| 11 | Mega mHC gains 45–85%, Mega Gate 30–75% | result | chart speedup (ranges) |
| 12 | One pass cuts boundary traffic from 34d to 10–16d | result (analytic) | chart traffic |
| 13 | Same token, same bits, any batch size | result | bullets |
| 14 | The cost: SM100, four streams, forward only | limits | two-col |
| 15 | Summary | | QR |
