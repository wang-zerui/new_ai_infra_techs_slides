# Story — deck 36: Route GEMM (BF16 x FP32 via dual-BF16 split)

- **Where it lives** (prosperity): MoE serving on Hopper (H20 first, H200 validated). The router projects
  BF16 hidden states onto FP32 router weights; the top-k of these scores picks the experts. Production
  example: LongCat-Flash (router K=6144, N=768 = 512 routed + 256 zero experts) and LongCat-Flash-Lite
  (K=3072, N=384) in SGLang; HPC-Ops also targets the state-compress GEMMs of sparse/linear attention.
- **Pain number** (betrayal): exact FP32 weights cost **1.96×** the time of TF32 for the same router GEMM
  (FP32 cuBLAS 403.7 µs vs TF32 cuBLAS 205.8 µs, M=K=4096, N=192, H20) — and TF32 is the "fast" option
  that already rounds every weight to 11 significant bits (max abs error 0.0646 vs FP32).
- **Failed suitors**:
  1. FP32 cuBLAS: exact, but no FP32 Tensor-Core mode on Hopper → CUDA-core (SIMT) speed, plus an
     `x.float()` cast of the BF16 activations.
  2. TF32 / BF16 Tensor Cores: fast, but drop 13 / 16 of the weight's 24 significand bits; small score
     changes can flip a near-tied top-k decision.
- **Villain**: Tensor Cores only multiply short-mantissa formats, while the router weight carries 24 bits
  of significand that the top-k decision is sensitive to (format mismatch, not a slow library).
- **Turn / insight**: an FP32 weight is (to 16 bits) the sum of two BF16 numbers, w ≈ w_hi + w_lo/256;
  because the GEMM is linear, x·w = x·w_hi + (x·w_lo)/256 — two BF16 Tensor-Core GEMMs whose only
  error is the dropped tail bits 17–24 (≤ 2^-16·|w|), because BF16 x is exact, BF16×BF16 products are
  exact in FP32, and ×1/256 is a power of two.
- **Mechanisms**:
  ❶ Offline hi/lo split (answers: Tensor Cores cannot take FP32 weights). Done once at model load;
     same 4 bytes per weight as FP32.
  ❷ One fused kernel, two FP32 accumulators (answers: two GEMMs would read x twice and write/re-read two
     partial outputs). TMA loads each x tile once into shared memory, both WGMMAs consume it, the
     epilogue computes acc_lo·(1/256) + acc_hi and stores once.
  ❸ Small-M tiling: swap-AB (weight's N on the 64-row WGMMA side, tokens on the 16-wide side) +
     split-K up to 8 with in-kernel reduction gated by per-tile counters (answers: decode M is tiny, too
     few tiles to fill the GPU).
- **Release**: 1.30–3.22× faster than FP32 cuBLAS and 1.25–1.78× faster than TF32 cuBLAS on H20, with max
  abs error 0.00177 vs 0.06464 for TF32 (≈36× lower); LongCat-Flash-Lite-FP8 input throughput +5.5–6.1%
  (bs 4–64, 8×H20) from the router GEMM alone; GSM8K unchanged (0.798 → 0.802, H200).
- **Limits**: below a measured M crossover the FP32 path is faster (SGLang min-M guards 64 / 128 on
  H200 → Lite decode keeps cuBLAS); split cached at load → online weight updates rejected while active
  (matters for RL weight sync); SM90 only, N % 64 == 0; inference forward only; precision is ~16 bits,
  not the full 24 ("close to FP32", not bit-identical).

## Slide plan (16 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Router scores pick each token's experts | prosperity | fig_router (SVG) |
| 3 | Exact FP32 weights cost 1.96× TF32's time | pain | `%% big: 1.96×` |
| 4 | Tensor-Core formats drop the weight's low bits | villain / failed suitors | fig_bits (SVG) |
| 5 | Insight: one FP32 weight is two BF16 weights | turn | boxed sentence (build) |
| 6 | Split once, multiply twice, store once | overview | fig_overview (SVG) |
| 7 | ❶ The split keeps bits 1–16 of the weight | mechanism | fig_split (SVG, worked example) |
| 8 | Only the dropped tail bits cost accuracy | why-it-works | chart bits_kept (analytic) |
| 9 | ❷ One kernel feeds both GEMMs from one tile | mechanism | fig_fused_1/2 (SVG build) |
| 10 | ❸ Swap-AB and split-K fill the GPU at small M | mechanism | fig_smallm (SVG) |
| 11 | Beats FP32 cuBLAS by 1.30–3.22× on H20 | result | chart latency (line, log-log) |
| 12 | Error stays 36× below TF32 at higher speed | result | chart error (bars, log) |
| 13 | LongCat prefill gains 5.5–6.1% input throughput | result | chart e2e (bars) |
| 14 | The cost: crossovers, frozen weights, Hopper only | limits | two-col text |
| 15 | Summary | | QR |
