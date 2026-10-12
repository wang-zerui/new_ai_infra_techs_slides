# Story: HPC-Ops (container deck)

- **Where it lives** (prosperity): Tencent Hunyuan AI Infra's operator library behind large-scale production
  serving of Hunyuan (Hy3) and DeepSeek-class MoE models on NVIDIA H20 (SM90); open source since Jan 2026,
  integrated into SGLang main in Aug 2026.
- **Pain number** (betrayal): on a skewed decode batch (1×128K + 31×4K, FP8 KV, H20) a static split-KV
  attention schedule takes 0.186 ms vs 0.063 ms, i.e. **2.95×** slower, while on a uniform batch the two are
  identical (1.00×). [L Table 1]
- **Escalation**: the same pattern repeats along the decode step: small uneven expert GEMMs whose glue (gather,
  launches, HBM round-trips) rivals the math; FP32-sensitive router weights that are either slow (FP32) or lossy
  (BF16/TF32); a sampler that is a chain of small vocabulary passes; AllReduce, residual add and RMSNorm as
  three stages.
- **Failed suitors**: (1) generic kernel libraries (FlashInfer/FA3, cuBLAS, NCCL, Triton MoE) tuned for
  uniform or large shapes and per-op boundaries; (2) the library's own static split-KV schedule, which is
  perfect on uniform batches and collapses under skew.
- **Villain**: kernels are designed per operator for benchmark shapes; production decode shapes on H20 are
  skewed, small and precision-sensitive, so the cost moves into imbalance, glue and precision fallbacks.
- **Turn / insight**: the same attention math runs at 1.00× on a uniform batch and 2.95× slower on a skewed
  one — the loss is in scheduling and glue, not in the math; so tune each hot path for the production shape
  distribution.
- **Mechanisms** (library design principles, each owned by sibling decks):
  ❶ schedule by live work (dynamic decode attention, deck 07; block-sparse FP8 prefill, deck 09) ↔ skewed
  and sparse KV work;
  ❷ fuse the glue around small ops (fused FP8 MoE, deck 31 + FP8 GroupGEMM, deck 29; AllReduce+RMSNorm,
  deck 13; fused sampler, deck 40) ↔ launch and HBM round-trip overhead;
  ❸ keep precision where it decides the output (Route GEMM dual-BF16, deck 36) ↔ FP32 router weights.
- **Release**: Hy3-FP8 on 8×H20, HPC-Ops Attention + MoE in SGLang: TPOT −15.1% to −48.8% at batch 4–64
  [L Table 7]; Tencent reports ~30% QPM for Hunyuan, ~17% for DeepSeek models in production [MTP].
- **Limits**: SM90/H20-tuned (other archs: only partial ports); head_dim 128 attention; fixed hidden sizes for
  the fused AllReduce; README numbers are per-operator maxima; some shapes are at parity or slower (MoE 0.98×
  at EP8 batch 512; H200 Hy3 MoE −4.2% to +6.3%); router GEMM needs crossover dispatch.

## Slide plan (15 logical slides, 8 minutes)

| # | Headline | Device |
|---|---|---|
| 1 | Title | front matter |
| 2 | Tencent serves its Hunyuan models on HPC-Ops | bullets |
| 3 | Skewed batches slow decode attention 2.95× | big 2.95× |
| 4 | Generic kernels misfit five decode hot spots | fig_before (red problems) |
| 5 | Insight: the loss is imbalance and glue, not math | boxed sentence |
| 6 | HPC-Ops: one tuned operator per hot path | fig_stack (layered, deck numbers) |
| 7 | ❶ Attention schedules by live KV work | fig_attn (two panels) — decks 07, 09 |
| 8 | ❷ Fused MoE removes the gather and launch gaps | fig_moe before/after — decks 31, 29 |
| 9 | ❷ Fusion shrinks AllReduce and sampler chains | fig_fuse two panels — decks 13, 40 |
| 10 | ❸ Route GEMM gets FP32 precision from BF16 | fig_route — deck 36 |
| 11 | Every hot path gains 1.6–8.5× at its best | chart_ops |
| 12 | Hy3-FP8 per-token latency drops up to 48.8% | chart_tpot |
| 13 | Tencent production gains ~30% QPM | bullets (QPM, SGLang upstream, H200) |
| 14 | Gains are H20-tuned and shape-specific | bullets |
| 15 | Summary | bullets + QR |
