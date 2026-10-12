# Story — deck 14: wave-based fine-grained communication/compute overlap

- **Where it lives** (prosperity): DeepSeek-V4 MoE layers (V4-Pro 1.6T/49B, 384 experts top-6; V4-Flash
  284B/13B, 256 experts top-6) under expert parallelism, in training, inference and RL rollout. Each layer:
  Dispatch → Linear-1 → SwiGLU + FP8 cast → Linear-2 → Combine. Running example: V4-Flash, EP8, 32 experts
  per rank.
- **Pain number** (betrayal): with cross-node EP, DeepSeek-V3 measured a computation-to-communication ratio of
  **≈ 1:1**: run serially, a MoE layer spends as long moving tokens as computing on them.
- **Failed suitors**:
  1. Batch-level overlap (V3 DualPipe in training, two micro-batches in prefill/decode): needs a second,
     equally sized independent batch; DualPipe keeps 2× parameters; nothing to pair with in a single small
     long-tail batch (RL rollout). Deep dive: deck 11.
  2. Per-stage overlap (Comet): Dispatch ∥ Linear-1, Linear-2 ∥ Combine. Theoretical 1.42× on V4-Flash. The
     pairs are mismatched: the small transfer (FP8 dispatch, h bytes) pairs with the big GEMM (Linear-1,
     4hd) and the big transfer (BF16 combine, 2h bytes) with the small GEMM (Linear-2, 2hd) — derived.
- **Villain**: the layer-wide stage barrier. Dispatch and combine are treated as collectives over all
  experts, so every expert waits for every other expert's tokens before any GEMM runs, and the network waits
  for every GEMM before any result returns.
- **Turn / insight**: profiling shows that within one V4 MoE layer total communication time is *less* than
  total computation; so if communication and computation are pipelined at expert granularity, compute is the
  only bottleneck left and the interconnect can be slower without hurting end-to-end speed.
- **Mechanisms**:
  ❶ Expert waves: split each rank's experts into waves; a wave's GEMMs start as soon as its own tokens land
     (answers: the layer-wide barrier).
  ❷ One mega-kernel with warp roles and pull-based dispatch: dispatch warps pull tokens from peers over
     NVLink, MMA warps run L1 then L2 tiles wave by wave, epilogue warps do SwiGLU+FP8 cast and write L2 output
     straight into the home rank's combine buffer (answers: separate kernels cannot hand off per expert; fine-
     grained push pays notification latency).
  ❸ Wave sizing: the smallest wave whose GEMM tiles fill every SM (×2 slack for routing imbalance, ≤ 32
     experts) (answers: big waves reintroduce the fill barrier, tiny waves starve SMs — matters most for
     small batches).
- **Release**: theoretical 1.92× vs 1.42× for per-stage overlap (V4-Flash); measured 1.50–1.73× on general
  inference and up to **1.96×** for latency-sensitive cases (V4-Flash, 1 token/rank, EP8) over the non-fused
  path. Same currency as the pain: the ≈ 1:1 ratio means serial costs up to ~2× the compute time.
- **Hardware corollary**: full hiding needs only C/B ≤ V_comp/V_comm = 2d = 6144 FLOPs/Byte for V4-Pro: 1 GB/s
  hides 6.1 TFLOP/s; extra bandwidth beyond that buys nothing.
- **Limits**: fill/drain stays exposed (first dispatch, last combine); needs symmetric memory, multi-process
  launch, SM100 kernel, FP8×FP4 at release ("still under development"); power throttling when compute,
  memory and network all run hot; SwiGLU exp/div stalls the GEMM pipeline; push-based dispatch would need
  lower-latency cross-GPU signalling.

## Slide plan (15 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Each MoE layer wraps GEMMs in two all-to-alls | prosperity | fig_moe_layer (SVG) |
| 3 | Communication costs as much as compute | pain | `%% big: ≈ 1 : 1` |
| 4 | Serial stages leave tensor cores idle | villain | fig_overlap_1 (panel a) |
| 5 | Per-stage overlap pairs mismatched stages | failed suitors | fig_overlap_2 (a+b) + batch-level bullet |
| 6 | Insight: communication is shorter than compute | turn | boxed sentence |
| 7 | Pipeline experts in waves inside one kernel | overview | fig_overlap_3 (a+b+c) |
| 8 | ❶ A wave computes as soon as its tokens land | mechanism | fig_waves (snapshot) |
| 9 | ❷ One kernel with three warp roles | mechanism | fig_kernel |
| 10 | ❸ Waves just big enough to fill every SM | mechanism | fig_wavesize |
| 11 | Waves raise the overlap ceiling to 1.92× | result (theory) | chart theory |
| 12 | Mega MoE runs 1.50–1.96× faster than non-fused | result (measured) | chart speedup |
| 13 | Bandwidth beyond C/B = 2d buys nothing | corollary | chart balance (analytic) |
| 14 | The cost: exposed edges, power, prerequisites | limits | two-col text |
| 15 | Summary | | QR |
