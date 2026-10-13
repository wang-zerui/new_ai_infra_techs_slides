# Story — deck 72: MagiAttention

- **Where it lives** (prosperity): Sand AI's MAGI-1, a 24B autoregressive video world model trained with
  contexts of up to 4M tokens; chunk-wise (24-frame) denoising gives a block-causal mask, and online
  Patch-and-Pack (PnP) packs videos of different lengths into one sequence, so the CP attention sees a
  varlen block-causal mask. MagiAttention is the open-source (Apache-2.0) CP attention library built for it,
  with Megatron-LM, PyTorch FSDP and HF Transformers integrations.
- **Pain number** (betrayal): the classic ring P2P CP attention delivers ≈513 TFLOPs/s per GPU on a full mask
  but only ≈29 on a packed (varlen) causal mask, at the same scale (64 H100, forward): ≈18× lower
  (MagiAttention CP benchmark, pixel-read; ratio derived). Same unit as the results (TFLOPs/s/GPU).
- **Failed suitors**:
  1. Ring P2P with zigzag sharding: the balance trick is tailored to one causal mask; per-sample zigzag on a
     packed mask fragments and pads; every KV chunk circulates to every rank (25% redundant for causal,
     33% for the varlen block-causal mask in the blog's 4-rank example).
  2. Ulysses (head sharding): CP size limited by head count (divisibility).
  3. Hybrids (USP, LoongTrain) combine both and inherit the ring traffic; Dynamic CP / Hybrid-CP (deck 68)
     resizes CP groups per sample but needs extra NCCL buffers and cross-group scheduling.
- **Villain**: CP layers assume a regular mask: they shard the sequence by position and move KV by position
  (ring order), not by which queries actually need it.
- **Turn / insight**: write the mask as a list of AttnSlices (Q range, K range, mask type); the slice list is
  a complete, distributable description of the work, so each rank's compute, its balance, and exactly which
  remote KV it needs can all be computed from it before the step runs.
- **Mechanisms** (container entry: four components):
  ❶ Flex-Flash-Attention (FFA): FA3-based kernel that runs any slice list (4 mask types, slice-level
     parallelism, atomic reduction); FFA_FA4 backend on Blackwell. Answers: no kernel runs Magi-1's mask.
  ❷ Dispatch solver: chunk-wise permutable sharding, greedy min-heap balancing of mask area with equal chunk
     counts per rank. Answers: imbalance under irregular masks.
  ❸ GroupCast / GroupReduce: send each KV chunk only to the ranks whose slices need it and reduce dKV back;
     prototype on all-to-all-v, now native DeepEP-based kernels with RDMA de-duplication.
     Answers: redundant ring traffic.
  ❹ Adaptive multi-stage overlap: remote KV split into overlap_degree stages; prefetch stage i+1 while FFA
     computes stage i (backward also reduces dKV of stage i-1). Answers: exposed communication.
- **Release**: 64 H100, varlen causal mask: 238 fwd / 265 bwd TFLOPs/s/GPU, ≈8× ring P2P and 1.24× / 1.06×
  the best baseline; on 64 B200, 1.9× / 1.4× the best baseline. Native group collectives add 9-15% (H100)
  and 23-41% (B200) over all-to-all-v at 64 GPUs. Kernel: FFA reaches 90-94% of FA3 forward on FA3's masks
  and 3.6× FlashMask on the varlen block-causal mask FA3 cannot run (H100, 32K).
- **Limits**: per-GPU throughput still falls with scale (H100 fwd 459→238 from 8 to 64 GPUs, varlen causal);
  no edge on a full mask (backward ≈5% behind USP / Ulysses); FFA ≈ 90% of FA3 on regular masks; vendor
  benchmarks with experimental features enabled; native internode kernels need IBGDA and 8 GPUs per node;
  overlap_degree is tuned by hand; static solver needs the mask at data-loading time and one mask for all
  layers (dynamic solver / QO-comm experimental); FFA_FA4 is a temporary backend; dispatch permutes tokens.

## Slide plan (16 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | MAGI-1 trains video at up to 4M tokens | prosperity | fig_hook (SVG) |
| 3 | Packed masks cut ring attention ≈18× | pain | `%% big: ≈18×` |
| 4 | Ring CP ships KV by position, not need | failed suitors + villain | fig_statusquo (SVG) |
| 5 | Insight: plan CP from the mask itself | turn | fig_slices (SVG) + boxed sentence |
| 6 | MagiAttention compiles the mask into a plan | overview | fig_overview (SVG) |
| 7 | ❶ FFA computes any slice list in one kernel | mechanism | fig_ffa (SVG) |
| 8 | ❷ Dispatch balances mask area, not tokens | mechanism | fig_dispatch (SVG, analytic) |
| 9 | ❸ GroupCast sends KV only where needed | mechanism | fig_groupcast (SVG) |
| 10 | ❸ Native kernels turn RDMA copies into NVLink | mechanism | fig_native (SVG) |
| 11 | ❹ Stages hide KV traffic under FFA | mechanism | fig_overlap (SVG) |
| 12 | FFA: ≈90% of FA3, 3.6× FlashMask | result 1 | chart kernel_tflops |
| 13 | 64 H100s: 8× ring P2P, 1.06-1.24× best baseline | result 2 | chart dist_scaling |
| 14 | Native kernels add up to 41% at 64 GPUs | result 3 | chart native_gain |
| 15 | The cost: hand tuning, IBGDA, no full-mask edge | limits | two-col |
| 16 | Summary | | QR |
