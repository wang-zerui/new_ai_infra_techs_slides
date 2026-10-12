# Story — deck 70: Megatron-FSDP / HSDP (with EP support)

- **Where it lives** (prosperity): Megatron-Core's own FSDP (Megatron-FSDP, also on PyPI), used for dense and
  MoE models; HSDP for MoE since MCore v0.13 (`--num-distributed-optimizer-instances`), FSDP + EP since
  v0.15. Giant fine-grained MoEs: DeepSeek-V3 has 685B parameters but 37B active per token (18× gap), and
  every expert's weights, gradients and Adam state stay resident.
- **Pain number** (betrayal): DeepSeek-V3 in BF16 on 256 GPUs (PP4 × VPP4 × EP64) carries **68.5 GB** of
  weights, gradients and optimizer state per GPU (36.4 + 32.1 GB), next to 131 GB of activations, on an
  80 GB H100 (MC Table 3).
- **Failed suitors**:
  1. Distributed optimizer (ZeRO-1): shards only the 12 bytes of FP32 master weights + Adam moments; BF16
     weights and FP32 grads stay replicated → 6 + 12/d bytes per parameter, a 6-byte floor no DP size removes.
  2. More model parallelism (PP, big EP, TP): shards weights but brings PP/VPP stage balancing, MTP/output
     layer placement, vision-encoder partitioning, and EP all-to-all across nodes.
  3. Generic FSDP on an EP model: shards experts over the full DP group (double sharding, wrong group).
- **Villain**: model state is replicated across data-parallel ranks even though each rank only touches one
  layer at a time; and MoE has two replica groups (dense over DP, experts over EDP) that one sharding scope
  cannot serve.
- **Turn / insight**: a DP replica needs a layer's full weights only while computing that layer, so every
  rank can keep 1/d of everything and all-gather the next layer while the current one computes. Then DP
  size, not PP/TP, becomes the memory knob.
- **Mechanisms**:
  ❶ FSDP units gathered just in time: all-gather before forward/backward, reduce-scatter after wgrad,
     prefetch on separate streams, two persistent buffers (answers: memory without stalls).
  ❷ Dual DeviceMesh: dense modules shard over DP (× CP), experts over their EDP group (answers: MoE's two
     replica groups).
  ❸ HSDP / HFSDP: shard inside the NVLink domain, replicate (or shard only optimizer state) across nodes
     (answers: per-layer collectives over slow inter-node links).
  ❹ Zero-copy collectives: per-module flat non-uniform shards + NCCL user-buffer registration (answers:
     buffer copies and SMs stolen from GEMMs).
- **Release**: analytic: bytes/param per GPU 18 (DDP) → 6 + 12/d (dist-opt) → 18/d (FSDP); measured:
  Qwen3-30B-A3B, EP8 → Megatron-FSDP + EP8: max reserved 45.6 → 23.5 GB (−48%) (user report, issue #2772);
  ~10% less comm overhead from non-uniform sharding (Llama3 405B); comm kernels 8–32 → 1–4 SMs on NVLink.
  DeepSeek-V3 reference script runs TP1 / EP8 / no PP with FSDP ZeRO-3.
- **Limits**: ZeRO-3 moves 1.5× the bytes of plain DP (3Ψ vs 2Ψ); same user report: 137.8 → 108.0 TFLOPS
  (−22%) when memory was not the bottleneck; no activation savings; double buffers add memory and need
  symmetric layers; fsdp_dtensor checkpoint format; `CUDA_DEVICE_MAX_CONNECTIONS` unset may slow TP/CP;
  performance claims in Megatron-Bridge are "doc_only".

## Slide plan (15 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Giant MoEs keep every expert's state resident | prosperity | fig_bytes (SVG) |
| 3 | Model state alone nearly fills an H100 | pain | `%% big: 68.5 GB` |
| 4 | The distributed optimizer stops at 6 bytes | failed suitors | fig_ladder_0 (SVG) |
| 5 | Insight: a rank needs each layer only briefly | turn | boxed sentence (build) |
| 6 | One flag picks how much state to shard | overview | fig_ladder_2 (SVG) |
| 7 | ❶ Gather each layer just in time | mechanism | fig_timeline (SVG) |
| 8 | ❷ Experts shard over their own replicas | mechanism | fig_dualmesh (SVG) |
| 9 | ❸ HSDP keeps per-layer traffic inside NVLink | mechanism | fig_hsdp (SVG) |
| 10 | ❹ Flat buffers make collectives zero-copy | mechanism | fig_zerocopy (SVG) |
| 11 | Full sharding removes the 6-byte floor | release (analytic) | chart bytes_per_param |
| 12 | FSDP + EP halves memory on Qwen3-30B-A3B | release (measured) | chart qwen_fsdp |
| 13 | DeepSeek-V3 recipe drops pipeline parallelism | adoption | two-col |
| 14 | The cost: 1.5× traffic that must hide | limits | two-col |
| 15 | Summary | | QR |
