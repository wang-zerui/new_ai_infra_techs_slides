# Story — deck 49: Precision-aware optimizer (BF16/FP8 optimizer states)

- **Where it lives** (prosperity): Adam is the default optimizer for LLM pretraining; mixed precision keeps an
  FP32 master weight and two FP32 moments per parameter (12 B), six times the BF16 weight. Megatron-Core
  (`--use-precision-aware-optimizer`, BF16 states since MCore v0.13) + Transformer Engine `FusedAdam`;
  used in MCore's best DeepSeek-V3 configs on H100 and GB200.
- **Pain number** (betrayal): DeepSeek-V3 (BF16, PP4 × VPP4 × EP64, 256 GPUs) spends **32.1 GB per GPU** on
  main weights and optimizer states, 16% of a 199.5 GB footprint, on an 80 GB H100.
  Hook number: Llama-405B needs ~4,860 GB of Adam state vs ~810 GB of BF16 weights.
- **Failed suitors**:
  1. Shard it (distributed optimizer, FSDP): 12/d per rank, but the total stays 12 B per parameter and needs
     DP ranks (deck 70).
  2. Offload it to CPU: saves 15–20 GB but costs 0.1–0.2 s per iteration and needs a fast host link.
  3. Store and compute in BF16 (naive): an update smaller than half a BF16 spacing of the weight is rounded
     away; this is why DeepSeek-V3 keeps master weights and gradients in FP32.
- **Villain**: tying storage precision to compute precision: Adam state is stored in the format the update is
  computed in, so making it smaller seems to make the update less precise.
- **Turn / insight**: storage precision and compute precision can be decoupled: moments tolerate BF16 storage
  as long as the update is computed in FP32 (DeepSeek-V3: no observable degradation), and the FP32 master
  weight's top 16 bits already exist as the BF16 weight.
- **Mechanisms**:
  ❶ In-kernel upcast: FusedAdam reads BF16/FP16/FP8 moments, computes in FP32, writes back; BF16 is cast in
     registers (fused), FP16/FP8 use one FP32 scale per tensor (answers: precision of the update).
  ❷ Master-weight remainders: FP32 master = BF16 weight + int16 low half (exact, round-to-nearest fix), on by
     default with TE ≥ 2.1 (answers: master weights must stay FP32).
  ❸ Composition: 6 + 12/d → 6 + 8/d with the distributed optimizer; smaller states make CPU offload cheaper
     (offload requires this flag); Megatron-FSDP exposes the same dtypes.
- **Release**: optimizer state 12 → 8 B (BF16 moments) → 6 B (+ remainders) → 4 B (FP8 moments) per parameter
  (analytic); DeepSeek-V3: 10–12 GB of 32.1 GB saved per GPU with no performance overhead (cast in fused
  kernel); part of the recipe that brings DeepSeek-V3 from 199.5 GB to < 80 GB per GPU; listed in both final
  DeepSeek-V3 configurations (Table 18).
- **Limits**: Adam only; TE FusedAdam required (≥ 2.1 for remainders); CUDA-graph capture only with FP32 or
  BF16 moments, FP32 master and no remainders; FP16/FP8 states: one scale per tensor and temporary FP32
  buffers; FP16 master is lossy; no published convergence study for FP8 moments in the sources read.

## Slide plan (15 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Adam's state outweighs the model six to one | prosperity | chart llama405b |
| 3 | Optimizer state takes 32 GB per GPU | pain | `%% big: 32.1 GB` |
| 4 | Sharding and offloading keep all 12 bytes | failed suitors | two-col |
| 5 | Naive BF16 Adam rounds small updates away | villain | fig_naive (SVG) |
| 6 | Insight: store compactly, compute in FP32 | turn | boxed sentence (build) |
| 7 | FusedAdam upcasts, updates, and rounds once | overview | fig_kernel (SVG) |
| 8 | ❶ Moments travel in BF16 or FP8 | mechanism | fig_moments (SVG) |
| 9 | ❷ BF16 weight plus remainder is exact FP32 | mechanism | fig_remainder (SVG) |
| 10 | ❸ Smaller state shards and offloads cheaper | mechanism | fig_compose (SVG) |
| 11 | Optimizer state drops from 12 to 6 bytes | release (analytic) | chart bytes_stack |
| 12 | BF16 moments free 10–12 GB per GPU | release (measured) | chart dsv3_state |
| 13 | DeepSeek-V3 recipes keep low-precision states | adoption | two-col |
| 14 | The cost: Adam only, one scale per tensor | limits | two-col |
| 15 | Summary | | QR |
