# Story — deck 50: Distributed checkpointing and elastic-resource training

- **Where it lives** (prosperity): every long pretraining run checkpoints and restarts many times.
  VeOmni (child of deck 28) saves with PyTorch Distributed Checkpoint (DCP); Kimi K2 (1.04T-parameter
  MoE, 32B active, 15.5T tokens on an H800 cluster) chose a parallel layout that runs on any multiple
  of 32 nodes because resources change during training.
- **Pain number** (betrayal): even asynchronous DCP saves stalled training **18.5 s per checkpoint**
  for Llama 3 405B on 432 H200 GPUs (PyTorch/Google blog, baseline column). Frequent saves are needed
  because interruptions are frequent; each one stalls every GPU.
- **Failed suitors**:
  1. Per-rank dumps (`torch.save` of each rank's shard): files only make sense to the exact layout
     that wrote them; a different TP/PP/EP/DP shape needs an offline conversion pass.
  2. Gather-to-one-file (full state dict on rank 0): layout-free but serializes TBs through one rank
     (K2: ~6 TB of BF16 weights + FP32 grad buffers per model-parallel group).
- **Villain**: a checkpoint that records *which rank* held bytes instead of *which global slice*.
- **Turn / insight**: a shard is just a slice of a global tensor. Write each rank's slice with its global
  offsets once; any later layout, with any number of GPUs, is just a different set of reads.
- **Mechanisms**:
  ❶ Save: every rank writes only its local shards + one global metadata file (chunk offsets); weights
     and optimizer are separate DCP dirs; completion = manifest + `.metadata`; async or staged.
  ❷ Load-time resharding: the new job allocates DTensors in its own layout; the planner intersects the
     requested chunks with the stored ones and reads only the overlap. VeOmni temporarily restores the
     EP dim so experts are stored as one global tensor (EP 8 × FSDP 2 → EP 4 × FSDP 2 example).
  ❸ Elastic layout (K2): keep the model-parallel unit fixed (16-way PP × 16-way EP = 256 GPUs =
     32 nodes) and let only ZeRO-1 DP grow or shrink; optimizer shards spread over replicas or go to
     CPU at small scale; ~30 GB of state per GPU at every scale.
- **Release**: save stall **18.5 s → 1.5 s** with DCP plan/metadata caching + process-based async +
  pinned memory (12× less, derived); K2 trained on any multiple of 32 nodes with one config; the same
  resharding idea moves K2's 1T weights from the training layout to 16 TP16 inference instances
  in **16.04 s** (checkpoint engine).
- **Limits**: DCP collectives (dedup, metadata, resharding) grow with scale; no backward compatibility
  across PyTorch versions; per-rank data cursors do not reshard (VeOmni resumes weights only if a rank
  has no cursor); K2's elasticity is in 32-node quanta and small scales offload optimizer state to CPU;
  staged and async saves cannot be combined; local checkpoints cost slightly more overhead (2.3 s).

## Slide plan (13 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Long runs restart on whatever GPUs remain | prosperity | fig_timeline (SVG) |
| 3 | Even async checkpoint saves stall every GPU | pain | `%% big: 18.5 s` |
| 4 | Rank-file checkpoints fit only one layout | villain + suitors | fig_bound (SVG) |
| 5 | Insight: a shard is a slice of a global tensor | turn | fig_chunks (SVG) |
| 6 | Save once, then load into any mesh | overview | fig_overview (SVG) |
| 7 | ❶ Every rank writes only its own shard | mechanism | fig_tree (SVG) |
| 8 | ❷ The loader reads only overlapping chunks | mechanism | fig_reshard (SVG) |
| 9 | ❸ K2 fixes model parallelism, flexes only DP | mechanism | fig_k2 (SVG) |
| 10 | Optimized DCP cuts the save stall 12× | release | chart dcp_overhead |
| 11 | K2 weights reshard into 16 replicas in 16 s | result | chart ckpt_engine |
| 12 | The cost: collectives, cursors, 32-node quanta | limits | two-col text |
| 13 | Summary | | QR github.com/ByteDance-Seed/VeOmni |
