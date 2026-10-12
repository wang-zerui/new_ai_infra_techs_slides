# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** VeOmni uses Torch Distributed Checkpoint for parallelism-agnostic save/load; K2 designs variable parallelism (16-way PP + 16-way EP + ZeRO-1 DP) for clusters whose resources change over time.
**After:** VeOmni's open-source repo saves weights and optimizer state with Torch Distributed Checkpoint, each rank writing its own shards, and loads reshard into a new layout; K2 fixes a 256-GPU model-parallel unit (16-way PP x 16-way EP) and varies only ZeRO-1 DP, so it can train on any multiple of 32 nodes as resources change.
**Why:**
- DCP is what the VeOmni repo uses ([V-CKPT], [V-SRC]); the VeOmni paper describes ByteCheckpoint instead ([VO] §3.2.5), so the claim is scoped to the repo.
- DCP mechanism: per-rank shard files plus global metadata, load-time resharding ([PTD]).
- K2's parallelism is not "variable" as a whole: PP and EP stay fixed (256 GPUs = 32 nodes x 8 GPUs, derived) and only ZeRO-1 DP scales with node count; "any number of nodes that is a multiple of 32" ([K2] §2.4.1-2.4.2).

## Impact / Measured Results
**Before:** Changing parallelism or hardware needs no checkpoint rewrite; training continues through node-count fluctuation.
**After:** Weights and optimizer state reload under a new parallel layout with no checkpoint rewrite, but per-rank data cursors and RNG state do not reshard; K2 trains on any multiple of 32 nodes (resizing a running job is not described).
**Why:**
- "No checkpoint rewrite" holds for weights and optimizer state; a job whose new ranks lack cursor files resumes weights only ([V-GS]), and DCP gives no format guarantee across PyTorch versions ([PTD]).
- K2 describes training on any multiple of 32 nodes under dynamic resource availability, not a job that keeps running through node-count changes ([K2] §2.4.2).

## Other properties
none
