# Sources — deck 50: Distributed checkpointing and elastic-resource training

## Sources read

| Tag | Source | Where |
|---|---|---|
| [K2] | Kimi Team, "Kimi K2: Open Agentic Intelligence", arXiv 2507.20534 (cached `arxiv-2507.20534.txt`). https://arxiv.org/abs/2507.20534 | Abstract; §2.3 (model size); §2.4.1 Compute Cluster; §2.4.2 Parallelism for Model Scaling (p.7-8); §2.5 (15.5T tokens); §3.3.2 Efficient Engine Switching, §3.3.3 Efficient System Startup (p.13-14); App. G (p.32) |
| [CE] | MoonshotAI/checkpoint-engine README (GitHub, fetched 2026-10-13 via `gh api`), `$SCRATCH/research/50/dl/gh-checkpoint-engine-README.md` | intro line; Architecture; Benchmark table |
| [PTB] | PyTorch blog, "Distributed Checkpoint: Efficient checkpointing in large-scale jobs" (Meta + Google, 2025-09-11). https://pytorch.org/blog/distributed-checkpoint-efficient-checkpointing-in-large-scale-jobs/ (linked from the Notion row) | "Collective communications cost", "Pinned memory staging", "Impact of checkpointing optimizations on goodput" table |
| [PTD] | PyTorch docs, `torch.distributed.checkpoint` (2.14). https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html | intro (load-time resharding; ≥ 1 file per rank; in place); `save` and `load` docstrings; DefaultLoadPlanner ("applies the resharding algorithm and computes the reads needed") |
| [V-CKPT] | VeOmni `docs/usage/checkpoint.md` (repo main @ 57e5f6d, 2026-10-12) | layout tree; "What each path holds"; "Completion"; "Staged and asynchronous saves" |
| [V-EP] | VeOmni `docs/key_features/ep_fsdp2.md` | "Sharding Dimension", "Parallelism Setup", "PyTorch Distributed Checkpoint (DCP) Support" |
| [V-XP] | VeOmni `docs/key_features/extra_parallel.md` | Motivation bullet "checkpoint save and (resharding) load for different parallelism patterns" |
| [V-SRC] | VeOmni `veomni/checkpoint/dcp_checkpointer.py` | `ModelState`, `OptimizerState`, `restore_extra_parallel_dim`, `drop_extra_parallel_dim` (placements `[Shard(1), Shard(0)]` on `(ep_fsdp, ep)`), `DistributedCheckpointer.save/load` |
| [V-GS] | VeOmni `veomni/trainer/callbacks/global_state_callback.py` | `GlobalStateCallback` docstring; `load_global_state` ("Trainer state missing on at least one rank; resuming weights only") |
| [VO] | VeOmni paper arXiv 2508.02317 §3.2.5 | "Efficient Distributed Checkpointing" (ByteCheckpoint) |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 18.5 s / 5.5 s / 1.5 s / 2.3 s | checkpointing overhead (training blocked) per checkpoint, excluding the first: baseline async to GCS / + DCP plan and metadata caching / + process-based async and pinned memory / + local checkpointing | [PTB] results table |
| Llama 3 405B, 432 H200 GPUs (54 A3 Ultra VMs) | setup of that table | [PTB] |
| ~126 s → ~47 s save; 94 s → 80 s load | total save time and load time, baseline vs + local checkpointing | [PTB] results table |
| 2× | staging-time improvement from pinned shared memory (notes only) | [PTB] "Pinned memory staging" |
| 12× | 18.5 / 1.5 = 12.3 (derived) | derived from [PTB] |
| 1.04T total, 32B activated | Kimi K2 size | [K2] §2.3 |
| 15.5T tokens | K2 pre-training tokens | [K2] §2.5, Abstract |
| H800; 8 GPUs and 2 TB RAM per node; 8×400 Gbps RoCE | K2 cluster | [K2] §2.4.1 |
| any multiple of 32 nodes | K2 elasticity unit | [K2] §2.4.2 |
| 16-way PP (virtual stages), 16-way EP, ZeRO-1 DP | K2 layout | [K2] §2.4.2 |
| ~6 TB, 256 GPUs | BF16 parameters + FP32 gradient-accumulation buffer, spread over one model-parallel group of 256 GPUs | [K2] §2.4.2 |
| 256 = 16 × 16 = 32 nodes × 8 GPUs | consistency of the model-parallel unit (derived) | derived from [K2] §2.4.1-2.4.2 |
| DP replicas = nodes / 32 | derived from the fixed 256-GPU model-parallel group | derived |
| e.g. 32 nodes → optimizer partly on CPU | small-scale optimizer offload | [K2] §2.4.2 ("When the total number of training nodes is small (e.g., 32), we can offload some optimizer states to CPU") |
| ~30 GB per GPU | GPU memory for all states, at every scale | [K2] §2.4.2 |
| 6.24 TB = 1.04T × (2 B + 4 B); 24 GB = 6.24 TB / 256 | sanity check of "~6 TB" and per-GPU share (derived, notes only) | derived |
| < 30 s | full parameter update train → inference for K2 with the checkpoint engine | [K2] §3.3.2 |
| 14.36 s (16×H20, TP16) / 16.04 s (256×H20, 16 TP16 instances) | K2-Instruct FP8 broadcast update time | [CE] Benchmark table |
| 10.19 s / 11.33 s | DeepSeek-V3.1 FP8 broadcast update, 16×H20 / 256×H20 | [CE] Benchmark table |
| 16× more GPUs, +12% time | 256/16 = 16; 16.04 / 14.36 = 1.117 (derived) | derived from [CE] |
| 3.47 s (8×H800 TP8), 6.22 s (8×H800 TP8) | GLM-4.5-Air BF16, Qwen3-235B-A22B BF16 broadcast update (notes only) | [CE] Benchmark table |
| ~20 s, "thousands of GPUs" | README summary claim for K2 (notes only) | [CE] intro |
| EP 8 × EP-FSDP 2 on 16 GPUs; expert weight [128, 2048, 768]; per-rank [16, 1024, 768] | save-side layout of the reshard example | [V-EP] "Parallelism Setup" |
| EP 4 × EP-FSDP 2 on 8 GPUs; per-rank [32, 1024, 768] | load-side layout in the reshard example (illustrative target; derived by the same rule EP × EP-FSDP = world size) | derived from [V-EP] rule |
| 2 reads per new rank | each new rank's [32, 1024, 768] slice overlaps two stored [16, 1024, 768] chunks (derived from the layouts) | derived |
| ≈ 2× | optimizer state is "roughly twice the size of the weights under Adam" (notes only) | [V-CKPT] |

## Mechanism facts (no numbers) used on slides

- DCP: each rank saves only its local shards of DTensor/ShardedTensor; at least one file per rank;
  a global `Metadata` maps every tensor to its stored chunks; load is in place into tensors already
  allocated in the destination layout; each rank reads the least data needed; "load-time resharding
  ... saving in one cluster topology and loading into another" ([PTD]).
- DCP does collectives for deduplication, global metadata, resharding and exception handling; their
  cost grows with job scale; plan and metadata caching amortize them ([PTB]).
- VeOmni step directory: `model/ckpt/` (weights, DCP, all ranks), `model/optimizer/` (DCP, all ranks),
  `model/lr_scheduler.pt` (rank 0), `loader/rank_{R}.pt` and `extra_state/rank_{R}.pt` (rank R),
  `checkpoint_manifest.json` (rank 0, last); `hf_ckpt/` is a rank-0 export after a collective gather,
  never used for resume ([V-CKPT]).
- Completion requires both the manifest and each DCP directory's `.metadata`; a crash mid-save leaves
  a step that discovery skips ([V-CKPT] "Completion").
- `stage_dir` (node-local staging) and `save_async` cannot be combined; staged multi-node saves need
  Gloo networking ([V-CKPT]).
- EP dim: at runtime VeOmni drops the EP mesh info from expert params; for DCP, `ModelState` /
  `OptimizerState` temporarily restore it as a DTensor on `(ep_fsdp, ep)` with placements
  `[Shard(1), Shard(0)]`, then drop it again after load ([V-EP], [V-SRC]).
- Resharding load across parallel patterns is supported for ExtraParallel (EP, embedding) +
  FSDP2 ([V-XP]). Caveat in [V-SRC]: for the Muon zero-comm layout (EP and FSDP on the same dim),
  resharding reads see a block-permuted layout.
- Data cursors are per DP rank; a rank without a cursor file makes the whole job resume weights only
  ([V-GS]).
- K2 checkpoint engine: each worker gets a local copy from the training engine, the full parameter set
  is broadcast to all workers, each inference engine takes only the shard it needs; parameter by
  parameter, pipelined H2D / broadcast / reload ([K2] §3.3.2, App. G; [CE]). Startup: workers
  collectively read the checkpoint once and broadcast ([K2] §3.3.3).

## Discrepancies

1. **Notion: "training continues through node-count fluctuation."** K2 says its strategy "allows Kimi K2
   to be trained on any number of nodes that is a multiple of 32" under "dynamic resource availability"
   and reuses one configuration for small and large runs; it does not describe changing node count
   inside one uninterrupted job, nor its checkpoint format. Slides say "restart on any multiple of
   32 nodes", not live elasticity.
2. **Notion: VeOmni "uses Torch Distributed Checkpoint".** True for the open-source repo ([V-CKPT],
   README); the VeOmni paper (§3.2.5) describes ByteCheckpoint instead. Slides follow the repo.
3. **Notion: "Changing parallelism or hardware needs no checkpoint rewrite."** Holds for weights and
   optimizer state (DCP load-time resharding). Per-rank data cursors and RNG state do not reshard: a job
   whose new ranks lack cursor files resumes weights only ([V-GS]); DCP gives no backward-compatibility
   guarantee across PyTorch versions ([PTD]). Slides list these as limits.
4. **K2 checkpoint-engine timing.** The K2 paper says "less than 30 seconds"; the checkpoint-engine
   README says "about 20s" and its table shows 14.36-16.04 s for K2-Instruct FP8 on H20 inference
   clusters. Different hardware/precision; slides use the README table for the chart and quote the paper's
   bound in the notes.
