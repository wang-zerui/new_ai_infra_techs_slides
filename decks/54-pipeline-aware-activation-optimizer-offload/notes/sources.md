# Sources — deck 54: Pipeline-aware activation / optimizer offload

## Sources read

| Tag | Source | Where |
|---|---|---|
| [MC] | NVIDIA, "Scalable Training of Mixture-of-Experts Models with Megatron Core", arXiv 2603.07685v2. https://arxiv.org/abs/2603.07685 (cached text) | Table 3 (p.20), §4.1.4 (p.22), §4.1.5 Motivation / High-Level Idea / Forward / Backward (p.23-24), PP and VPP Handling / Key Technical Features / Peak Memory / Performance / Table 5 (p.25), §4.1.6 (p.25-27), Table 6 (p.29), §6.2 (p.58), Table 13 (p.72), Table 18 + §9.2.2 (p.74-75) |
| [RD] | Megatron-LM `megatron/core/transformer/moe/README.md` (cached) | What's New "MCore dev (2026/01): Pipeline-aware fine-grained activation offloading"; Memory Optimization table (`--optimizer-cpu-offload`); Fine-grained Activation Offloading usage |
| [DOC] | `docs/user-guide/features/fine_grained_activation_offloading.md` (main, fetched 2026-10-13 via `gh api`) | offloadable modules; tuning parameters (`--min-offloaded-tensor-size`, `--activation-offload-fraction`, `--delta-offload-bytes-across-pp-ranks`); CUDA graph constraints; combination with recompute; architecture (PipelineOffloadManager, ChunkOffloadHandler); warmup and margin |
| [FGO] | `megatron/core/pipeline_parallel/fine_grained_activation_offload.py` (main) | `PipelineOffloadManager` (d2h/h2d streams, pinned CPU pool, chunk queue, `flush`, `post_warmup_callback`), `ChunkOffloadHandler` (`on_group_commit_forward`, `bulk_offload`, `should_bulk_offload`, `on_group_start_backward`, `bulk_reload`, `pre_reload_last_layer`) |
| [CFG] | `megatron/core/transformer/transformer_config.py` (main) | `fine_grained_activation_offloading`, `offload_modules`, `delta_offload_bytes_across_pp_ranks` docstrings; legacy `cpu_offloading` checks (no PP > 1, no recompute) l.2363-2378 |
| [ARG] | `megatron/training/arguments.py` (main) and (dev) | main: TE-only + `NVTE_CPU_OFFLOAD_V1=1` for TE ≥ 2.10; `--optimizer-cpu-offload` requires precision-aware optimizer. dev: `--offload-optimizer-states` = deprecated alias of `--chunked-optimizer-state-offload`; needs torch_dist checkpoints, synchronous optimizer saves, no optimizer CUDA graphs; not with Megatron-FSDP |
| [OCFG] | `megatron/core/optimizer/optimizer_config.py` (main and dev) | `optimizer_cpu_offload`, `optimizer_offload_fraction`; dev: `chunked_optimizer_state_offload` ("Keep selected optimizer tensor states and master weights on CPU between updates") |
| [OPTDOC] | `docs/user-guide/features/optimizer_cpu_offload.md` (main) | `--optimizer-cpu-offload --optimizer-offload-fraction 1.0 --use-precision-aware-optimizer`; `--overlap-cpu-optimizer-d2h-h2d` |
| [K2] | Kimi Team, "Kimi K2: Open Agentic Intelligence", arXiv 2507.20534 (cached) | §2.4.1 (H800, 2 TB RAM per node) p.7; §2.4.2 (optimizer offload when nodes are few; ~30 GB per GPU for states) p.8; §2.4.3 (selective recompute, FP8 storage, activation CPU offload) p.8-9; Fig. 7 |
| [K3] | Kimi Team, "Kimi K3: Open Frontier Intelligence", arXiv 2607.24653 (cached) | abstract (2.8T, 104B active) p.1; §5.2.2 Memory-Efficient Training (unified activation manager, remote offload across PP ranks via Mooncake, Pipeline ZeRO-2 gradient shards in CPU memory) p.20; Fig. 11 |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 199.5 GB; 36.4 / 32.1 / 131.0 GB; 80 GB | DeepSeek-V3 BF16 per-GPU breakdown (PP4 × VPP4 × EP64, 256 GPUs); H100 capacity | [MC] Table 3 p.20 |
| 685B / 37B | DeepSeek-V3 total / active | [MC] §1.2 p.6 |
| 2 TB RAM per 8-GPU node, H800 | Kimi K2 cluster | [K2] §2.4.1 p.7 |
| 15% (pain), 800 → 920 TF/s, TP2/EP16 → TP1/EP64, 172 → 175 GB, +1.7% | Qwen3-235B with fine-grained offloading | [MC] Table 5 p.25 and §4.1.5 Performance ("improves throughput by 15.0% with nearly the same memory cost") |
| 169 → 151 GB (−10.7%), 945 → 930 TF/s (−1.6%) | DeepSeek-V3 full, TP1 PP8 EP32 VPP4, MXFP8, with offloading | [MC] Table 5 p.25 |
| 10–18% memory, 1.6–2% throughput | summary of offloading results | [MC] §4.1.5 p.25 |
| ~33% | full-layer recompute overhead | [MC] §4.1.4 p.22 |
| L × layer_input (+1 × intermediate) vs 1 × layer_input (+1 × intermediate) | GPU peak, full recompute vs offload | [MC] §4.1.5 "Peak Memory Advantage" p.25 |
| 4 stages, 6 micro-batches; 4/3/2/1 micro-batches held | analytic 1F1B schedule drawn in fig_inflight (backward = 2 × forward); stage s holds P − s micro-batches at peak | analytic (standard 1F1B); qualitative support [K2] §2.4.3, [K3] §5.2.2 |
| 15–20 GB (47–62% of 32.1 GB); 0.1–0.2 s per iteration | DeepSeek-V3 optimizer-state offload (GB200 NVLink-C2C context) | [MC] §4.1.6 p.26-27 |
| ~2% worst case; 256 H100; ~50% MFU; ≥ 16K tokens | optimizer CPU offloading, DeepSeek-V3 long context | [MC] §6.2 p.58 |
| toolbox chart: 42.4 / ~26.3 / 15–20 / ~16 / 10–12 GB | per-technique savings on DeepSeek-V3 vs Table 3 | [MC] Table 4 p.24; §4.1.2 p.21; §4.1.6 p.27; §4.1.3 p.22; §4.1.6 p.26 |
| 1T (K2); 2.8T (K3) | model sizes | [K2] abstract; [K3] abstract |
| TE ≥ 2.10 + NVTE_CPU_OFFLOAD_V1=1 | requirement for fine-grained activation offloading | [ARG] main |
| 2026/01 | MCore dev release listing pipeline-aware fine-grained activation offloading | [RD] |

## Mechanism facts (no numbers) used on slides

- Forward: offload "immediately after module computation, running in parallel with the next module's computation
  via a dedicated D2H stream"; last layer's activations are not offloaded ([MC] p.24). Code: `on_group_commit_forward`
  makes the D2H stream wait on the compute stream and bulk-offloads the tensors captured by `saved_tensors_hooks`
  in that group into a pinned CPU pool; forced-released tensors get `untyped_storage().resize_(0)` ([FGO]).
- Backward: "Layer-Staggered Reload": reload the same module of the next layer (in backward order) while computing
  the current layer; reload after each module's backward, so only one activation per module type is resident
  ([MC] p.24). Code: `on_group_start_backward` → `bulk_reload` pops the last offloaded group; `on_group_commit_backward`
  makes the compute stream wait on that group's reload event; `post_warmup_callback` keeps the last group of each
  module name on GPU (offload margin) ([FGO]).
- PP/VPP: a `ChunkOffloadHandler` per (micro-batch, VPP stage); "VPP stages in reverse order (FILO) and microbatches
  in normal order (FIFO)" ([MC] p.25); `pre_reload_last_layer` of the next backward chunk; `delta_offload_bytes_across_pp_ranks`
  keeps more on GPU at higher PP ranks "because they have fewer microbatches in flight" ([DOC], [FGO]).
- Module choice and composition with recompute (moe_act: recompute output and offload input) ([MC] p.25; [DOC]).
- CUDA graphs: attn_norm / mlp_norm cannot be offloaded; TE impl; scope must include attn and moe_router ([DOC]).
- Optimizer state offload: "keeps the optimizer step on GPU but transfers optimizer states (exp_avg, exp_avg_sq) and
  master weights to CPU after optimizer.step() and reloads them before the next step" ([MC] p.26). CPU optimizer
  (`--optimizer-cpu-offload`): CPU step with grads D2H and params H2D ([OPTDOC], [OCFG]).
- Kimi K2: recompute LayerNorm, SwiGLU, MLA up-projections, MoE down-projections; FP8-E4M3 storage for MoE
  up-projection and SwiGLU inputs; "All remaining activations are offloaded to CPU RAM"; copy engine overlaps
  compute and communication; "During the 1F1B phase, we offload the forward activations of the previous micro-batch
  while prefetching the backward activations of the next"; EP communication remains fully overlapped ([K2] p.8-9).
- Kimi K3: recomputation, quantization, offload and remote offload as per-tensor storage policies; remote offload of
  activations to other PP ranks (Mooncake Transfer Engine) to balance PP ranks; sharded gradients stored in CPU
  memory with a double grad buffer on GPU ([K3] §5.2.2 p.20).

## Discrepancies

1. **"Offloads activations to CPU per pipeline stage."** The mechanism is per *module* (each listed submodule's saved
   input), tracked per (micro-batch, VPP chunk) handler; PP awareness is in the ordering and the per-rank amount
   ([MC] p.25, [DOC]). Slides describe the module granularity.
2. **`--offload-optimizer-states`.** This flag appears in the paper (Table 13) and on the Megatron-LM dev branch, where
   it is now a deprecated alias of `--chunked-optimizer-state-offload`; main and the MoE README document
   `--optimizer-cpu-offload`, which is a different mechanism (the step runs on the CPU) ([ARG], [OCFG], [RD]).
   Slides show both and label which is which.
3. **"Sustains stable trillion-scale MoE training on H800 clusters with 2TB of host memory per node."** The hardware
   facts are Kimi K2's ([K2] p.7) and K2 does offload all remaining activations, but the report does not attribute
   training *stability* to offload (its stability claim is about MuonClip / zero loss spikes) and gives no offload
   measurement; K3's report does not name its GPUs. The measured numbers on the slides come from Megatron-Core
   (Table 5, §4.1.6, §6.2).
4. **"K2/K3 combine CPU activation offload with selective recomputation."** Correct, and both also compress
   activations (K2: FP8 storage; K3: block-wise FP8 + remote offload to other PP ranks) ([K2], [K3]).
5. **Table 5 hardware** is not stated; the DeepSeek-V3 row uses MXFP8 (a Blackwell format) and 169 GB exceeds an
   80 GB H100, so it is a Blackwell-class run. Slides do not name the GPU for Table 5.
