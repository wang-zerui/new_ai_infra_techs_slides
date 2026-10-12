# Sources — deck 70: Megatron-FSDP / HSDP (with EP support)

## Sources read

| Tag | Source | Where |
|---|---|---|
| [MC] | NVIDIA, "Scalable Training of Mixture-of-Experts Models with Megatron Core", arXiv 2603.07685v2. https://arxiv.org/abs/2603.07685 | §2 distributed optimizer for MoE (p.11-12); §3.2.3 challenge 3 (p.16); §3.3.2 "Distributed Optimizer + EP / FSDP + EP" (p.17-18); §4.1.1 + Table 3 (p.20); §4.1.6 (p.26-27); §4.1.7 "FSDP for MoE" + Fig. 11-12 (p.27-29); Table 6 (p.29); Table 12 (p.70); Table 17 (p.74) |
| [FSDP] | Megatron-LM `docs/user-guide/features/megatron_fsdp.md` (main, fetched 2026-10-13). https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/megatron_fsdp.md | Overview/Features/Optimizations; Megatron-LM flags; "FSDP Fundamentals"; "FSDP Unit Modules"; "Double Buffering"; "Data-Parallel Sharding Strategies"; "Understanding Hybrid-FSDP"; "Mixing FSDP & Model Parallelism"; "Non-Uniform / Un-Even Model Sharding"; "Mixed-Precision"; "NCCL" |
| [DO] | Megatron-LM `docs/user-guide/features/dist_optimizer.md` | bytes-per-parameter table (bf16 params, fp32 grads: 18 vs 6 + 12/d); key steps |
| [RD] | Megatron-LM `megatron/core/transformer/moe/README.md` (cached copy) | "What's New" v0.13 (HSDP for MoE via `--num-distributed-optimizer-instances`), v0.15 (FSDP with EP); feature list "Distributed Optimizer for MoE (ZeRO-1 equivalent)"; DP communication overlap table |
| [EX] | Megatron-LM `examples/megatron_fsdp/README.md` and `sbatch_mfsdp_deepseek_v3.sh` | DeepSeek-V3 configuration summary: TP=1, EP=8, CP=1, FSDP ZeRO-3, BF16, no PP flags; MBS 4, GBS 2048 |
| [ARGS] | Megatron-LM `megatron/training/arguments.py` (main) | `--num-distributed-optimizer-instances`, `--outer-dp-sharding-strategy`, `--enable-full-sharding-in-hsdp`, `--suggested-communication-unit-size` help texts |
| [BR] | Megatron-Bridge `docs/training/megatron-fsdp.md`, `docs/performance-guide.md`, `docs/training/moe-optimization.md`, `skills/nemo-mbridge-perf-megatron-fsdp/card.yaml` (main, fetched 2026-10-13) | DDP vs distributed optimizer vs Megatron FSDP table; "FSDP reduces model-state memory, not activation memory"; card: performance claims `doc_only`, validated 2026-03-14 |
| [ZeRO] | Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", arXiv 1910.02054 | §3.1 (K = 12 for mixed-precision Adam, 16Ψ), §5, §7.1-7.2 (DP 2Ψ; ZeRO-1/2 2Ψ; ZeRO-3 3Ψ = 1.5×) |
| [I2772] | Megatron-LM issue #2772 "Using Megatron-FSDP + EP consumes more GPU memory than using EP alone" (opened 2025-12-29, closed 2026-01-14) and its comments | user measurement on Qwen3-30B-A3B, 32 H800; after the TE FusedAdam fix (TE commit fe8fad5): EP8 137.8 TFLOPS / 45,552 MB max reserved; M-FSDP + EP8 108.0 TFLOPS / 23,548 MB |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 685B params, 37B active, 18× gap | DeepSeek-V3 | [MC] §4.1.1 p.20 |
| 256 experts, top-8 | DeepSeek-V3 | [MC] §4.1.1 p.20 |
| 18 bytes per parameter = 2 (BF16 weight) + 4 (FP32 main grad) + 12 (FP32 master weight + Adam m + v) | replicated model state, non-distributed optimizer | [DO] table (18 for bf16 params, fp32 grads); the 12-byte split from [ZeRO] §3.1 (K = 12: fp32 params, momentum, variance) |
| 36.4 GB, 32.1 GB, 131.0 GB, 199.5 GB | DeepSeek-V3 BF16, PP4 × VPP4 × EP64, 256 GPUs: weights & grads; main weights & optimizer states; activations; total per GPU | [MC] Table 3 p.20 |
| 68.5 GB | weights + gradients + main weights + optimizer states per GPU (36.4 + 32.1) | derived from [MC] Table 3 |
| 80 GB | H100 capacity | [MC] §4.1.1 p.20 ("well beyond the 80 GB capacity of an H100") |
| 6 + 12/d | bytes per parameter with the distributed optimizer (bf16 params, fp32 grads) | [DO] table |
| 6-byte floor | limit of 6 + 12/d as d grows | derived from [DO] |
| 2 + 16/d (optim_grads), 18/d (optim_grads_params), 18/n (HSDP, inner size n), 6/n + 12/d (HFSDP) | per-strategy bytes per parameter (analytic) | derived: strategy definitions in [FSDP] "Data-Parallel Sharding Strategies" + HFSDP memory formula in [FSDP] "Understanding Hybrid-FSDP", applied to the 18-byte split above |
| v0.13 (HSDP for MoE), v0.15 (FSDP with EP) | release versions | [RD] "What's New" |
| 5–10× | NVLink vs cross-node bandwidth gap | [MC] §3.2.3 p.16 |
| ~10% | communication overhead reduction from non-uniform per-module sharding, Llama3 405B | [MC] §4.1.7 p.28 |
| 8–32 SMs → 1–4 SMs | SM footprint of communication kernels on NVLink with persistent buffers + NCCL UBR | [MC] §4.1.7 p.28 |
| 16–32 → 1–6 SMs | with SHARP on multi-node NVLink systems | [FSDP] "Optimizations" (notes only) |
| 45,552 MB → 23,548 MB (−48%); 137.8 → 108.0 TFLOPS (−22%) | Qwen3-30B-A3B on 32 H800: EP8 vs Megatron-FSDP + EP8 (user report, after TE fix) | [I2772] comment of 2026-01-14; −48% = 1 − 23,548/45,552 = 0.483; −22% = 1 − 108.0/137.8 = 0.216 |
| 45.6 GB, 23.5 GB | the same values in GB (÷1000, rounded) | [I2772] |
| TP1, EP8, CP1, no PP, FSDP ZeRO-3, MBS 4, GBS 2048 | Megatron-FSDP DeepSeek-V3 reference script | [EX] |
| TP2 / PP8 / EP64, VPP4 on 1,024 H100; TP1 / PP4 / EP64, VPP4 on 256 GB200 | MCore's tuned N-D-parallel DeepSeek-V3 configs | [MC] Table 17 p.74 |
| 2Ψ vs 3Ψ (1.5×) | per-step communication volume: DP / ZeRO-1/2 vs ZeRO-3 | [ZeRO] §7.1-7.2.2 |
| 2 FSDP units | at most two units un-sharded at once with double buffering | [FSDP] "Double Buffering"; "FSDP Unit Modules" |

### Derived numbers (analytic chart `charts/bytes_per_param.py`)

Assumptions (the distributed-optimizer doc's mixed-precision case): BF16 compute weights (2 B), FP32 main
gradients (4 B), FP32 main weights + Adam first and second moments (12 B) per parameter; d = number of
data-parallel ranks the parameter is sharded over. Transient un-sharded buffers (two FSDP units) and
activations are excluded.

- DDP / `no_shard`: 18 (everything replicated).
- Distributed optimizer / `optim` (ZeRO-1): 2 + 4 + 12/d = 6 + 12/d → floor 6 as d → ∞ ([DO] gives exactly this).
- `optim_grads` (ZeRO-2): 2 + (4 + 12)/d = 2 + 16/d.
- `optim_grads_params` (FSDP / ZeRO-3): 18/d.
- HSDP with inner (sharding) group n, replicated across d/n outer groups: 18/n (constant once d ≥ n); plotted
  for n = 8 (one 8-GPU NVLink node).
- HFSDP: 6/n + 12/d (formula in [FSDP]: optimizer/(inner × outer) + (gradient + weight)/inner).

Values at d = 64 (dist-opt 6.19 and FSDP 0.28 appear on slide 12; HSDP 2.25 too): DDP 18; dist-opt
6 + 12/64 = 6.19; ZeRO-2 2 + 16/64 = 2.25; FSDP 18/64 = 0.28; HSDP(8) 18/8 = 2.25; HFSDP(8) 6/8 + 12/64 = 0.94.
Ratio quoted in notes: 6.19 / 0.28 ≈ 22×.

For an expert parameter with EP = e, the EP split happens first and FSDP shards the remaining copies over
the EDP group, so the per-GPU share is (bytes)/(e · EDP) of the layer's expert parameters ([MC] §4.1.7,
[FSDP] "Mixing FSDP & Model Parallelism": FSDP sharding is always performed last).

## Mechanism facts (no numbers) used on slides

- FSDP shards optimizer state, weight gradients and weights; activations are not sharded ([FSDP]
  "FSDP Fundamentals"; [BR]).
- Parameters are all-gathered per FSDP unit before forward and before backward and freed right after;
  weight gradients are reduce-scattered after backward; next unit is prefetched on separate CUDA streams;
  overlapping needs two units of un-sharded memory ([FSDP] "FSDP Fundamentals", "FSDP Unit Modules").
- FSDP = DDP's all-reduce decomposed into reduce-scatter + sharded optimizer step + all-gather ([FSDP]).
- Strategies: `--data-parallel-sharding-strategy {no_shard, optim, optim_grads, optim_grads_params}` (DDP, ZeRO-1,
  ZeRO-2, ZeRO-3); `--outer-dp-sharding-strategy {no_shard, optim}` (HSDP, HFSDP);
  `--num-distributed-optimizer-instances` = DP-Outer size ([FSDP], [ARGS]).
- HSDP: full sharding within DP-inner, replication across DP-outer; outer gradients all-reduced in the last
  backward before the optimizer step. HFSDP: optimizer state additionally sharded across outer ranks; outer
  reduce-scatter / all-gather happen once per optimization cycle ([FSDP]).
- Dual DeviceMesh: dense modules on the primary mesh (DP-Shard, DP-Outer, TP, CP), expert FFNs on an Expert
  DeviceMesh with FSDP scoped to EDP; expert AllGather / ReduceScatter stay within small EDP groups; separate
  outer-DP groups for expert and non-expert parameters ([MC] §4.1.7; [FSDP] "Mixing FSDP & Model
  Parallelism": FSDP sharding is always performed last, dense mesh uses DP and CP ranks for sharding).
- FSDP+EP avoids PP/VPP stage balancing, MTP/output-layer placement and vision-encoder partitioning ([MC]
  §4.1.7 "PP-free Flexibility").
- Zero-copy: Megatron-FSDP flattens and concatenates a module's parameters and shards the flat buffer
  non-uniformly (FSDP2 shards each parameter uniformly and needs copies); two persistent buffers cycled
  across units; NCCL user buffer registration ([MC] §4.1.7 + Fig. 11-12; [FSDP] "Double Buffering",
  "Non-Uniform / Un-Even Model Sharding", "NCCL").
- Distributed optimizer for MoE: expert params reduce over expt_dp_group, dense over dp_cp_group ([MC] §2).
- Recommended flags: `--use-megatron-fsdp`, `--ckpt-format fsdp_dtensor`, unset CUDA_DEVICE_MAX_CONNECTIONS
  (may slightly affect TP and CP performance), `--init-model-with-meta-device`, `--calculate-per-token-loss`,
  `--grad-reduce-in-bf16`, `--use-nccl-ub --fsdp-double-buffer --fsdp-manual-registration` ([FSDP]).
- Double buffering needs depth-wise symmetric units; hybrid models use the MaxPool allocator ([FSDP]).
- `fsdp_dtensor` checkpoints are not interchangeable with `torch_dist`/`zarr` ([BR]); conversion script in [EX].
- Hybrid (dynamic) CP asserts `not use_megatron_fsdp` (Megatron-LM `arguments.py`).

## Discrepancies

1. **Notion "complements the distributed optimizer (ZeRO-1 equivalent)".** Correct, and in the code Megatron-FSDP
   *uses* the distributed optimizer (`--use-distributed-optimizer` is set automatically, [FSDP] feature table);
   its `optim` strategy is the ZeRO-1 point of the same ladder. Slides present them as one ladder.
2. **Notion "hybrid sharding (--num-distributed-optimizer-instances)".** That flag sets the DP-outer size. For the
   classic distributed optimizer it gives HSDP-style partial optimizer sharding (MCore v0.13 "Hybrid Shard
   Data-Parallel support for MoE"); for Megatron-FSDP it is combined with `--outer-dp-sharding-strategy`
   (`no_shard` = HSDP, `optim` = HFSDP). Slides name both flags.
3. **Notion "model-state sharding and DP become interchangeable".** Not a phrase from the sources; the closest
   source statements are [MC] §4.1.7 "Per-GPU memory and collective volume both scale with EDP size" and
   "PP-free flexibility". Slides phrase it as "DP size becomes the memory knob" and label it as our summary.
4. **No first-party throughput benchmark for Megatron-FSDP (+EP) was found** ([BR] card marks performance
   claims `doc_only`). The only end-to-end measurement used is a public user report ([I2772]); slides label it
   as such. [MC] contributes the ~10% and SM-count numbers.
5. **Parameter count.** [MC] uses 685B for DeepSeek-V3 (with MTP); [EX] says 671B. Slides use [MC]'s 685B
   together with its Table 3.
