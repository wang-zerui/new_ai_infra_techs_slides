# Sources — deck 35: Mega MoE fused MegaKernel

## Sources read

| Tag | Source | Where |
|---|---|---|
| [V4] | DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence", arXiv 2606.19348v1 (26 Apr 2026). https://arxiv.org/abs/2606.19348 (cached text) | §1 p.4-5 (infra list), §3.1 + Fig. 5 (p.15-16), §3.3 (p.18-19), §4.2.1 (p.24-25) |
| [RD] | DeepGEMM README (cached `gh-deepgemm-README.md`) | "News" lines 11-18; "Mega MoE" lines 112-144; env var `DG_COMM_KERNEL_DEBUG` line 187 |
| [PR304] | DeepGEMM PR #304 "[Public release 26/04] Introducing Mega MoE …" (merged 2026-04-17) | body |
| [PR316] | DeepGEMM PR #316 "Add various optimizations and Mega MoE benchmarks" (merged 2026-04-24) | body: two benchmark tables |
| [PR364] | DeepGEMM PR #364 (merged 2026-06-23) | body: BF16 Mega MoE, ring buffer |
| [PR377] | DeepGEMM PR #377 "Public release 26/07" (merged 2026-07-15) | body: MegaMoE bullets |
| [PR432] | DeepGEMM PR #432 "Public Release 26/09" (merged 2026-09-10) | body: Mega MoE for V4.1, determinism switch |
| [PR462] | DeepGEMM PR #462 "Public release 26/09/30" (merged 2026-09-30) | body: locality-aware Mega MoE, combine-order fix |
| [K] | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` at main 057ca596 (2026-09-30) | whole file (warp roles, dispatch pull, epilogues, combine) |
| [SCHED] | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh` (same commit) | `MegaMoEScheduler`, `L2KBlockDependency`, `get_num_l1_warmup_waves` |
| [LAYOUT] | `deep_gemm/include/deep_gemm/layout/mega_moe.cuh`, `layout/sym_buffer.cuh` (same commit) | `MegaMoEBuffer`, `MegaMoESignals`, `SymBuffer::map`, `kNumMaxRanks = 72` |
| [BAR] | `deep_gemm/include/deep_gemm/comm/barrier.cuh` | `grid_sync`, `nvlink_barrier` |
| [HEUR] | `csrc/jit_kernels/heuristics/mega_moe.hpp`; `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` | thread layout (lines 126, 220-221), launch (grid = all SMs, cluster 2, lines 191-197), "must use the whole GPU" comment |
| [API] | `csrc/apis/mega_moe.hpp`; `deep_gemm/mega/__init__.py` | symmetric buffer allocation (`symm_mem.empty` + `rendezvous`), arch dispatch (`arch_major == 10` only), SwiGLU-only asserts |
| [TEST] | `tests/test_mega_moe.py` at main, and at the PR #316 merge commit 891d57b4 | baseline `run_baseline`, bitwise check, HBM/NVLink byte model |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 256 routed + 1 shared experts, top-6, d = 4096, I = 2048; 284B total / 13B active; 43 layers | DeepSeek-V4-Flash config | [V4] §4.2.1 p.24; dims also in [PR316] |
| 384 routed + 1 shared experts, top-6, d = 7168, I = 3072; 1.6T / 49B; 61 layers | DeepSeek-V4-Pro config | [V4] §4.2.1 p.25; dims also in [PR316] |
| EP8 (8-way expert parallelism), batch = tokens per rank 1 / 512 / 8192 / 32768, averages over 8 ranks | benchmark setup | [PR316] |
| V4-Flash: 56.5 / 146.5 / 1283.1 / 4855.5 µs; 5 / 1056 / 1928 / 2038 TFLOPS; 1311 / 3192 / 998 / 794 GB/s HBM; 1 / 266 / 499 / 529 GB/s NVLink; 1.96 / 1.73 / 1.56 / 1.62× vs legacy | fused kernel time and rates per batch size | [PR316] table 1 |
| V4-Pro: 108.1 / 369.6 / 2818.5 / 10655.2 µs; 7 / 1098 / 2304 / 2438 TFLOPS; 1758 / 4619 / 1094 / 692 GB/s HBM; 1 / 182 / 393 / 417 GB/s NVLink; 1.61 / 1.54 / 1.50 / 1.54× vs legacy | same | [PR316] table 2 |
| **≈ 111 µs** (derived) | five-kernel ("legacy") time of one V4-Flash MoE layer at 1 token/rank, EP8 | derived: 56.5 µs × 1.96 = 110.7 µs ([PR316] table 1, row 1) |
| 1.50–1.96× | range of speedups vs legacy over both models and all four batch sizes | [PR316] (min 1.50 = Pro 8192, max 1.96 = Flash 1) |
| 1.50–1.73× general inference; up to 1.96× latency-sensitive (RL rollout, agent serving) | paper's summary of the same kernel | [V4] §3.1 p.16 |
| 1.42× vs 1.92× | theoretical speedup of per-stage overlap (Comet) vs expert-wave pipelining, V4-Flash config | [V4] Fig. 5 (p.15) |
| "communication time < computation time" within one MoE layer | profiling claim behind the design | [V4] §3.1 p.15 |
| 2038 / 2438 TFLOPS with 529 / 417 GB/s NVLink at 32768 tokens/rank | large-batch regime | [PR316] |
| 5 kernels | non-fused path in the benchmark: DeepEP dispatch, grouped FP8×FP4 GEMM, TileLang SwiGLU+FP8 kernel, grouped GEMM, DeepEP combine | [TEST] at 891d57b4, `run_baseline` |
| 11 kernels into 1; 20% faster than the previous version; 10%+ end-to-end inference | July 2026 release after fusing the shared expert and the new L1/L2-interleaving scheduler | [PR377] |
| ~15% faster at low latency, ~4% at high throughput | Mega MoE optimizations for V4.1 | [PR432] |
| 100 W saved; 5–10% higher GPU clock for heavy workloads | locality-aware Mega MoE | [PR462] |
| PyTorch ≥ 2.9 | symmetric-memory requirement | [RD] line 118, [PR304] |
| 72 ranks | `kNumMaxRanks` in `SymBuffer` | [LAYOUT] sym_buffer.cuh |
| 512 threads/CTA = 128 dispatch + 128 (TMA-A, TMA-B, MMA, scheduler warps) + 256 epilogue; 1 CTA per SM; 2-CTA clusters; grid = all SMs | kernel launch shape | [HEUR] lines 126, 220-221; impl .hpp lines 191-197; [K] warp-role branches |
| per-32-element UE8M0 scales; FP8 E4M3 activations; FP4 E2M1 or FP8 weights; FP32 accumulate; BF16 combine | data types | [K] (kGranK = 32, `a_dtype_t`, `weight_dtype_t`, epilogue casts); [RD] |
| 10·I vs 2·I bytes (analytic, 5×) | linear-1 → linear-2 hand-off bytes per routed token. Non-fused: GEMM writes BF16 [2I] (4I B), SwiGLU kernel reads it (4I B), writes FP8 [I] (I B), linear-2 reads it (I B) = 10·I. Fused: epilogue writes FP8 [I] (I B), linear-2 reads it (I B) = 2·I. Scale factors (I/32 B) ignored in both. | derived from [TEST] `run_baseline` tensor shapes (`l1_y` BF16 `(n, 2I)` → `swiglu_apply_weight_to_fp8` → FP8) and [K] L1 epilogue (TMEM → SwiGLU → FP8 TMA store into `l2_token_buffer`); the fused accounting matches [TEST]'s own HBM model ("L1 output write (FP8)" + "L2 acts read (FP8)") |
| 4.86 ms | V4-Flash fused kernel time at 32768 tokens/rank (4855.5 µs, rounded) | [PR316] table 1 |
| "2+ PFLOPS" | 2038 (Flash) and 2438 (Pro) TFLOPS at 32768 tokens/rank | [PR316] |
| 3·d bytes per routed token over NVLink (d FP8 pulled + 2d BF16 written back) | NVLink byte model | [TEST] `num_nvlink_bytes = num_recv_tokens * hidden * 3`; [V4] §3.1 "3h bytes of communication (FP8 Dispatch + BF16 Combine)" |

Derived numbers: ≈ 111 µs (above). The legacy times for the other points are only in the speaker notes:
Flash 146.5 × 1.73 ≈ 253 µs, 1283.1 × 1.56 ≈ 2002 µs, 4855.5 × 1.62 ≈ 7866 µs; Pro 108.1 × 1.61 ≈ 174 µs,
369.6 × 1.54 ≈ 569 µs, 2818.5 × 1.50 ≈ 4228 µs, 10655.2 × 1.54 ≈ 16409 µs. The 10·I → 2·I hand-off model is
analytic and labelled so on the slide.

Charts: `charts/speedup.py` plots the eight "Speedup (vs legacy)" values of [PR316]; `charts/tflops.py` plots
the eight TFLOPS values of [PR316] and annotates the NVLink GB/s of the 32768-token rows.

## Mechanism facts (no numbers) used on slides

- Five stages in one kernel: dispatch, linear 1, SwiGLU, linear 2, combine; overlapping NVLink communication
  and tensor-core compute; requires multi-process launch with symmetric memory ([RD] lines 112-114; [PR304]).
- Symmetric memory: `torch.distributed._symmetric_memory.empty` + `rendezvous` gives every rank's base
  pointer; `SymBuffer::map(ptr, rank)` = local address + (peer base − own base) ([API] `__init__.py`; [LAYOUT]).
- Buffer contents (one allocation, same layout on every rank): signals/workspace; inputs x (FP8), x_sf,
  topk_idx, topk_weights; shared-expert buffers; ring buffers for L1 tokens (+SF, +routing weight) and L2
  tokens (+SF); combine buffer with (top-k + shared) BF16 slots per token ([LAYOUT] `MegaMoEBuffer`).
  The caller copies x, x_sf, topk_idx, topk_weights into the buffer before each call ([RD] lines 132-137).
- Dispatch warps ([K] lines 346-690; pull loop from line 433): count tokens per expert (shared-memory atomics) → claim per-SM
  offsets with global atomics → write each token's source index into the destination rank's workspace
  (NVLink store) → grid sync; SM 0 publishes per-expert counts to peers → NVLink barrier → pull loop: each
  warp picks a received slot (round-robin over source ranks), TMA-loads the token's FP8 row from the peer
  (remote address) into shared memory in chunks and TMA-stores it into the local L1 ring; copies its scales
  and routing weight; records (src rank, token, top-k slot) metadata; release-adds the block's
  `l1_full_count`. [V4] §3.1: pull-based dispatch avoids the notification latency of fine-grained push.
- Other warps ([K] lines 694-1000): TMA-A load warp (waits `l1_full_count` for linear-1 tiles; per-K-block bitmask for
  linear-2 tiles), TMA-B weight load warp, MMA warp (tcgen05 block-scaled MMA, 2-CTA, FP32 accumulators in
  tensor memory, leader CTA only), scheduler warp, two epilogue warpgroups.
- Scheduler ([SCHED]): persistent; tasks claimed from global atomic counters in pool-block order (experts in
  order); linear-1 tasks run a few warmup waves ahead, then each scheduler alternates linear-2 and linear-1
  tasks ("interleaving L1/L2 blocks", [PR377]); an L2 task waits until all L1 tasks of its M-block have
  been claimed (deadlock-free); `L2KBlockDependency`: each finished L1 N-block toggles one bit, an L2
  K-block starts once its bits are set; ring slots are recycled through `l1_empty_count`/`l2_empty_count`.
  Shared-expert linear-1 tasks do not depend on dispatch and are scheduled first.
- L1 epilogue ([K] lines 1066-1255): TMEM → registers; clamp; SwiGLU silu(gate)·up (gate/up interleaved
  by 8 in the weights); × routing weight; per-32 amax → UE8M0 scale; cast FP8; TMA store into the L2 ring
  (+ scale into `l2_sf_buffer`); then mark L2 dependency bits and L1-slot empty.
- L2 epilogue ([K] lines 1256-1360): TMEM → BF16 in shared memory → each row written with 16-byte stores
  to `combine_token_buffer[topk slot][token]` on the token's source rank (`sym_buffer.map(dst, rank)`).
- Combine ([K] lines 1365-1525): grid sync; SM 0 tells peers "my writes into you are done" (grid-index
  tag); each rank, per token chunk, waits for the ranks of its selected experts, TMA-loads the top-k (+ shared)
  BF16 slots from its own buffer, accumulates in FP32 in fixed order (shared first, then top-k order),
  casts to BF16 and TMA-stores y.
- Bitwise check: [TEST] calls `deep_gemm.use_deterministic_algorithms(True)` and asserts
  `torch.equal(fused_y, baseline_y)` and equal expert-count stats; the PR #316 version of the test is
  commented "must be bitwise identical". [PR462] fixed the combine reduction order for bitwise agreement
  with the shared-expert reference. [PR432]: determinism became opt-in (default False) in 2026-09.
- Limits: SM100 only (`arch_major == 10`, kernel asserts sm_100f) ([API], [K]); SwiGLU only ([API]);
  grid uses every SM ("we must use the whole GPU", [HEUR]); FP8×FP4 only at first release and "still under
  development" ([PR304]); power throttling under full concurrency, SwiGLU exp/div stalls the GEMM pipeline
  ([V4] §3.1 p.16); grid syncs and NVLink barriers at kernel start and end ([K], [BAR]).

## Discrepancies

1. **Notion "eliminates … intermediate buffer write-backs".** Not all of them: the post-SwiGLU FP8 activation
   still goes through a global-memory ring buffer between linear 1 and linear 2 ([K] L1 epilogue TMA-stores
   into `l2_token_buffer`; [TEST]'s HBM model counts "L1 output write" and "L2 acts read"). What disappears
   is the BF16 [n, 2I] linear-1 output and its re-read by a separate SwiGLU kernel, the local BF16 linear-2
   output re-read by the combine kernel, and the kernel boundaries. Slides say "fewer and smaller
   round trips" and give the analytic 10·I → 2·I.
2. **Notion "single-stream overlap is friendlier to numerical reproducibility than dual-stream or micro-batch
   splitting".** No primary source makes this comparison. What the sources show: the fused output is bitwise
   equal to the non-fused path in the test, and the combine sums in a fixed order. Slides state only that.
3. **Notion "linear1 (FP8 x FP4 / FP8 x FP8)".** At first release only FP8×FP4 was supported ([PR304]);
   FP8×FP8 (`mma_type='fp8xfp8'`) and BF16×BF16 (`bf16_mega_moe`, [PR364]) were added later ([RD], [API]).
   Slides say FP8×FP4 or FP8×FP8 (current README).
4. **Paper vs PR on the speedup range.** [V4] gives 1.50–1.73× for general inference and up to 1.96× for
   latency-sensitive cases; [PR316]'s eight points span 1.50–1.96× (V4-Pro at 1 token/rank is 1.61×).
   Slides plot the PR table and quote "1.50–1.96×".
5. **"Waves".** [V4] Fig. 5 describes splitting experts into waves; the released kernel implements this as a
   persistent tile scheduler (linear-1 warmup waves, then interleaved L1/L2 tiles gated by counters) rather
   than explicit expert groups ([SCHED]). Slides describe the code; deck 14 covers the wave model.
