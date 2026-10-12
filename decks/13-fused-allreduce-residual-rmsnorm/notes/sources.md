# Sources — deck 13: Fused AllReduce + residual + RMSNorm (HPC-Ops)

All repository files fetched 2026-10-13 from `github.com/Tencent/hpc-ops` branch `main` (raw.githubusercontent.com /
`gh api`), copies in `$SCRATCH/research/13/`.

## Sources read

| Tag | Source | Where |
|---|---|---|
| [RD] | HPC-Ops README (main; cached `research/cache/gh-hpc-ops-README.md`) | "Fused AllReduce + RMSNorm" update (June 2026); Operator Catalog row "Communication"; Performance table row "Fused AllReduce + Residual + RMSNorm"; "Why HPC-Ops?" (H20, production) |
| [PR48] | PR #48 "add fuse ar rms op" (merged 2026-06-09), description | two modes, "two-shot (reduce-scatter + all-gather) schedule", HT = single kernel over NVSwitch multicast for large token counts, LL = Lamport P2P split into two kernels overlapped via PDL for small token counts, LL needs power-of-two world size |
| [BR] | `benchmark/fuse_allreduce_rmsorm/README.md` | operator definition, single 8-GPU SM90/H20 node, BF16, default hidden 7168, supported 4096/5120/7168, token sweep `8 32 128 512 4096 8192 16384 32768`, CUDA-graph replay + best-round median, output fields `hpc_best_speedup = baseline_best_us / hpc_best_us` |
| [BS] | `benchmark/fuse_allreduce_rmsorm/benchmark_fuse_allreduce_rmsnorm.py` | docstring (4 providers); `NcclRunner.step` = `dist.all_reduce` + `flashinfer.norm.fused_add_rmsnorm`; `FlashInferRunner` (trtllm single fused kernel, mnnvl two kernels with PDL); `HpcOpsHighThroughput` (per-rank `[start:end)` slice); `HpcOpsLowLatency` (workspace `M_pad`); `fi_max_tokens` comments (trtllm Lamport buffer clamp; mnnvl "solid up to 8192" on sm90/H20) |
| [HT] | `src/allreduce/fuse_allreduce_rmsnorm_high_throughput.cu` | `fused_allreduce_rmsnorm_kernel`: signal handshake at entry/exit; per token `multi_load_reduce_add` (multimem) → `+ residual` → sum of squares, warp+block reduce, `rsqrtf` → `× weight` → `multi_store` |
| [UT] | `src/utils/utils.cuh` | `multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2`, `multimem.st.relaxed.sys.global.v4.f32`, `put_signal_*`/`wait_signal_*` |
| [LL] | `src/allreduce/fuse_allreduce_rmsnorm_low_latency.cu` | `twoshotAllreduceKernel` (destRank = token % WorldSize; P2P store to owner's scatter slot; owner spins on `isNegZero`; sum; store through `mcastPtr`; `cudaTriggerProgrammaticLaunchCompletion` right after buffer setup; `clearDirtyLamportBuf`); `rmsNormLamport` (cp.async prefetch of residual and gamma, poll broadcast buffer, add + RMSNorm with block/cluster reduce, `cudaGridDependencySynchronize` only before `waitAndUpdate`); launch with `cudaLaunchAttributeProgrammaticStreamSerialization`; world-size switch 2/4/8/16/32/64 |
| [LLH] | `src/allreduce/fuse_allreduce_rmsnorm_low_latency.h` | `MNNVLTwoShotStage {SCATTER, BROADCAST}`; `negZero`/`isNegZero`; `LamportBufferLayout::sNumLamportBuffers = 3`; `LamportFlags::waitAndUpdate` rotates current→dirty |
| [EN] | `src/allreduce/entry.cc` | TORCH_CHECKs: BF16 only, `hidden_size == 4096 || 5120 || 7168` (HT), 16-byte alignment; LL: `2 <= world_size <= 64` |
| [API] | `hpc/allreduce.py`, `hpc/multicast_handle.py`, `src/communicator/multicast_object_manager.cc` | `empty_multimem` symmetric buffers "over single-node NVLink", "No NVSHMEM dependency"; `cuMulticastCreate` / POSIX-FD handle export |
| [TST] | `tests/test_fuse_allreduce_rmsnorm_high_throughput.py`, `tests/test_fuse_allreduce_rmsnorm_low_latency.py` | per-rank slice `start = N_pad // world_size * rank`; LL workspace `M_pad = 2 * ceil(N / world_size) * world_size * NUM_LAMPORT_BUFFERS (=3)`, buffer filled with `0x80000000`; world sizes 4 and 8 tested; comment "twoshotAllreduceKernel only supports power-of-two world sizes {2,4,8,16,32,64}" |
| [VLLM] | vLLM docs v0.19.1, "Fusion torch.compile passes", https://docs.vllm.ai/en/v0.19.1/design/fusions/ | Quick-reference table row "AllReduce + RMSNorm": E2E speedup "5-20%", num_tokens "Low"; detail section: "only profitable for small num_tokens, so the fusion is only performed in the lower compiled range"; FlashInfer / TRT-LLM kernel |
| [MEG] | Shoeybi et al., Megatron-LM, arXiv 1909.08053 | §3 p.4: "two all-reduces in the forward path and two in the backward path" per transformer layer |
| [LMSYS] | LMSYS blog "HPC-Ops × SGLang", 2026-08-07, https://www.lmsys.org/blog/2026-08-07-hpc-ops-sglang/ | integration table lists Attention, Router GEMM, MoE only (used for "not yet in SGLang") |
| [FLUX] | Flux, arXiv 2406.06858 (Notion body) | read for context (TP communication is a substantial share of inference time, Fig. 1); no number used |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 8×H20, one NVLink node | benchmark / target platform | [BR] "single 8-GPU SM90/H20 node with NVLink/NVSwitch"; [RD] "tailored for NVIDIA H20" |
| 2 AllReduces per layer | forward-pass all-reduces per transformer layer under Megatron TP | [MEG] §3 p.4 |
| BF16; hidden 4096 / 5120 / 7168 | supported dtype and shapes | [EN]; [BR]; [RD] Performance table |
| 5–20% | indicative E2E speedup vLLM lists for its AllReduce + RMSNorm fusion, small num_tokens | [VLLM] quick-reference table |
| 8,192 tokens | largest token count the benchmark runs FlashInfer's mnnvl backend at on sm90/H20 | [BS] `fi_max_tokens = 8192` comment |
| 112 KiB, 448 MiB | message size at 8 and 32,768 tokens, hidden 7168, BF16 | derived (below) from [BR] token sweep |
| 4 GPUs × 4 tokens | figure example | illustrative; tests use world sizes 4 and 8 [TST] |
| 3 rotating buffers | Lamport buffer count | [LLH] `sNumLamportBuffers = 3`; [TST] `NUM_LAMPORT_BUFFERS = 3` |
| ≥ 6, ≥ 10, 2.25 hidden vectors; ≥ 2.7× | analytic HBM traffic per token per GPU | derived (below) from [BS], [HT], [LL] |
| 1.76× | max speedup of best HPC-Ops mode over best of NCCL / FlashInfer | [RD] Performance table "Up to 1.76x"; metric definition [BR] |
| 2–64, power of two | LL world sizes | [LL] switch cases; [EN] check; [TST] comment |
| 6× the activation | LL workspace size | derived (below) from [TST] / [BS] |

## Derived numbers

1. **Message sizes.** hidden 7168 × 2 bytes (BF16) = 14,336 B per token. 8 tokens → 114,688 B = 112 KiB.
   32,768 tokens → 469,762,048 B = 448 MiB. Ratio 32,768 / 8 = 4,096× (speaker notes only).
2. **HBM traffic per token per GPU** (unit = one hidden vector, 2H bytes; P = 8). Counted from code, steady state:
   - NCCL + `fused_add_rmsnorm` ([BS] `NcclRunner.step`): all-reduce reads its input (≥1) and writes the reduced
     tensor in place (1); the norm reads it back (1), reads residual (1), writes residual (1), writes the normalized
     output (1). Total **≥ 6**; NCCL-internal staging is not counted, so this is a lower bound.
   - HT ([HT], [TST] slicing): each GPU's input row is read once by some owner's `multimem.ld_reduce` (1); the
     `multimem.st` of every owner lands in every GPU's output buffer (1); residual is read and written only for the
     owned 1/P slice (2/P = 0.25). Total **2 + 2/P = 2.25**. Weight vector reads (L2-resident) ignored.
   - LL ([LL], [LLH]): kernel 1 reads the local input (1); P2P scatter stores landing in this GPU's slots (each GPU
     owns N/P tokens × P senders = 1 per token); owner reads its slots (1); clear of the dirty scatter buffer
     (`round_up(N,P)·dim` bytes = 1); multicast broadcast landing (1); clear of the dirty broadcast buffer (1);
     kernel 2 reads the broadcast buffer (1) and residual (1), writes residual (1) and output (1). Total **≥ 10**
     (spin re-reads not counted).
   - Ratio 6 / 2.25 = 2.67 → "≥ 2.7×" (the baseline count is a lower bound).
3. **LL workspace.** `M_pad = 2 · ceil(N/P) · P · 3` rows of `hidden` BF16 elements ([TST], [BS]) ≈ 6·N rows, i.e.
   6× the activation tensor (2 stages × 3 Lamport buffers).
4. **"Norm on 1/8 of the tokens"** (notes, slide 9 notes): HT rank r processes rows `[N_pad/P · r, N_pad/P · (r+1))`
   ([TST], [BS]), P = 8.

## Mechanism facts (no numbers) used on slides

- HT: one kernel; per-block put/wait signal handshake with every peer at entry and exit; per token
  `multimem.ld_reduce ... add.acc::f32` on the multicast address returns the cross-GPU sum; add residual; sum of
  squares → warp reduce → shared-memory block reduce → `rsqrtf(sum/H + eps)`; multiply by weight; `multimem.st` to
  the multicast output address ([HT], [UT]).
- LL kernel 1: token t owned by rank `t % P`; each rank writes its whole token-t vector into the owner's scatter
  slot `[t/P][rank]` with a plain store to the peer pointer (P2P over NVLink); senders replace −0.0 by +0.0; slots
  are pre-filled with −0.0 (`0x80000000` / `getPackedLamportInit`); owner spins with `ld.volatile` until no −0
  remains, sums in FP32, stores the sum through the multicast pointer into every rank's broadcast buffer ([LL], [LLH]).
- LL kernel 2: launched with programmatic stream serialization (PDL); kernel 1 calls
  `cudaTriggerProgrammaticLaunchCompletion` right after buffer setup; kernel 2 issues cp.async copies of residual
  and gamma into shared memory, polls the broadcast buffer for −0, adds residual, normalizes (block + cluster
  reduce), writes residual-out and output, and calls `cudaGridDependencySynchronize` only before rotating the
  Lamport flags ([LL]).
- Lamport buffers: 3 buffers × 2 stages; `waitAndUpdate` makes current → dirty and advances current; dirty buffer
  is cleared by the next call ([LLH]).
- Both modes are "two-shot (reduce-scatter + all-gather)" ([PR48]); HT for large token counts (prefill), LL for
  small (decode) ([PR48]); the benchmark reports `hpc_best = min(ht, ll)` — there is no automatic mode switch in
  the Python API ([BR], [API]).
- Buffers come from `hpc.empty_multimem` (symmetric, multicast-mapped, single-node NVLink, no NVSHMEM) ([API]).
- SGLang integration (Aug 2026) covers Attention, Router GEMM, MoE ([LMSYS] integration table).

## Discrepancies

1. **Notion body cites the LMSYS HPC-Ops × SGLang blog as "vs NCCL / FlashInfer".** The blog covers only
   Attention, Router GEMM and MoE; it has no AllReduce + RMSNorm result. The 1.76× figure comes from the HPC-Ops
   README. Slides cite the README.
2. **Notion calls the LL mode "Lamport P2P".** In the code only the scatter (reduce) phase is P2P; the broadcast
   phase stores through the multicast address ([LL]). Slides show both.
3. **Notion: "fuses ... into an NVLink-native kernel".** HT is one kernel; LL is two kernels chained by PDL
   ([PR48], [LL]). Slides say "one op", "one kernel" (HT) and "two kernels" (LL).
4. Minor: the benchmark README command names `bench_allreduce_rmsnorm.py`; the script is
   `benchmark_fuse_allreduce_rmsnorm.py` (open PR #77). No effect on slides.
