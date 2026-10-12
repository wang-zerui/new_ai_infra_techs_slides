# Sources — deck 31: Fused FP8 MoE low-latency pipeline (HPC-Ops)

Repository files fetched 2026-10-13 from `github.com/Tencent/hpc-ops` (main); copies in `$SCRATCH/research/31/`.

## Sources read

| Tag | Source | Where |
|---|---|---|
| [RD] | HPC-Ops README (main; cached `research/cache/gh-hpc-ops-README.md`) | "Fused MoE" update (June 2026): fuses routing, Gate-Up GEMM, activation quantization, Down GEMM and top-k weighted reduction "into one pipelined execution path"; gather-free Gate-Up; shared-memory counting; "The cp.async GEMM path removes Warp Specialization in the low-latency regime"; PDL chaining. Performance row "Fused FP8 MoE · DeepSeek-V3, Hunyuan-V3, Qwen3-235B under TP=8 EP=1, TP=1 EP=8 · vLLM CUTLASS, vLLM Triton, SGLang · Up to 1.6x TP, 1.5x EP". Roadmap "Megakernel" |
| [RD-JAN] | HPC-Ops README @ `bea9bbc2cb` (2026-01-26) | FusedMoE (fp8) vs TensorRT-LLM, vLLM: max 1.49x prefill, 1.14x decode (context; not on slides) |
| [PR49] | PR #49 "add fused fp8 moe kernel for low-latency llm inference" (merged 2026-06-09), description | gather-then-GEMM critique (TMA + warp specialization, extra gather traffic); shared-memory counting with contiguous per-expert ranges; Gate-Up reads original input through routing indices; GEMM path without warp specialization, higher CTA residency; five stages chained with PDL; H20, CUDA 13, PyTorch 2.11; "Relative to the median of the compared backends: TP=8 EP=1 about 1.5x to 1.6x; TP=1 EP=8 about 1.2x to 1.5x" |
| [LMSYS] | LMSYS blog "HPC-Ops × SGLang", 2026-08-07, https://www.lmsys.org/blog/2026-08-07-hpc-ops-sglang/ | MoE section (six-stage conventional chain; overhead rivals the expert GEMMs; fused pipeline description); Table 4 (Hy3 TP8/EP1, H20) and Table 5 (Hy3 TP1/EP8, H20) with "Speedup vs. best"; Highlights (mean 1.08× TP8/EP1, 1.21× TP1/EP8; H200 up to 4.21× over Triton at eight tokens); Table 6 (H200 Qwen3 MoE kernel 0.89–4.21× vs SGLang Triton fused experts); Table 7 (Hy3-FP8 TPOT, 8×H20, TP8, FP8 KV cache, 8K in / 4K out); Table 11 (H200 serving MoE: Qwen3 parity to +2.7%, Hy3 −4.2% to +6.3%) |
| [BR] | `benchmark/fused_moe/README.md` | model presets (hunyuan-v3: 192 experts, top-8, hidden 4096, intermediate 1536; qwen3-235b 128/8/4096/1536; deepseek-v3 256/8/7168/2048); modes (`tp8_ep1`: intermediate_per_rank = intermediate / TP; `tp1_ep8`: experts_per_rank = experts / EP); `avg/group = bs * topk / experts_per_rank`; nsys + CUDA Graph median timing |
| [BB] | `benchmark/fused_moe/backends/hpcops.py`, `base.py` | HPC-Ops provider calls `hpc.fuse_moe` (per-tensor FP8, per-expert weight scales, `use_bf16_mul=True`); w1 [E, 2N, K]; uniform routing (`randperm` top-k per token) |
| [EN] | `src/fuse_moe/entry.cc` | `fuse_moe_entry` dispatches to `fuse_moe_cp_async_entry` when `intermediate_size/2 <= 512`, `(intermediate_size/2) % 64 == 0`, `hidden % 64 == 0` (intermediate_size = gate+up = 2N); cp.async checks: `num_topk <= 128`, `num_expert_local <= 512`; default path allocates gathered `gate_up_input [num_seq·num_topk, hidden]` |
| [CPF] | `src/fuse_moe/sm90/cp_async/fuse_moe.cu` | five stages: `count_and_build_indices_async` → `group_gemm_fp8_scatter_async` (Gate-Up, `row_indices`, PDL) → `act_mul_and_quant_async` → `group_gemm_fp8_multistage_async` (Down, PDL) → `reduce_async` (PDL, shared_output) |
| [CNT] | `src/fuse_moe/sm90/cp_async/count.cu` | `count_seq_kernel` (shared histogram + one global atomicAdd per expert per block); `build_indices_kernel` (shared counters, one global reservation per expert per block, `row_indices[pos] = token`, `topk_pos[token·topk + j] = pos`, task maps, sentinel tail); `count_and_build_kernel` single block when `num_seq <= 128`; `kUsePDL = true` |
| [DEF] | `src/fuse_moe/sm90/count_and_gather.cu`, `fuse_moe.cu` | default path: `gather_kernel` uses a global `atomicAdd` per (token, expert) pair and writes the gathered copy; then TMA warp-specialized grouped GEMMs (deck 29), act + quant, reduce; `use_pdl = true` |
| [SCAT] | `src/group_gemm/sm90/cp_async/group_gemm_fp8_scatter.cu`, `group_gemm_fp8.cu`, `config.h`, `common.cuh` | 128 threads per CTA (one warpgroup loads with cp.async and runs WGMMA), `__launch_bounds__(128, 5)` (7 for TileM 48), kStage 2 (scatter), kTileN 64, TileM 8–64 via `SM90_64x{8..64}x32` (swapAB), `shm_row_indices` loaded per tile, `cp.async.cg.shared.global` per row chunk; grid = SM count × `grid_multiplier` (8, 9, 8, 7, 5 for TileM 8, 16, 32, 48, 64); `cudaGridDependencySynchronize` at start, `cudaTriggerProgrammaticLaunchCompletion` at end |
| [WS] | `src/group_gemm/sm90/kernels.cuh`, `group_gemm_pertensor_fp8.cu` | TMA path: `__launch_bounds__(384, 1)`, grid = SM count, producer + 2 math warpgroups, 8 stages |
| [ACT] | `src/activation/activation.cu`, `src/fuse_moe/sm90/reduce.cu` | act_mul_and_quant and reduce launched with programmatic stream serialization; reduce adds `shared_output` |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 192 experts, top-8, hidden 4096 (intermediate 1536) | Hunyuan-V3 benchmark preset | [BR] |
| 0.67 tokens per expert | TP8, batch 16 | derived: 16 × 8 / 192 (formula from [BR]) |
| 8× (pain slide), "x × 8 copy" | gathered buffer holds num_seq × top-k rows (top-8) | [EN] / [DEF] buffer shape |
| Six kernels, five launch gaps | conventional gather-then-GEMM chain (6 kernels → 5 boundaries) | [LMSYS] MoE section (route, gather, Gate-Up, act + quant, Down, top-k reduce) |
| 5 stages (6–7 kernels) | cp.async pipeline | [CPF], [CNT] |
| ≤ 512, multiple of 64 | cp.async dispatch condition (per-rank intermediate) | [EN] |
| 8-row tile; 2 stages; 128 threads; ≥ 5 CTAs per SM; 5–9 CTAs per SM grid | cp.async GEMM design | [SCAT] |
| 384 threads; 1 CTA per SM; 8 stages | TMA warp-specialized design | [WS] |
| 128 tokens | single-block count path threshold (notes) | [CNT] |
| 1.08× / 1.21× | mean per-batch speedup over the best baseline, Hy3, TP8/EP1 and TP1/EP8, H20 | [LMSYS] Highlights + MoE text |
| 1.18× (TP8 max, batch 512); 1.60× (EP8 max, batch 64); 0.98× (EP8, batch 512); per-batch values in chart | per-row "Speedup vs. best" | [LMSYS] Tables 4 and 5 |
| 1.5–1.6× (TP8, notes) | vs the median baseline | [PR49] |
| 3.3%, 15.1%–48.8%, 41.5%; TPOT ms values in chart | Hy3-FP8 TPOT, HPC-Ops attention + MoE vs SGLang default | [LMSYS] Table 7 |
| 8×H20, TP8, FP8 KV cache, 8K in / 4K out | E2E setup | [LMSYS] Table 7 text |
| 0.89–4.21× | H200 Qwen3 FP8 MoE kernel vs SGLang Triton fused experts, 1–4,096 tokens | [LMSYS] Table 6 |
| −4.2% to +6.3% | H200 Hy3 serving output throughput, HPC-Ops MoE vs SGLang default | [LMSYS] Table 11 |

## Derived numbers

1. **0.67 tokens per expert**: avg/group = bs × topk / experts_per_rank = 16 × 8 / 192 = 0.667 (TP8: all 192 experts
   local). For comparison (notes only): TP1/EP8 → 24 local experts → 16 × 8 / 24 = 5.3.
2. **8 copies**: the gathered Gate-Up input of the gather-then-GEMM design has num_seq × top-k rows of `hidden`
   ([EN] default path `gate_up_input`), i.e. 8× the input rows for top-8.
3. **Path attribution of the LMSYS MoE tables**: TP8/EP1 → intermediate_per_rank = 1536 / 8 = 192 ≤ 512 and a
   multiple of 64 → cp.async path; TP1/EP8 → 1536 > 512 → TMA path ([BR] modes + [EN] condition; the benchmark's
   HPC-Ops provider calls `hpc.fuse_moe` [BB]).
4. fig_layer: one sample of uniform top-8 routing (numpy seed 7): 107 experts get 0 tokens, 52 get 1, 26 get 2,
   4 get 3, 3 get 4. Illustration only; no number from it appears on a slide.

## Mechanism facts (no numbers) used on slides

- Count/index: per-block shared histogram; one global atomic per expert per block reserves a contiguous range;
  `row_indices[slot] = token`, `topk_pos[token][k] = slot`; task maps for both GEMMs ([CNT]).
- Gate-Up: loads its tile's row_indices into shared memory and fetches x rows with cp.async — no gathered copy
  ([SCAT]); swapped operands as in deck 29 ([SCAT] config).
- One warpgroup does loads and WGMMA, no producer/consumer split; many resident CTAs ([SCAT]; [PR49] wording).
- PDL on all five stages ([CNT], [SCAT], [ACT]).
- Intermediates (Gate-Up output BF16, requantized FP8, Down output BF16) are HBM tensors ([EN] allocations).

## Discrepancies

1. **Notion: "Fuses ... into a single pipeline".** It is five PDL-chained *stages* with HBM intermediates, not one
   kernel: the count/index stage is itself 2 kernels (≤ 128 tokens: `count_and_build_kernel` + task-map build) or
   3 kernels (> 128 tokens), so 6–7 kernels in total ([CPF], [CNT]). The README says "one pipelined execution
   path". The gain is not fewer kernels but no gather copy, no per-pair global atomics, overlapped launches and
   higher occupancy. Slides say "five stages, chained with PDL".
2. **Notion: "Up to 1.6x (TP) and 1.5x (EP) over vLLM CUTLASS, Triton and SGLang".** PR #49 states these as
   speedups relative to the *median* of the compared backends (TP8 1.5–1.6×, EP8 1.2–1.5×). Against the *best*
   baseline per batch (LMSYS Tables 4–5, Hy3) the maxima are 1.18× (TP8) and 1.60× (EP8), means 1.08× and 1.21×.
   Slides use the best-baseline numbers.
3. **The low-latency (cp.async, gather-free, no warp specialization) path is only taken when the per-rank
   intermediate is ≤ 512** ([EN]). In the EP8 benchmark mode (intermediate 1536) the TMA warp-specialized path with
   a gather runs. Notion attributes all results to the low-latency mechanisms; slides label the EP8 curve "TMA path".
4. **Benchmark README says "Operator: SM90 blockwise FP8 FusedMoE"**, but its HPC-Ops provider calls the per-tensor
   `hpc.fuse_moe` with per-expert scales ([BB]). Slides describe the per-tensor path.
5. Notion body cites MegaBlocks and FlashInfer papers; neither is used for numbers.
