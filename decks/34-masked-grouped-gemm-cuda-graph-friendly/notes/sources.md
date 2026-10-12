# Sources — deck 34: Masked grouped GEMM (CUDA Graph friendly)

Downloads: `$SCRATCH/research/34/` (DeepEP V1 README and `buffer.py` at commit `2be0d4f8e9`, LMSYS blog under
`web/`, FlashInfer under `arxiv/`) and the DeepGEMM files shared with deck 33 in `$SCRATCH/research/33/`.
DeepGEMM `main` = `057ca5964a` (2026-09-30), DeepEP cached README = V2.5.

## Sources read

| Tag | Source | Where |
|---|---|---|
| [RD] | DeepGEMM README, current (cached) | l.82-86: masked layout for decoding "when CUDA graph is enabled and the CPU is unaware of the number of tokens each expert receives"; mask tensor; example input = DeepEP low-latency output; l.63 (SM90 NT only); news 2026.04.16 (Mega MoE) |
| [RD-0325] | DeepGEMM README @ `8002b769c0` (2025-03-25) | l.10 setup (H800 SXM5, NVCC 12.8, vs internal CUTLASS 3.6-based implementation); "Grouped GEMMs for MoE models (masked layout)" table l.47-56 |
| [OLD-API] | `deep_gemm/jit_kernels/m_grouped_gemm.py` @ `8002b769c0` | `m_grouped_gemm_fp8_fp8_bf16_nt_masked` docstring l.114-136 (`lhs [num_groups, m_max, k]`, `out [num_groups, m_max, n]`, `masked_m[i]` = actual rows of group i, `expected_m` "a value hint (which is a value on CPU)"); l.161-169: config chosen from `expected_m`, grid = `num_sms`, `m % block_m == 0` required when `num_groups > 1` |
| [API] | `csrc/apis/gemm.hpp` @ main | `m_grouped_fp8_fp4_gemm_nt_masked` l.269-318: "Shape must be `[G, M, K] @ [G, N, K].mT`", K-major A/B, `masked_m` int32 of length G, `expected_m > 0`, D BF16 |
| [SCHED] | `deep_gemm/include/deep_gemm/scheduler/gemm.cuh` @ main | `MGroupedMasked` branch of `get_next_block` l.188-204: per group `num_m_blocks = ceil(grouped_layout[g] / BLOCK_M)` read from device memory, CTAs walk groups until `kNumGroups`; `is_computation_valid` l.301-302 skips WGMMA sub-blocks at or beyond `masked_m[g]` |
| [IMPL] | `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp`, `heuristics/sm90.hpp` @ main | masked launcher l.216-240 (`expected_m` → `GemmDesc`); cost model l.202-207 uses `ceil(expected_m / block_m)` blocks to choose block sizes; launch grid = `num_sms` (l.190-199) |
| [TEST] | `tests/test_fp8_fp4.py` @ main | `test_m_grouped_gemm_masked` l.178-238: `expected_m = 1.2 × expected_m_per_group`; "TODO: when the actual m is greater than expected_m_per_group, efficiency may significantly decrease"; psum variant runs the contiguous kernel with `use_psum_layout=True, expected_m_for_psum_layout` |
| [GEN] | `tests/generators.py` @ main | `enumerate_m_grouped_masked` l.213-226 (`max_m = 4096`; groups/expected m (32,192), (6,1024), (32,20), (6,20)); `generate_m_grouped_masked` l.446-485 |
| [HEUR] | `csrc/jit_kernels/heuristics/runtime.hpp` @ main | l.57-64: on SM100 `expected_m` is ignored for the alignment, "small values may incur performance loss; use Mega MoE directly for such workloads" |
| [PR103] | DeepGEMM PR #103 | skipping useless M computation: masked GEMM +10-20 TFLOPS (notes only) |
| [PR112] | DeepGEMM PR #112 (NVIDIA, SM100 support) | "Masked grouped GEMM optimization" listed as ongoing |
| [EP1] | DeepEP README @ `2be0d4f8e9` (2025-10-21, V1 era) | l.7 hook-based overlap "does not occupy any SM resource"; l.26-37 low-latency table (H800, 128 tokens per batch, hidden 7168, top-8, FP8 dispatch / BF16 combine); l.173 normal dispatch "the CPU will wait for GPU's signal to arrive, so this is not compatible with CUDA graph"; l.249-250 low-latency mode uses "much more space", recommend `num_max_dispatch_tokens_per_rank` < 256; l.265, l.281 dispatch/combine "compatible with CUDA graph" |
| [EP1-BUF] | `deep_ep/buffer.py` @ `2be0d4f8e9` | `low_latency_dispatch` docstring l.554-600: `recv_x` = `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` FP8 + scales (`hidden // 128` FP32 or packed UE8M0), column-major scales "for TMA compatibility"; "not all tokens are valid ... as we do not synchronize CPU received count with GPU"; `recv_count` = `[num_local_experts]` int; `low_latency_combine` l.616-650: input `x` with the same 3D shape, BF16 |
| [EP2] | DeepEP README, current V2.5 (cached) | l.19 V1 fully removed in V2.5; l.163 `do_cpu_sync=False` for decoding: outputs at configured capacity, GEMMs use `handle.psum_num_recv_tokens_per_expert` |
| [LMSYS] | LMSYS blog 2025-05-05 (large-scale EP on 96 H100) | l.163 normal dispatch: shapes "incompatible with CUDA Graph" ... decode, "where kernel launch overhead becomes a significant bottleneck"; l.165 low-latency dispatch supports CUDA Graph but needs a fixed preallocation, exceeding it is a runtime error; l.167-172 mode table; l.182-188 masked kernel for decode, pairs with low-latency dispatch; l.282 decode 22,282 tokens/s per node on 9 nodes (EP72, H100), 5.2× over TP16, "largely attributed to 4× larger batch sizes enabled by EP" |
| [FI] | Ye et al., FlashInfer, MLSys 2025, arXiv 2501.01005 | §3.3 p.7: CUDA-Graph compatibility via persistent kernels whose grid size is fixed and fixed workspace offsets; App. D.1 p.19: captured arguments (pointers and scalars) are fixed, so buffers are allocated at maximum capacity |
| [MC] | Megatron-Core MoE report, arXiv 2603.07685v2 | §4.3.6 p.38-39: each kernel launch costs several µs of CPU time; CUDA Graphs need static shapes; §4.3.7 p.43: host must query per-expert counts before launching grouped GEMM; worst-case buffers can need O(EP_size)× the working set; §4.2.1 p.30: 58 MoE layers |
| [DSV3] | DeepSeek-V3 report, arXiv 2412.19437v2 | §3.4.2 p.19-20: decode EP320, one expert per GPU, 64 GPUs for redundant/shared experts, IBGDA; batch per expert "usually within 256 tokens", memory-bound; §4.2 p.22: 256 routed experts, top-8, hidden 7168 |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| EP320, 1 expert per GPU | DeepSeek-V3 decode deployment | [DSV3] §3.4.2 p.19 |
| ≤ 256 tokens per expert | decode batch per expert ("usually within 256") | [DSV3] §3.4.2 p.20 |
| several µs per launch | CPU cost of one kernel launch | [MC] §4.3.6 p.38 |
| 58 | MoE layers in DeepSeek-V3 = host count reads per decode step for a host-shaped grouped GEMM (one per MoE layer) | derived: [MC] §4.2.1 p.30 (58 layers) × one read per layer ([MC] §4.3.7 p.43; [EP1] l.173) |
| 77–194 µs | DeepEP low-latency dispatch latency, EP8 → EP256, 128 tokens per batch, H800 | [EP1] l.32-37 |
| `[G, max_m, K]`, max_m = tokens per rank × ranks | masked input layout from low-latency dispatch | [OLD-API]; [EP1-BUF] |
| < 256 | recommended `num_max_dispatch_tokens_per_rank` | [EP1] l.250 |
| 815–1233 TFLOPS | masked FP8 grouped GEMM on H800 (1×1024, 2×512, 4×256 rows; N/K = 4096/7168 and 7168/2048) | [RD-0325] masked table |
| 1.1–1.2× | masked GEMM over internal CUTLASS 3.6-based implementation | [RD-0325] masked table |
| 2064 GB/s | memory bandwidth of the 4 × 256, N=4096 K=7168 case | [RD-0325] masked table |
| 22,282 tokens/s per node | SGLang decode, 9 × 8 H100, EP72, low-latency dispatch + masked GEMM in the stack (system result) | [LMSYS] l.282, l.306 |
| 32× | worst-case slot ÷ balanced load per expert = (T·R) ÷ (T·R·k/E) = E/k = 256/8 for DeepSeek-V3 | derived from [EP1-BUF] slot shape and [DSV3] §4.2 (E = 256, k = 8) |
| 128 rows | BLOCK_M in the running example; `max_m` must be a multiple of BLOCK_M when G > 1 | [OLD-API] l.167-168 |

Running-example values in diagrams (masked_m = 300, 60, 200 with max_m = 512, BLOCK_M = 128) are illustrative.

## Mechanism facts (no numbers) used on slides

- Masked layout targets decoding with CUDA Graphs when the CPU does not know per-expert counts; the kernel computes
  only valid parts; DeepEP low-latency output is the intended input ([RD] l.84-86).
- Shapes: A `[G, max_m, K]`, B `[G, N, K]`, D `[G, max_m, N]`, `masked_m[G]` int32 on device; `expected_m` is a host
  int used only to pick the kernel configuration ([OLD-API]; [API]; [IMPL]).
- The persistent kernel is launched on `num_sms` CTAs; each CTA walks the groups, computing
  `ceil(masked_m[g] / BLOCK_M)` M-tiles per group from device memory, so the launch is identical every step
  ([SCHED] l.188-204; [OLD-API] l.161-169). Rows at or beyond `masked_m[g]` are not computed at sub-block
  granularity and D rows there are not valid ([SCHED] l.301-302; [TEST] compares only `valid_mask` rows).
- DeepEP low-latency dispatch returns `recv_x` in exactly that 3D layout plus `recv_count`; low-latency combine
  consumes a BF16 tensor of the same 3D shape; both are CUDA-Graph compatible; the receive hook overlaps RDMA
  traffic without SMs ([EP1]; [EP1-BUF]).
- A CUDA Graph fixes kernel arguments: pointers, scalars, grid size; capacity-sized buffers and persistent
  kernels are the standard way to be graph-safe ([FI] §3.3, App. D.1).

## Discrepancies

1. **Notion "removes the per-MoE-layer host sync".** Correct in spirit, but the sync is removed by the pair
   (low-latency dispatch + masked GEMM); the GEMM alone only removes the need for host-side shapes. Normal dispatch
   still blocks the CPU ([EP1] l.173). Slides credit the pair.
2. **Notion "it consumes DeepEP low-latency kernel output directly".** True for DeepEP V1's `low_latency_dispatch`
   ([EP1-BUF]); DeepEP V2.5 removed the V1 API ([EP2] l.19) and its decode path (`do_cpu_sync=False`) emits an
   expanded 2D layout with GPU-side prefix sums, which DeepGEMM consumes through the contiguous kernel's psum layout
   with `expected_m_for_psum_layout` ([EP2] l.163; [TEST] l.203-205). Slides present V1 masked as the original
   pairing and the psum layout as its successor.
3. **"rollout".** The primary sources describe inference decoding only; RL rollout is decoding, so it applies, but
   no source measures it. Slides do not claim rollout numbers.
4. **No measured host-sync cost.** None of the sources reports the latency saved by removing the sync; the 58 on
   the pain slide is a derived count and is labelled as such.
5. **Performance numbers are from March 2025 (H800)**, the last published masked table ([RD-0325]).
