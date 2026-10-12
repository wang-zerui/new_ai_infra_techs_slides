# Sources — deck 33: M-axis grouped GEMM (contiguous layout)

All downloads live in `$SCRATCH/research/33/` (READMEs at old commits, source files under `src/`, blogs under
`web/`, MegaBlocks under `arxiv/`). DeepGEMM `main` was at commit `057ca5964a` (2026-09-30, v2.8.1) when read.

## Sources read

| Tag | Source | Where |
|---|---|---|
| [RD] | DeepGEMM README, current (cached `gh-deepgemm-README.md`, = main @057ca596) | l.63 (NT naming, SM90 NT only), l.76-80 (contiguous layout, alignment, K-grouped pointer), l.82-86 (masked), l.154-155 (alignment utilities) |
| [RD-0325] | DeepGEMM README @ commit `8002b769c0` (2025-03-25), "Performance" section | dense table (l.16-35) and "Grouped GEMMs for MoE models (contiguous layout)" table (l.38-45); setup sentence l.10: H800 SXM5, NVCC 12.8, speedups vs "internally and carefully optimized implementation based on CUTLASS 3.6"; l.107 "each expert segment must be aligned to the GEMM M block size" |
| [OLD-API] | `deep_gemm/jit_kernels/m_grouped_gemm.py` @ `8002b769c0` | docstring of `m_grouped_gemm_fp8_fp8_bf16_nt_contiguous` (l.36-58): shapes `[m_sum, k]`, `[num_groups, n, k]`, `[m_sum, n]`, `m_indices[m_sum]`, "values of m_indices in every m-alignment block must be the same", alignment 128; launch (l.86-111): config from actual `m`, grid = `num_sms`, N/K/BLOCK sizes/NUM_GROUPS compiled in |
| [API] | `csrc/apis/gemm.hpp` @ main | `m_grouped_fp8_fp4_gemm_nt_contiguous` (l.181-250): "Shape must be `[M, K] @ [G, N, K].mT`", A K-major, `grouped_layout` int32 of length M (plain) or G (psum), D BF16 N-major; registrations l.870-888 (`use_psum_layout=False`, `ensure_zero_padding=True`, `expected_m_for_psum_layout`) and aliases l.926-928 (`m_grouped_fp8_gemm_nt_contiguous`) |
| [SCHED] | `deep_gemm/include/deep_gemm/scheduler/gemm.cuh` @ main | `MGroupedContiguous`: `num_blocks = num_m_blocks * num_n_blocks` (l.92-94); group offset from `grouped_layout[m_block_idx * BLOCK_M]` (l.151-153); psum variant walks groups with `last_psum_m = align(current_psum_m, BLOCK_M)` (l.205-225); TMA multicast only if `grouped_layout` of the peer M block equals this one (l.275-291); padding blocks skipped when `grouped_layout[...] < 0` (l.296-300); swizzle groups of 8/16 blocks for L2 (l.14-26, 108-144) |
| [HEUR] | `csrc/jit_kernels/heuristics/runtime.hpp` @ main | `kLegacyMKAlignmentForContiguousLayout = 128` (l.12); theoretical alignment 256 on SM100 "friendlier to MoE cast, M-grouped, and K-grouped operators" (l.57-64) |
| [GEN] | `tests/generators.py` @ main | `enumerate_m_grouped_contiguous` (l.194-210): groups (4, 8192) and (8, 4096); `generate_m_grouped_contiguous` (l.378-419): actual M = expected × U(0.7, 1.3), aligned to the alignment, padding rows set to −1 (plain) and zeroed; psum stores the actual end per group |
| [TEST] | `tests/test_fp8_fp4.py` @ main | `test_m_grouped_gemm_contiguous` (l.114-175) |
| [PR103] | DeepGEMM PR #103 "Grouped GEMM skip useless computation for unaligned Ms" (2025-05-27) | body: skipping padding computation |
| [PR74] | DeepGEMM PR #74 (2025-04) | contiguous grouped table with 128×160 tiles: 1381 / 1262 / 1383 / 1259 TF (not used on slides) |
| [PR304] | DeepGEMM PR #304 "[Public release 26/04]" | "DeepEPv2 MoE GEMM layout", "much faster MoE GEMM" |
| [EP2] | DeepEP README, current (cached `gh-deepep-README.md`) | l.17 (zero padding between experts), l.30 (expanded layout for grouped expert GEMMs), l.161-163 (`expert_alignment` = `deep_gemm.get_mk_alignment_for_contiguous_layout()`, `do_cpu_sync=True` for training/prefill gives exact sizes and CPU expert counts; `False` for decoding uses `handle.psum_num_recv_tokens_per_expert`), l.198-200 (`do_expand`, `do_zero_padding`) |
| [EP1] | DeepEP README @ `2be0d4f8e9` (2025-10-21) | l.167-180: normal dispatch "the CPU will wait for GPU's signal to arrive, so this is not compatible with CUDA graph" |
| [LMSYS] | LMSYS blog "Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs" (2025-05-05), https://www.lmsys.org/blog/2025-05-05-large-scale-ep/ | "DeepGEMM Integration": contiguous kernel for prefill; normal-dispatch output needs a permutation into the contiguous format (custom Triton kernel); masked kernel pairs with low-latency dispatch |
| [MB] | Gale et al., "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts", MLSys 2023, arXiv 2211.15841 | abstract p.1; §3 p.4 (token dropping vs capacity factor: "MoE-layer math operations increased by over 2× in order to avoid dropping tokens"; capacity factors "as high as 11" citing Tutel); §5.2 p.7 (pads each expert's tokens to a multiple of 128); Fig. 7 p.8 (1.38×/2.0×/4.35× over Tutel's padding approach) |
| [DSV3] | DeepSeek-V3 Technical Report, arXiv 2412.19437v2 | §3.3.2 p.16 (1×128 activation tiles, 128×128 weight blocks); §3.4.1 p.19 (prefill EP32, 8 experts + 1 redundant per GPU); §3.4.2 p.20 (decode batch per expert usually within 256 tokens); §4.2 p.22 (61 layers, hidden 7168, all FFNs except first three are MoE, 1 shared + 256 routed experts, expert intermediate 2048, top-8) |
| [MC] | Megatron-Core MoE report, arXiv 2603.07685v2 | §4.2.1 p.30 (58 MoE layers in DeepSeek-V3); §4.3.2 p.36 (grouped GEMM vs sequential GEMMs: overlaps the wave tail; CUTLASS grouped GEMM needs per-precision/shape development and tuning, TE has BF16 on Hopper only); §4.3.7 p.43-44 (host-initiated grouped GEMMs need a CPU list of per-expert token counts) |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 58 MoE layers | DeepSeek-V3 MoE layer count (61 layers, first three dense) | [DSV3] §4.2 p.22; [MC] §4.2.1 p.30 |
| 256 routed experts, top-8 | DeepSeek-V3 routing | [DSV3] §4.2 p.22 |
| hidden 7168, expert width 2048 | DeepSeek-V3 dims | [DSV3] §4.2 p.22 |
| N = 4096, K = 7168 (gate+up) and N = 7168, K = 2048 (down) | per-expert GEMM shapes; 4096 = 2 × 2048 fused gate/up (derived from [DSV3] dims); the same shapes are the README benchmark shapes | [RD-0325] tables; derivation here |
| EP32, 8 + 1 experts per GPU | DeepSeek-V3 prefill deployment | [DSV3] §3.4.1 p.19 |
| > 2× | MoE-layer math increase needed to avoid dropping tokens with capacity-factor batching | [MB] §3 p.4 |
| capacity factor up to 11 | required by some MoEs to avoid dropping (Tutel, as cited by MegaBlocks) | [MB] §3 p.4 |
| 128 | M alignment of the contiguous layout (`get_mk_alignment_for_contiguous_layout()` default; legacy constant) | [HEUR] l.12; [OLD-API] docstring l.45-46 ("aligned to `get_m_alignment_for_contiguous_layout()` (128)") |
| 256 | theoretical alignment on SM100 (UMMA_N = 256) | [HEUR] l.57-64 |
| 127 / 255 rows | worst-case padding per expert = alignment − 1 | derived from 128 / 256 |
| 3.1% | worst-case padding at 4096 tokens per expert, alignment 128: 127 / 4096 = 3.10% | derived (analytic) |
| 1346 TFLOPS, 1214 TFLOPS | contiguous grouped FP8 GEMM, (4 × 8192 or 8 × 4096 rows), N=4096 K=7168 and N=7168 K=2048, H800 | [RD-0325] contiguous table |
| 1.2–1.3× | speedup of the contiguous grouped GEMM over the internal CUTLASS 3.6-based implementation | [RD-0325] contiguous table |
| 1265 TFLOPS, 1168 TFLOPS | dense FP8 GEMM, M=4096, (N=4096, K=7168) and (N=7168, K=2048), same README | [RD-0325] dense table |
| 1.06×, 1.04× | grouped ÷ dense at the same per-expert shape: 1346/1265 = 1.064, 1214/1168 = 1.039 | derived |
| 4096, 8192 tokens per expert | README benchmark "M per group"; test `expected_m_per_group` | [RD-0325]; [GEN] l.196 |
| ≤ 256 tokens per expert | DeepSeek-V3 decode batch per expert ("usually within 256 tokens") | [DSV3] §3.4.2 p.20 |
| 1.38×, 2.0×, 4.35× | MegaBlocks dropless over Tutel's padding approach (notes only) | [MB] Fig. 7 p.8 |

Analytic chart `charts/padding.py`: worst-case padding overhead = (alignment − 1) / M for alignment 128 and
256, M = tokens per expert ∈ {256, 512, 1024, 2048, 4096, 8192}. Values: align 128 → 49.6, 24.8, 12.4, 6.2,
3.1, 1.6 %; align 256 → 99.6, 49.8, 24.9, 12.5, 6.2, 3.1 %. Labelled "analytic, worst case" on the slide.

Running-example token counts in the diagrams (E0 = 300, E1 = 60 (overview) or 70 (alignment figure), E2 = 200
or 142 rows, BLOCK_M = 128) are illustrative instance labels for the mechanism, not measurements. Visit-order numbers
in fig_sched follow the scheduler rule (groups of 8/16 M blocks walked down M first, [SCHED] l.108-144) for a 6 × 4
tile grid.

## Mechanism facts (no numbers) used on slides

- Grouping is on M only; N and K are fixed because MoE experts share a shape; tokens of all experts are
  concatenated ("contiguous"); used for training forward and inference prefill ([RD] l.78).
- Plain layout: `m_indices[i]` = expert of row i; every alignment block must carry one value; padding rows
  hold −1 and are zero ([OLD-API]; [GEN] l.396-406).
- The scheduler computes one tile grid of ceil(ΣM / BLOCK_M) × ceil(N / BLOCK_N) and reads each M tile's
  expert from its first row, offsetting B by expert × N ([SCHED] l.92-94, 151-153). Padding blocks are skipped
  ([SCHED] l.296-300; [PR103]). TMA multicast of B across a CTA pair only when both M tiles belong to the same
  expert ([SCHED] l.284-290). Persistent grid of `num_sms` CTAs; N, K, block sizes and group count compiled in by
  the JIT; M is a runtime argument ([OLD-API] l.86-111).
- psum layout: `grouped_layout` holds one int per expert (end offset), each expert starts at the next aligned
  offset; `expected_m_for_psum_layout` supplies a host hint ([API] l.212-220; [SCHED] l.205-225; [GEN] l.399-401).
- DeepEP V2 dispatch writes the expanded layout directly with `expert_alignment` = DeepGEMM's alignment and
  zero padding between experts ([EP2] l.17, 161, 198-200); before, SGLang permuted DeepEP normal-dispatch output
  into the contiguous format with a custom Triton kernel ([LMSYS]).
- Normal dispatch blocks the CPU until counts arrive, so it is not CUDA-Graph compatible ([EP1] l.173-174);
  DeepEP V2 `do_cpu_sync=True` returns exact sizes and CPU expert counts for training/prefill ([EP2] l.163).
- SM90 FP8: only NT layouts (K-major A and B) ([RD] l.63; test `get_major_ab(False, arch != 9 ...)` [GEN] l.207).

## Discrepancies

1. **Notion "without padding waste".** The layout does pad: every expert segment is rounded up to the M
   alignment (128 rows by default, 256 on SM100), with −1 / zero rows ([RD] l.78; [HEUR]). What it avoids is
   capacity padding (padding every expert to a fixed capacity). Slides say "≤ one tile of padding per expert".
2. **Notion Core Mechanism "unlike CUTLASS generic grouped GEMM"** matches [RD] l.78. No discrepancy, but note
   the baseline in the performance table is DeepSeek's internal CUTLASS 3.6-based implementation, not a public
   CUTLASS grouped GEMM ([RD-0325] l.10); slides say "tuned CUTLASS 3.6 baseline".
3. **API names drift.** The 2025 API was `m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(lhs, rhs, out, m_indices)`
   ([OLD-API]); main exposes `m_grouped_fp8_fp4_gemm_nt_contiguous` with aliases `m_grouped_fp8_gemm_{nt,nn}_contiguous`
   and a `grouped_layout` that is either per-row indices or per-expert prefix sums ([API]). Slides use the current alias.
4. **Performance numbers are from March 2025 (H800).** The README dropped its performance tables in April 2025
   (PR #86); newer releases claim "much faster MoE GEMM" ([PR304]) without published grouped-GEMM tables, so the
   slides show the last published table and say so.
