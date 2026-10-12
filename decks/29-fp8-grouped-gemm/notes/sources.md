# Sources — deck 29: FP8 grouped GEMM (HPC-Ops)

Repository files fetched 2026-10-13 from `github.com/Tencent/hpc-ops` (main) and `github.com/deepseek-ai/DeepGEMM`
(main); copies in `$SCRATCH/research/29/`.

## Sources read

| Tag | Source | Where |
|---|---|---|
| [RD] | HPC-Ops README (main; cached `research/cache/gh-hpc-ops-README.md`) | Performance table row "GroupGEMM FP8 · Expert grouped matmul, prefill, decode · DeepGEMM · Up to 1.1x prefill, 1.88x decode"; Operator Catalog row "Grouped GEMM · FP8 grouped GEMM · Expert-parallel and grouped expert matmul · FP8 · hpc/group_gemm.py"; Basic Usage example (`num_tokens_per_group` = 8 per group, `cu_num_tokens_per_group` = cumsum, `group_gemm_pertensor_fp8`); requirements (SM90) |
| [RD-JAN] | HPC-Ops README @ `bea9bbc2cb` (2026-01-26) | "Maximum observed speedup per operator" table: GroupGEMM (fp8) vs DeepGEMM, prefill 1.1x, decode 1.88x; "We focus on maximum speedup ..." note |
| [API] | `hpc/group_gemm.py` | signatures and shapes: x [total_seq, K] fp8, weight [G, N, K] fp8, seqlens [G] int32, cu_seqlens [G+1] int32, y_scale [G] fp32 (per-tensor), x_scale [K/128, total_seq_pad] and w_scale [G, N/128, ceil(K/128/4)·4] (block-wise), output [total_seq, N] bf16; `reformat_x_scale` for DeepEP-format input |
| [EN] | `src/group_gemm/entry.cc` | dtype checks (fp8 e4m3 in, bf16 out); task map only when `num_seq_per_group_avg <= 8` and a workspace is passed; `reformat_x_scale` accepts K/128 = 16, 32, 56 (K = 2048, 4096, 7168) and tile 8/16/32/48/64; cp.async path `pick_tile_m`, `num_group <= 512` |
| [PT] | `src/group_gemm/sm90/group_gemm_pertensor_fp8.cu` | tile table (avg ≤8→8, ≤16→16, ≤32→32, ≤48→48, ≤64→64, ≤96→48, ≤128→32, ≤144→48, else 64); kTileN = 128, kTileK = 128, 8 stages, 2 math warpgroups; `use_pdl = true`; grid = SM count, block = 384; update kernel grid = num_group + 1 (2·num_group + 1 with task map); task-loop policies 0 (task map), 1 (k ≤ 1024 or n ≤ 1024), 2 (otherwise) |
| [BW] | `src/group_gemm/sm90/group_gemm_blockwise_fp8.cu` | same tile table; kTileS = 64; `use_pdl` passed through from caller |
| [CFG] | `src/group_gemm/sm90/config.h` | `mma_selector`: SM90_64x{8,16,32,48,64,96,128}x32_F32E4M3E4M3_SS_TN; TMA loads for X, W (and scales), TMA store for Y; Y layout (N, M) |
| [KER] | `src/group_gemm/sm90/kernels.cuh` | `update_grouped_tma` (per-group TMA descriptor for X at `x + cu_seqlens[g]·K` with `seqlens[g]` rows, and for Y; tiles = ceil(seqlens/TileM), BlockScan → cu_tiles; task map for TileM = 8); `group_gemm_fp8_kernel` (producer warpgroup with `warpgroup_reg_dealloc`, math warpgroups `partition_A(sB)` = weights, `partition_B(sA)` = tokens; per-K-block `ScaleOut::Zero` WGMMA then `tDr = tCr * scale + tDr`; `get_next_tile_horizon` / `get_next_tile_vert` (binary search over cu_tiles) / task map); block-wise kernel `tCS = x_scale · w_scale` per K block, `w_scale` read 4 K-blocks at a time |
| [TST] | `tests/test_group_gemm_pertensor.py`, `tests/test_group_gemm_blockwise.py` | per-tensor: 8 groups, actual_m 8…512, N 4096, K 7168; block-wise DeepEP format: 256 groups, 1,280-row slots, 30 valid tokens, `cu_seqlens = [0, 1280, 2560, …]`, `reformat_x_scale` before the GEMM |
| [GMMA] | `3rd/cutlass/include/cute/arch/mma_sm90_gmma.hpp` and `mma_sm90_gmma_ext.hpp` (in the HPC-Ops repo) | FP8 E4M3 atoms `MMA_64xNx32_F32E4M3E4M3_*` exist for N = 8, 16, …, 256 (step 8); M is always 64 |
| [DG-RD] | DeepGEMM README (cached `gh-deepgemm-README.md`) | l.76-78 contiguous layout: "each expert segment must be aligned to the GEMM M block size"; l.82-86 masked layout for decoding with CUDA graphs |
| [DG-H] | DeepGEMM `csrc/jit_kernels/heuristics/sm90.hpp` and `runtime.hpp` (main, fetched 2026-10-13) | SM90 `block_m_candidates`: MGroupedContiguous = `get_mk_alignment_for_contiguous_layout()`; MGroupedMasked = {64, 128}; `kLegacyMKAlignmentForContiguousLayout = 128` |
| [PR53] | HPC-Ops PR #53 (merged 2026-06-11) | block-wise fused MoE hang when tile-size tables disagreed across entry / gather / activation / GEMM; fixed by unifying the per-group tile size |
| [LMSYS] | LMSYS blog "HPC-Ops × SGLang" (2026-08-07) | MoE section: "each expert in an MoE layer receives only a handful of tokens" during decode; task-map-driven persistent expert GEMMs (context only) |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| seqlens [3, 1, 5, 2], cu_seqlens [0, 3, 4, 9, 11]; experts with 8, 3, 13, 1 tokens | illustrative running examples | constructed for the figures |
| FP8 e4m3 in, BF16 out; SM90 / H20 | interface and target | [EN], [RD] |
| 87.5% | padding share of a 64-row tile holding 8 tokens | derived: 1 − 8/64 (below); 64 from [GMMA] + [DG-H]; 8 tokens/group from [RD] example |
| 64 (fixed M), 8–256 step 8 (N) | Hopper FP8 WGMMA shapes | [GMMA] |
| 128-row boundary | DeepGEMM contiguous-layout default alignment | [DG-H] runtime.hpp; [DG-RD] l.78 |
| 64-row tile | smallest DeepGEMM SM90 masked BLOCK_M | [DG-H] sm90.hpp |
| 8, 16, 32, 48, 64 | HPC-Ops token tile sizes | [PT], [BW] |
| 1,280-row slots, 30 tokens | DeepEP-format test (speaker notes) | [TST] |
| 100% vs 12.5%; 8× | analytic MMA-row utilization at ≤ 8 tokens per expert | derived (below) |
| 128-wide K block; 1×128 and 128×128 scale blocks | promotion granularity and block-wise scaling | [PT] kTileK = 128; [API] shapes |
| 1.88× (decode), 1.1× (prefill) | max observed speedup over DeepGEMM on H20 | [RD], [RD-JAN] |
| K = 2048 / 4096 / 7168 | `reformat_x_scale` supported K | [EN] |
| PR #53 | tile-table mismatch hang | [PR53] |

## Derived numbers

1. **87.5%**: an expert with t = 8 tokens in a 64-row token-major tile → padding = (64 − 8) / 64 = 87.5%.
2. **MMA-row utilization** (chart `charts/tile_util.py`), equal group size t, util = t / (ceil(t/tile)·tile):
   - HPC-Ops tile = table above applied to avg = t: t = 1, 2, 4, 8, 16, 32, 64, 128, 256 → 12.5, 25, 50, 100,
     100, 100, 100, 100, 100 %.
   - 64-row tile: 1.6, 3.1, 6.2, 12.5, 25, 50, 100, 100, 100 %.
   - 128-row segment: 0.8, 1.6, 3.1, 6.2, 12.5, 25, 50, 100, 100 %.
   - Ratio HPC-Ops / 64-row for t ≤ 8: 8× (e.g. 12.5 / 1.6, 100 / 12.5).
   This is an arithmetic property of the tile shapes, not a measured speedup; the slide says "analytic".

## Mechanism facts (no numbers) used on slides

- Swap-AB: weights are the WGMMA A operand (M side, 2 warpgroups × 64 = 128 rows per CTA tile), tokens the B
  operand (N side, TileM = 8…64); output written transposed through TMA (Y viewed as (N, M)) ([KER], [CFG]).
- Kernel 0 (`update_grouped_tma`): one CTA per group rewrites the X and Y TMA descriptors for that group; one extra
  CTA computes tiles / cu_tiles with a block scan; optional task map of (m-tile, n-tile, group) for TileM = 8
  ([KER]). TMA zero-fills out-of-range rows on load and clips stores (standard TMA bounds behaviour of the
  per-group descriptor shape).
- Kernel 1: persistent (grid = SM count), 384 threads = 1 producer warpgroup + 2 math warpgroups, 8-stage
  shared-memory pipeline with mbarriers; tile id → (group, m-tile, n-tile) via scan, binary search over cu_tiles,
  or the task map ([PT], [KER]).
- Scaling: per-tensor one FP32 scale per group; block-wise per-token-per-128-K activation scales and per-128×128
  weight-block scales; every K block's WGMMA result is scaled and added into a separate FP32 accumulator ([KER]).
- PDL: the per-tensor launcher forces `use_pdl = true` (kernel 0 → kernel 1); block-wise uses the caller's flag ([PT], [BW]).
- Host needs only `num_seq_per_group_avg` (to pick the tile); per-group counts stay on the device ([API], [EN]).

## Discrepancies

1. **Notion: "accepting cu_num_tokens input".** The API arguments are `seqlens` and `cu_seqlens`; the README
   example names the tensors `num_tokens_per_group` / `cu_num_tokens_per_group`. Same concept; slides use the API
   names.
2. **Notion body: LMSYS blog as the "prefill/decode comparison vs DeepGEMM".** The LMSYS blog has no grouped-GEMM
   vs DeepGEMM comparison (its MoE tables compare the fused MoE against SGLang / vLLM). The 1.1× / 1.88× figures
   come from the HPC-Ops README (current and January 2026 versions). Slides cite the README.
3. **Notion: "vs DeepGEMM: 1.1x on prefill and 1.88x on decode".** The README states these as maxima ("Up to",
   "Maximum observed speedup"); slides say "up to".
4. Notion lists MegaBlocks and the FP8-formats paper as references; neither is about this kernel and neither is
   used for numbers.
