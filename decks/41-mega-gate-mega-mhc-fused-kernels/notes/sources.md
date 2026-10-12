# Sources — deck 41: Mega Gate / Mega mHC fused kernels

## Sources read

| Tag | Source | Where |
|---|---|---|
| [V4] | DeepSeek-AI, "DeepSeek-V4", arXiv 2606.19348v1 (cached text) | §2.1 p.7 (Sqrt(Softplus) affinity, aux-loss-free bias), §2.2 p.7-8 (HC/mHC equations, Sinkhorn, t_max = 20), §3.3 p.19 (mHC matmul with output dimension 24, split-K, deterministic reduction), §3.4.2 p.20 (fused mHC kernels, recomputation, 6.7%), §4.2.1 p.24-25 (n_hc = 4, t_max = 20, d = 4096 / 7168, 256 / 384 experts, top-6) |
| [MHC] | Xie et al., "mHC: Manifold-Constrained Hyper-Connections", arXiv 2512.24880 (cached text) | §1 p.4 (6.7% at n = 4), §3 p.5-6 (HC mappings), §3.2 + Table 2 p.7-8 (memory access per token), §4.2 p.9 (parameterization, Sinkhorn), §4.3.1 p.9-10 (kernel fusion, Eq. 10-19, F_post,res I/O), §4.3.2 + Table 3 p.10-11 (recomputation) |
| [PR432] | DeepGEMM PR #432 "Public Release 26/09" (merged 2026-09-10) | body: "Mega Gate … 30–75% speedups", "Mega mHC … 45–85% speedups", determinism opt-in |
| [PR462] | DeepGEMM PR #462 (merged 2026-09-30) | body: Mega Gate only `sqrtsoftplus`; Mega mHC test baseline no longer depends on tile-kernels |
| [RD] | DeepGEMM README (cached) | line 3 (HC kernels in the library), line 14 (2026.09.10 news) |
| [MHC-API] | `csrc/apis/mega_mhc.hpp` at main 057ca596 | docstring (args, shapes, dtypes, split-K: 16 in deterministic mode, FP8 gran 32, y_shared_sf = Mega MoE shared-expert layout), asserts |
| [MHC-K] | `deep_gemm/include/deep_gemm/impls/sm100_mega_mhc.cuh` (same commit) | role layout (lines 78-100, 185), Post workers (lines 387-560), output-store warp EVICT_LAST comment (lines 342-343), workspace epilogue (570-655), Mix (657), Norm (669-690) |
| [MHC-L] | `deep_gemm/include/deep_gemm/layout/mega_mhc.cuh` | kNumRoutes = 4, kNumHCOutputs = 24, BLOCK_M = 64, BLOCK_K = 64, 6 warpgroups, kDefaultNumSplits = 16, kNumMaxSplits = 64 |
| [MHC-T] | `tests/test_mega_mhc.py` at main and at PR #432 merge 66081d4c; `third-party/tilelang_ops/ref_mhc.py` at 66081d4c | reference math, baseline kernels (NORMAL_KERNELS 4, SHIFTED_KERNELS 7), bitwise checks, byte model ("logical" vs "incl. Scratch I/O") |
| [GATE-API] | `csrc/apis/mega_gate.hpp` | docstring and asserts (≤ 512 routed experts, top-k ≤ 32, hidden % 256, sqrtsoftplus only, ranking on score + bias, weights from unbiased scores) |
| [GATE-K] | `deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_gate.cuh` | TMA warp, MMA warp ("weights are the A operand, so a TMEM lane holds one expert and a column one token"), gate warps (score workspace, score barrier, top-k loop) |
| [GATE-T] | `tests/test_mega_gate.py` at main and at 66081d4c | baseline = `cublaslt_gemm_nt` (kernels nvjet / splitKreduce / cutlass) + TileLang `moe_topk_gate`; perf cases T = 1…32768, H = 4096/5120/7168, E = 384, top-6 |
| [MOE-RD] | DeepGEMM README "Mega MoE" (lines 112-144) | Mega MoE buffer takes `topk_idx` (int64), `topk_weights` (FP32), FP8 x + scales |

## Numbers on slides

Notation: the mHC paper writes C for the layer width; V4 and the slides write d (C = d).

| Number on slide | Meaning | Source |
|---|---|---|
| n_hc = 4 | residual-stream expansion in V4-Flash and V4-Pro | [V4] §4.2.1 p.24-25 |
| t_max = 20 | Sinkhorn-Knopp iterations in V4 | [V4] §2.2 p.8, §4.2.1 |
| 24 coefficients per token per layer | n(n+2) = 4 + 4 + 16 (H_pre, H_post, H_res); "output dimension of only 24" | [V4] §3.3 p.19; [MHC] Eq. 10; [MHC-L] kNumHCOutputs |
| d = 4096 / 7168 | V4-Flash / V4-Pro hidden size | [V4] §4.2.1 |
| 256 / 384 routed experts, top-6; Sqrt(Softplus) scores | V4 router | [V4] §2.1 p.7, §4.2.1 |
| 3C vs (5n+1)C + n² + 2n reads and (3n+1)C + n² + 2n writes | per-token memory access of one residual layer, forward, excluding F's internal I/O | [MHC] Table 2 p.8 |
| 34C vs 3C, **≈ 11×** (derived) | Table 2 at n = 4: reads 21C + 24, writes 13C + 24 → 34C + 48 elements vs 3C for a plain residual; 34/3 = 11.3 (the + 48 is negligible for C ≥ 4096) | derived from [MHC] Table 2 |
| (3n+1)C → (n+1)C reads, 3nC → nC writes | I/O of the mHC paper's fused F_post,res kernel | [MHC] §4.3.1 p.10 |
| 6.7% | mHC (n = 4) wall-time overhead with fused kernels + recomputation + adjusted DualPipe ("of the overlapped 1F1B pipeline stage") | [V4] §3.4.2 p.20; [MHC] §1 p.4, §4.3 p.9 |
| 4 kernels (normal) / 7 kernels (shifted) | multi-kernel baseline at one layer boundary: `mhc_post_fwd`, `tf32_hc_prenorm_gemm`, `mhc_pre_big_fuse_kernel`, `_norm_kernel` (normal); + `mhc_pre_apply_mix_fwd`, `mhc_reduce_partials_and_rmsnorm_fwd`, `mhc_pre_split_mixes_fwd`, `mhc_sinkhorn_fwd` (shifted, without big_fuse) | [MHC-T] ref_mhc.py NORMAL_KERNELS / SHIFTED_KERNELS |
| 45–85% faster (1.45–1.85×) | Mega mHC vs that baseline (sum of baseline kernel times / Mega mHC time) | [PR432]; ratio definition in [MHC-T] test at 66081d4c (`baseline_t / kernel_t`) |
| 30–75% faster (1.30–1.75×) | Mega Gate vs cuBLASLt GEMM + TileLang top-k gate | [PR432]; [GATE-T] |
| 6 positions, 30 runs | batch-invariance check: one token copied to rows 63, 64, 511, 512, T−1 besides row 0 (6 positions when T > 512); 30 repeated calls compared bitwise | [MHC-T] `make_inputs` duplicates, `test_mega_mhc` loop |
| 16 splits | fixed split-K count in deterministic mode (otherwise ≤ 64, chosen per call) | [MHC-API] notes; [MHC-L] |
| 6 warpgroups (768 threads) per CTA | Mega mHC CTA: WG0 TMA/MMA/store, WG1-2 Post, WG3 split epilogue, WG4 Mix, WG5 Norm | [MHC-K] line 185; [MHC-L] kNumThreads = 6 × 4 × 32 |
| ≤ 512 routed experts, top-k ≤ 32, hidden % 256 = 0 | Mega Gate limits | [GATE-API] |
| 10C logical, 16C incl. scratch (normal mode), 12C (shifted) — analytic | Mega mHC per-token traffic in elements of C at n = 4: reads x (C) + residual (4C), writes new residual (4C) + y (C) = 10C; the test's scratch term adds the materialised norm boundary (2C) and, in normal mode, the new-residual re-read (4C) | derived from [MHC-T] byte model (`logical_bytes`, `intermediate_io_bytes`), BF16 outputs only; mixes and fn (amortised over tokens) excluded, as Table 2 excludes them |
| 20C (analytic) | 4-kernel baseline traffic per token: reads x C + residual 4C (post) + new residual 4C (GEMM) + new residual 4C (pre-apply) + norm input C (norm) = 14C; writes new residual 4C + norm input C + y C = 6C | derived from [MHC-T] ref_mhc.py data flow (`mhc_post` → `tf32_hc_prenorm_gemm` → `mhc_pre_big_fuse` → `norm`) — used only in speaker notes |

Charts: `charts/speedup.py` draws the two published ranges as floating bars (no per-shape data is
published); `charts/traffic.py` draws the analytic per-token element counts (3C, 34C, 10C/16C) and is
labelled "analytic" on the slide.

## Mechanism facts (no numbers) used on slides

- What Mega mHC computes per token ([MHC-T] `mhc_reference`, [MHC-API] docstring): inputs x [d] BF16 (layer
  output), residual [4, d] BF16, post_mix [4] and comb_res_mix [4×4] FP32 (this layer's H_post, H_res), fn
  [24, 4d] FP32, mix_scales [3], mix_bases [24], rmsnorm_weight [d]. Steps: new_residual = H_post·x +
  H_resᵀ·residual (rounded to BF16); mixes = new_residual(flattened 4d) · fnᵀ (24 outputs), divided by the RMS
  of the flattened row (norm reordered after the GEMM, as [MHC] §4.3.1), × α + b; H_pre = σ(·) + eps,
  H_post = σ(·) × scale, H_res = Sinkhorn(4×4); norm input = Σ_i H_pre[i]·new_residual[i]; y = RMSNorm(norm
  input) × weight → BF16 and/or FP8 + per-32 UE8M0 scales (GEMM column-major layout, or the routed + shared
  layouts Mega MoE uses). Outputs: new_residual, next H_post, next H_res, (shifted: next H_pre), y.
- Shifted variant: takes H_pre as an input (`shifted_prev_mix`) and emits the next one (`new_prev_mix`), so
  the norm input is formed in the same pass as the post-mix ([MHC-T], [MHC-K] Post worker x1 path).
- Post workers ([MHC-K] lines 450-535): read x and the 4 residual rows from shared memory (TMA-loaded),
  compute each new-residual element once, round to BF16, then (a) write it back to the shared staging buffer
  that the output-store warp TMA-stores to HBM, (b) store the values into tensor memory as the A operand of
  the TF32 MMA for the 24-wide projection, (c) accumulate squared sums for the RMS; in shifted mode also
  accumulate x1 = Σ H_pre·new_residual and its squares.
- Output-store warp: new residual stored with an L2 EVICT_LAST hint because normal-mode Norm workers re-read
  it ([MHC-K] lines 342-343).
- Split-K: each task is one (64-token M-block, K-split) pair; partial 24-wide sums and squared sums go to a
  workspace; split barriers gate the Mix; fixed accumulation order ([MHC-K] line 267 comment "Hidden blocks,
  UMMA K atoms, and routes accumulate into one HC output block in fixed order"); deterministic mode fixes 16
  splits ([MHC-API]). [V4] §3.3 describes the earlier approach: split parts written out and reduced
  deterministically "in a subsequent kernel".
- Bitwise checks in [MHC-T]: 30 repeated runs bitwise equal; duplicated tokens produce identical rows
  (batch invariance); CUDA-graph replay and two concurrent streams bitwise equal to eager; streams must be
  warmed up before graph capture (split barriers allocated per stream).
- Mega Gate ([GATE-API], [GATE-K]): BF16 GEMM x[T, d] · W[E, d]ᵀ with the weights as the UMMA A operand
  (TMEM lane = expert, column = token); gate warps apply sqrt(softplus), add the balance bias (and an
  image-token bias if given), rank per token on score + bias, take top-k, emit weights = unbiased scores
  normalised over the top-k sum × routed scaling factor; optional mask, fixed routing, logical→physical
  expert-replica map, shared-expert-as-routed; scores pass through a per-token-block workspace with a score
  barrier when experts or K are split across CTAs. Output `topk_idx` int64 and `topk_weights` FP32 — the
  types Mega MoE's buffer takes ([MOE-RD]).
- Baselines: mHC = TileLang `tile_kernels` mHC ops with DeepGEMM's `tf32_hc_prenorm_gemm` as the GEMM
  ([MHC-T] ref_mhc.py); gate = cuBLASLt GEMM writing FP32 logits [T, E] + TileLang `moe_topk_gate`
  ([GATE-T]).
- Limits: SM100 only (tests filtered on arch 10, `sm100_*` impls); hc_mult fixed to 4 ([MHC-L] kNumRoutes);
  hidden must be a multiple of 1024 for Mega mHC (16 × BLOCK_K, [MHC-API] assert); forward kernels only (no
  backward API in [MHC-API] / [GATE-API]); Mega Gate scoring fixed to sqrtsoftplus ([PR462]).

## Discrepancies

1. **Notion "brings mHC's extra memory traffic into an acceptable range".** The quantified "acceptable range"
   in the primary sources (6.7% overhead) was achieved by the mHC paper's TileLang kernels plus recomputation
   and DualPipe changes in training ([MHC] §4.3, [V4] §3.4.2), before Mega mHC existed (2026-09). DeepGEMM
   reports Mega mHC only as 45–85% faster than that multi-kernel path ([PR432]). Slides keep the two apart.
2. **Notion "avoiding … intermediate tensors".** Mega mHC still writes the new residual (it is an output) and,
   per the test's own byte model, materialises the norm boundary and re-reads the new residual in normal
   mode (kept in L2 by an EVICT_LAST hint); Mega Gate still passes scores through a workspace when experts
   or K are split across CTAs. What disappears are the kernel boundaries and the re-read of the residual
   for the 24-wide GEMM. Slides say "one pass over the stream", not "no intermediates".
3. **Notion source "DeepSeek V4".** The V4 report describes fused mHC kernels for training and inference
   but does not name Mega mHC or Mega Gate; those names and numbers come from DeepGEMM PR #432.
4. **Sinkhorn iterations.** V4 uses t_max = 20 ([V4], [MHC]); the DeepGEMM test uses 10 as a parameter
   (`num_sinkhorn_iters` is an argument). Slides quote 20 for V4.
5. **H_post scale.** [V4]/[MHC] define H_post = 2σ(·); the DeepGEMM API takes `hc_post_scale` as an argument
   (the test uses 1.25). Slides say "σ × scale".
