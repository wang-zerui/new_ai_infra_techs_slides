# Sources — deck 32: K-axis grouped GEMM (MoE weight backward)

Downloads: `$SCRATCH/research/32/` (PR bodies, `wgrad_gemm.py` and `test_core.py` at PR #95's commit `04278f6dee`,
`python_api.cpp` at `9da4a23561`) and the DeepGEMM main-branch files in `$SCRATCH/research/33/src/`.
DeepGEMM `main` = `057ca5964a` (2026-09-30, v2.8.1).

## Sources read

| Tag | Source | Where |
|---|---|---|
| [RD] | DeepGEMM README, current (cached) | l.24 news 2025.05.14 (weight-gradient kernels for dense and MoE backward, PR #95); l.63 (D = C + A @ B naming; SM90 NT only, SM100 all layouts); l.78-80 (M-grouped contiguous; "K-axis-grouped API for MoE weight backward (with M and N must remain fixed)", `k_grouped_fp8_gemm_tn_contiguous`); l.154-155 (M/K alignment utilities); l.162 (`get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`: K-grouped GEMM packing kernel) |
| [PR95] | DeepGEMM PR #95 "Weight gradient kernels for dense and MoE models" (2025-05-14) | body: `wgrad_gemm_fp8_fp8_fp32_nt`, `k_grouped_wgrad_gemm_fp8_fp8_fp32_nt`, "~20% speedup compared to the internal CUTLASS implementation"; dense wgrad table (12 rows) and grouped wgrad table (6 rows); no GPU named (DeepGEMM was Hopper-only then, README @04278f6dee l.5) |
| [V1-API] | `deep_gemm/jit_kernels/wgrad_gemm.py` @ `04278f6dee` | l.135-175: `k_grouped_wgrad_gemm_fp8_fp8_fp32_nt(lhs, rhs, out, batch_sizes)`, "processing each batch sequentially": a Python loop calling the dense wgrad kernel per expert; 1×128 scaling on both operands; FP32 output accumulated; `batch_sizes` is a host list |
| [V1-TEST] | `tests/test_core.py` @ `04278f6dee` | l.270-297: grouped wgrad benchmark (groups 4/8, base K 4096/8192 ± 128, (M, N) = (7168, 4096), (2048, 7168)); time = kineto kernel time of the per-expert kernels |
| [V2-SM100] | `csrc/python_api.cpp` @ `9da4a23561` (PR #112, 2025-07-18) | l.293-330: `k_grouped_fp8_gemm_tn_contiguous`, one kernel `fp8_k_grouped_gemm_1d1d`, SM100 only, MN-major A and B, host `ks` + device `ks_tensor` |
| [V3-SM90] | `csrc/apis/gemm.hpp` @ `38f8ef73a4` (PR #231, 2025-11-21) | l.293-337: `k_grouped_fp8_gemm_nt_contiguous` for SM90 with a per-SM tensor-map buffer |
| [API] | `csrc/apis/gemm.hpp` @ main | `check_k_grouped_args` l.55-76 (host K list optional only with psum layout); `k_grouped_fp8_gemm_tn_contiguous` l.320-371 (recipe (1, 1, gran_k), gran_k 32 or 128, `k_alignment % 128 == 0`, "All A/B padding must be zero", A `[sum_k, M]`, B `[sum_k, N]` MN-major, D `[G, M, N]`, `grouped_layout[i]` = padded K of group i or psum end, SM100 only); `k_grouped_fp8_gemm_nt_contiguous` l.373-427 (recipe (1,1,128), "No psum on FP8 NT", host K list and C required, K-major, `num_sms * 4` tensor maps, SM90); FP4 variant l.429-486 (`k_alignment % 256 == 0`); registrations l.895-918 |
| [SCHED] | `deep_gemm/include/deep_gemm/scheduler/gemm.cuh` @ main | l.40-43 ("K alignment must be a multiple of BLOCK_K (128)"); l.63-82 (`current_k_start`, `current_shape_k`, psum: next group starts at `align(previous_end, kKAlignment)`); l.101-104 (`num_blocks = num_m_blocks * num_n_blocks`, same for every group); l.157-168 (MN offset = group × shape, K offset = `current_k_start`, SF offset = `current_sf_k_cumsum`); l.226-246 (CTAs walk group by group) |
| [SM90-K] | `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh` @ main | l.179 (skip empty groups); l.193-201 (on a group change, `tensor_map_replace_global_addr_in_smem` and `..._inner_dim_stride_in_smem` with the group's K) |
| [LAYOUT] | `csrc/apis/layout.hpp` @ main | `transform_k_grouped_sf_into_required_layout` l.89-116: SM90 FP32 scales with gran_k = k_alignment = 128; SM100 converts FP32 scales into packed UE8M0 with the K-grouped packing kernel |
| [GEN] | `tests/generators.py` @ main | `enumerate_k_grouped_contiguous` l.229-262: shapes (8, 768, 2048, 128) "stressing the SM90 in-place tensor map update", (4, 4096, 7168, 8192) and (4, 7168, 2048, 8192) labelled EP64, (8, …, 4096) EP32, (16, …, 2048) EP16; SM100 output options FP32 accumulate, FP32, BF16; `k_grouped_cast_fp8_fp4_with_major` l.488-540 (each group cast independently; SF rows "compact by group and padded to k_alignment"); `generate_k_grouped_contiguous` l.551-608 (`ref_d[i] = a[start:end].T @ b[start:end] + c[i]`) |
| [HEUR] | `csrc/jit_kernels/heuristics/runtime.hpp` @ main | l.12 default M/K alignment 128; l.57-64 SM100 theoretical 256, "friendlier to ... K-grouped operators" |
| [PR432] | DeepGEMM PR #432 "Public Release 26/09" | "Faster MoE backward GEMMs, with ~20% gains" |
| [DSV3] | DeepSeek-V3 report, arXiv 2412.19437v2 | §3.3.1 p.15 (Fprop, Dgrad, Wgrad all in FP8; "theoretically doubles the computational speed compared with the original BF16 method"; FP8 Wgrad lets activations be cached in FP8); §3.3.2 p.16 (1×128 activation tiles, 128×128 weight blocks, scaling along the inner dimension); §3.3.3 p.18 (weight gradients kept in FP32; some activations converted from 1×128 to 128×1 tiles for the backward pass); §4.2 p.22 (hidden 7168, expert width 2048, 256 routed experts, top-8) |
| [MC] | Megatron-Core MoE report, arXiv 2603.07685v2 | §5.4.1 p.54-55: in the weight-gradient GEMM the dot-product dimension is the token dimension, which varies dynamically and needs zero padding (16 for blockwise FP8, 32 for MXFP8/NVFP4); tokens-per-expert padding raised to 128 for grouped quantization and CUDA Graphs; §5.4.3 p.55: Wgrad uses transposed activations and gradients |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 1/3 | Wgrad share of an expert's training GEMM FLOPs: Fprop, Dgrad, Wgrad each 2·M·N·K | analytic (derived); FP8 for all three: [DSV3] §3.3.1 p.15 |
| 2× (theoretical) | FP8 vs BF16 GEMM speed, DeepSeek-V3's statement | [DSV3] §3.3.1 p.15 |
| 128 tokens | K alignment of each expert segment (= BLOCK_K); SM90 requires exactly 128 | [SCHED] l.40-43; [LAYOUT] l.100; [HEUR] l.12 |
| 256 | SM100 theoretical alignment | [HEUR] l.57-64 |
| ≤ 127 tokens | worst-case zero padding per expert at 128 alignment | derived |
| 7168, 4096, 2048 | DeepSeek-V3 hidden, fused gate+up width (2 × 2048), expert width | [DSV3] §4.2 p.22 (4096 derived) |
| 939, 900, 1044, 1033 TFLOPS | grouped wgrad, 4 experts, (M, N, K per expert) = (7168, 4096, 4096), (2048, 7168, 4096), (7168, 4096, 8192), (2048, 7168, 8192) | [PR95] grouped table |
| 942, 902 TFLOPS | grouped wgrad, 8 experts, K = 4096, same (M, N) | [PR95] grouped table |
| 942, 920, 1061, 1037 TFLOPS | dense wgrad with the same per-expert (M, N, K) | [PR95] dense table |
| ≥ 97.8% | min of grouped ÷ dense: 939/942 = 99.7%, 900/920 = 97.8%, 1044/1061 = 98.4%, 1033/1037 = 99.6%, 942/942 = 100%, 902/920 = 98.0% | derived |
| 900–1044 TFLOPS | range of the 4-expert grouped wgrad numbers | [PR95] |
| ~20% | wgrad kernels vs internal CUTLASS implementation (May 2025) | [PR95] |
| ~20% | faster MoE backward GEMMs in the Sept 2026 release | [PR432] |
| 2025-05 / 2025-07 / 2025-11 / 2026-09 | loop → SM100 single kernel → SM90 single kernel → psum layout | [V1-API]; [V2-SM100]; [V3-SM90]; [API] (@66081d4c, PR #432) |
| EP16 / EP32 / EP64 → 16 / 8 / 4 local experts with 2048 / 4096 / 8192 tokens each | test shapes | [GEN] l.251-254 |

Mapping of benchmark shapes to projections (derived from [DSV3] dims): (M, N) = (7168, 4096) ↔ gate+up weight
[4096, 7168]; (2048, 7168) ↔ down weight [7168, 2048].

Running-example token counts in diagrams (300, 60, 200 tokens; K block = 128) are illustrative.

## Mechanism facts (no numbers) used on slides

- Wgrad of expert e: dW_e = dY_eᵀ X_e (sum over the expert's tokens); DeepGEMM computes D[g] = C[g] + A_gᵀ B_g with
  A = `[ΣK, M]`, B = `[ΣK, N]`, both token-major (MN-major) on SM100 ([API] l.340-343; [GEN] l.583-588).
- Fprop and Dgrad of the experts use the M-grouped contiguous kernels, `nt` and `nn` ([RD] l.78).
- Segments start on multiples of the K alignment; padding must be zero; empty experts are allowed ([API] l.333-345;
  [SCHED] l.63-82; test variants with K = 0).
- Every group has the same `num_m_blocks × num_n_blocks` tiles; the K loop of a tile covers only its group's
  segment; scale offsets advance per group ([SCHED]).
- Scales: recipe (1, 1, 128) = one scale per row channel per 128 K (= tokens) on both operands; SM100 packs
  them to UE8M0 per group with the K-grouped packing kernel ([LAYOUT]; [RD] l.162).
- psum layout: `grouped_layout[i]` = logical K end of group i; group i+1 starts at the next aligned offset;
  the host K list becomes optional (SM100) ([API] l.55-76, l.345-349).
- SM90: operands per expert are K-major blocks; each CTA keeps TMA descriptors in shared memory and rewrites address
  and stride when it moves to the next expert; needs the host K list and accumulation into FP32 C ([SM90-K]; [API]).

## Discrepancies

1. **Notion "purpose-built ... with a matching K-grouped UE8M0 packing kernel"** matches [RD] l.162, but the UE8M0
   packing applies to SM100 only; SM90 uses FP32 scales ([LAYOUT] l.103-109). Slides say so.
2. **Notion "puts the MoE backward pass on the FP8 fine-grained-scaling path ... unlike inference-only operator
   libraries".** True for weight gradients; the comparison with "inference-only libraries" is not in the primary
   sources and is not repeated on the slides.
3. **Notion names only `k_grouped_fp8_gemm_tn_contiguous`.** That API is SM100-only ([API] l.361-370); Hopper uses
   `k_grouped_fp8_gemm_nt_contiguous` (since Nov 2025), and the May 2025 Hopper version was a host loop
   ([V1-API]). Slides show both.
4. **The only published per-kernel numbers ([PR95]) measure the May 2025 per-expert loop** (kernel time, launch gaps
   excluded, [V1-TEST] l.294), on an unnamed Hopper GPU; there are no published numbers for the single-kernel
   versions beyond "~20% faster MoE backward GEMMs" ([PR432]). Slides label the chart accordingly.
