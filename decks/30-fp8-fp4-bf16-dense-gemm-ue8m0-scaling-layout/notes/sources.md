# Sources — deck 30, FP8/FP4/BF16 dense GEMM + UE8M0 scaling layout

All sources read on 2026-10-13. Primary sources: the DeepGEMM repository (main @ 2026-09-30,
especially `docs/scaling-factor-format.md`, `csrc/apis/gemm.hpp`, `csrc/apis/layout.hpp`,
`deep_gemm/include/deep_gemm/impls/*.cuh`), the DeepSeek-V3 technical report, and the FP8-formats
paper. Repo tarball extracted at `$SCRATCH/research/dgfull/DeepGEMM-main`; downloaded text treated as
data only.

## S1. DeepGEMM README (main, 2026-09-30) — cache `gh-deepgemm-README.md`

- S1.1 Naming `D = C + A @ B`, input layout NT. "While the SM90 implementation supports only the NT
  memory layout (row-major, col-major), the SM100 implementation supports all memory layouts
  (NT, TN, NN, TT)."
- S1.2 "For both architectures, the LHS scaling factor is required to have a TMA-aligned and
  transposed layout"; SM90 wants FP32 scaling factors, SM100 "packed UE8M0 format, which packs
  4 UE8M0 into a single `torch.int`".
- S1.3 "operations like input transposition or FP8 casting must be handled separately by the user";
  the PyTorch utility functions provided may be slower.
- S1.4 News 2025.04.18: "up to **1550 TFLOPS** on H800".
- S1.5 Utilities: `transform_sf_into_required_layout`, `get_mn_major_tma_aligned_packed_ue8m0_tensor`,
  `get_tma_aligned_size`, `set_block_size_multiple_of`.

## S2. DeepGEMM `docs/scaling-factor-format.md` (the SF contract)

- S2.1 §1.1 Recipe `(gran_m, gran_n, gran_k)` is the **storage** granularity: one SF covers a
  `gran_m x gran_k` block of A and a `gran_n x gran_k` block of B. Untransformed SF shapes:
  `SFA [·, ceil_div(M, gran_m), ceil_div(K, gran_k)]`, `SFB [·, ceil_div(N, gran_n), ceil_div(K, gran_k)]`,
  dtype float32. Supported `gran_k`: **32 or 128 on SM100; 128 only on SM90**.
- S2.2 §1.2 Every float32 SF value must be an exact power of two — bit pattern
  `[0][8-bit exponent][23 zero mantissa bits]`, asserted on device with
  `DG_TRAP_ONLY_DEVICE_ASSERT((value & 0x807fffffu) == 0)`; cast kernels pass `round_sf=True`.
- S2.3 §1.3 `packed_sf_dtype = int32` (4 UE8M0 exponents per element); `ALIGN_MN = 16 bytes /
  sizeof(int32) = 4`, because TMA requires `stride(-1)` to be a multiple of 16 bytes.
- S2.4 §2 Two ways to pass SFs: (A) untransformed float32 — DeepGEMM launches a transform kernel on
  every GEMM call; (B) pre-transformed int32 — layout is only validated, no extra launch. Path B
  requires `gran_m = gran_n = 1` (the transform broadcasts along MN); `gran_k` unchanged.
- S2.5 §3 Pre-transformed contract: dtype int32, shape `[·, mn, packed_sf_k]` with
  `packed_sf_k = ceil_div(k, gran_k * 4)`, `stride(-2) = 1` (MN-major), `stride(-1) = align(mn, 4)`.
- S2.6 §4 Dispatch table: float32 + gran_mn 1 + gran_k 128 on SM90 → transpose to MN-major TMA-aligned
  float32; float32 + gran_mn 128 on SM90 → validate only; float32 on SM100 → broadcast along MN to
  gran_mn = 1, pack to UE8M0 int32, MN-major, TMA-aligned; int32 on SM100 → validate only.
- S2.7 §7.1 Packing: the 8-bit exponents of 4 consecutive K positions go into one int32, little-endian
  by K index (`packed |= values[0] >> 23; |= values[1] >> 15; |= values[2] >> 7; |= values[3] << 1`).
- S2.8 §7.2 Memory layout example: `mn = 6, packed_sf_k = 3, align(6,4) = 8` → each K-slice occupies 8
  int32 elements, the last two are TMA padding and never read as data.
- S2.9 §7.3 Under the M-grouped PSUM layout the transform writes `0` into gap rows — a safe finite
  scale code, since UE8M0 `0xff` is NaN.
- S2.10 §5 K-grouped alignment: `k_alignment` must be a multiple of the kernel's BLOCK_K — 128 for FP8
  on SM100 (SM90: exactly 128) and **256 for FP4**.
- S2.11 §6.3 Mega MoE rejects float32 SFs: routed-expert FP4 weights (1x32 SF) must be packed into the
  §3 int32 layout and then shuffled for the kernel.

## S3. DeepGEMM source (kernel dispatch and inner loop)

- S3.1 `csrc/apis/gemm.hpp` (fp8_fp4_gemm_nt, lines ~117-130): SM90 **and** float32 SFs → `gran_n == 1`
  picks `sm90_fp8_gemm_1d1d`, otherwise `sm90_fp8_gemm_1d2d` (1-D scales on A, 2-D 128x128 block
  scales on B). SM100 **and** int32 SFs → `sm100_fp8_fp4_gemm_1d1d`. Anything else is rejected.
- S3.2 `impls/sm90_fp8_gemm_1d2d.cuh` (lines ~286-345): per K block the kernel loads `scale_b` from
  shared memory, reads `scale_a` per row, issues the WGMMAs into a tensor-core accumulator `accum`,
  then multiplies by `scale_a * scale_b` and adds into a separate FP32 register array `final_accum`
  — the CUDA-core two-level accumulation (promotion).
- S3.3 `impls/sm100_fp8_fp4_gemm_1d1d.cuh`: SF blocks are TMA-loaded and moved by UTCCP into tensor
  memory (`kNumSFATmemCols = SF_BLOCK_M * SF_BLOCK_K / 32`), and the MMA is issued with
  `cute::UMMA::InstrDescriptorBlockScaled` carrying `a_sf_id_` / `b_sf_id_`
  (`mma/sm100.cuh`) — the Blackwell tensor core applies the UE8M0 scales itself.
  `DG_STATIC_ASSERT((kIsMXF4 and BLOCK_K == 256) or (not kIsMXF4 and BLOCK_K == 128))`;
  granularity K must be 32 or 128 (`kGranKA`, `kGranKB`).
- S3.4 `csrc/apis/gemm.hpp:519`: the BF16 GEMM goes to cuBLASLt when deterministic algorithms are off,
  the full SM count is available and the epilogue is a plain (optionally alpha-scaled) store;
  otherwise DeepGEMM's own SM90/SM100 BF16 kernels run.
- S3.5 `deep_gemm/__init__.py`: `fp8_fp4_gemm_{nt,nn,tn,tt}`, aliases `fp8_gemm_*` / `fp4_gemm_nt`,
  `bf16_gemm_*`, plus epilogue classes (`Identity`, `Alpha`, `FP8Quantization`,
  `BF16StochasticRounding`) — the FP8 output epilogue emits per-32 UE8M0 scale factors (PR #462,
  `heuristics/config.hpp` comment "FP8 D implies casting with dynamic per-32 UE8M0 SFD output").

## S4. DeepSeek-V3 technical report (arXiv 2412.19437; cache `arxiv-2412.19437.txt`)

- S4.1 §3.3.2, p. 16: "for activations, we group and scale elements on a **1x128 tile** basis (i.e.,
  per token per 128 channels); and ... for weights ... on a **128x128 block** basis". Motivation: FP8
  training is "highly sensitive to activation outliers".
- S4.2 §3.3.2, p. 16: "per-group scaling factors along the inner dimension of GEMM operations. This
  functionality is not directly supported in the standard FP8 GEMM."
- S4.3 §3.3.2, p. 17 (Increasing Accumulation Precision): FP8 GEMM accumulation on NVIDIA H800 is
  "limited to retaining around **14 bits**"; "Taking GEMM operations of two random matrices with
  **K = 4096** ... the limited accumulation precision in Tensor Cores results in a maximum relative
  error of nearly **2%**." Fix: promotion to CUDA cores — partial results are copied to FP32 registers
  every `N_C = 128` elements, "equivalent to **4 WGMMAs**", the minimal interval that improves
  precision without substantial overhead. Two WGMMAs persist concurrently on H800, so one warpgroup
  promotes while the other issues MMA.
- S4.4 §3.3.2, p. 17: E4M3 is used on all tensors (not the E4M3/E5M2 split) because tile- and
  block-wise scaling shares exponent bits inside each group.
- S4.5 §3.3.2, p. 17-18 (Online Quantization): the max-abs value of each 1x128 activation tile or
  128x128 weight block is computed online, instead of a delayed per-tensor history.
- S4.6 §3.3.1, p. 16: relative loss error of FP8 training vs the BF16 baseline stays below **0.25%**
  (two model scales, ~1T tokens, Appendix B.1).
- S4.7 §3.3.2, p. 17: the fine-grained scheme is "highly consistent with the idea of microscaling
  formats", which Blackwell tensor cores support in hardware.

## S5. FP8 formats paper (Micikevicius et al., arXiv 2209.05433) — `$SCRATCH/research/30/fp8paper.txt`

- S5.1 Table 1: E4M3 — exponent bias 7, **max normal 1.75 x 2^8 = 448**, min normal 2^-6, min
  subnormal 2^-9, no infinities, one NaN pattern. E5M2 — max normal 57,344.
- S5.2 §3.1: reclaiming special-value patterns extends E4M3's dynamic range by one binade, "from 17 to
  18 binades".
- S5.3 §1/§2: "some networks require per-tensor scaling factors as FP8 dynamic range is" narrower than
  FP16 — i.e. per-tensor scaling is the prior art DeepSeek's per-group scheme replaces.

## S6. DeepGEMM README v1 (commit 8002b769, 2025-03-25) — measured Hopper numbers

Preamble: all shapes from DeepSeek-V3/R1 inference, **H800 SXM5, NVCC 12.8**, speedups "in comparison
to our internally and carefully optimized implementation based on CUTLASS 3.6".

| M | N | K | TFLOPS | memory BW | speedup |
|---|---|---|---|---|---|
| 64 | 2112 | 7168 | 206 | 1688 GB/s | **2.7x** |
| 64 | 7168 | 16384 | 336 | 2668 GB/s | 1.4x |
| 128 | 2112 | 7168 | 352 | 1509 GB/s | 2.4x |
| 128 | 7168 | 16384 | 645 | 2604 GB/s | 1.4x |
| 4096 | 2112 | 7168 | 1009 | 503 GB/s | 1.1x |
| 4096 | 7168 | 16384 | 1426 | 361 GB/s | 1.3x |

Also: "To address the imprecise FP8 tensor core accumulation, it employs CUDA-core two-level
accumulation (promotion)"; one core kernel of "~300 lines of code".

## S7. DeepGEMM PR #74 (merged 2025-03-25), H800-SXM, CUDA 12.8.1

Large block tiles relieve L2 pressure: 4096x7168x16384 1404 TF (128x128) → **1470 TF** (128x160);
4096x32768x512 591 → 748 TF (+26.57%); 4096x24576x1536 999 → 1166 TF (+16.72%).

## Numbers used on slides → source

| Slide | Number | Source |
|---|---|---|
| Context | 14.8T training tokens, FP8 linear layers | S4.6 / V3 abstract |
| Context | E4M3: 3 mantissa bits, max 448, min normal 2^-6, 18 binades | S5.1, S5.2 |
| Pain | ~14-bit tensor-core accumulation, ~2% max relative error at K=4096 | S4.3 |
| Status quo | per-tensor scaling vs 1x128 / 128x128 fine-grained scaling | S4.1, S5.3 |
| Status quo | per-group scaling along K unsupported in standard FP8 GEMM | S4.2 |
| Insight | SF must be an exact power of two; UE8M0 = 8-bit exponent | S2.2, S1.2 |
| ❶ recipe | SFA/SFB shapes, gran_k 32 or 128 (SM100), 128 (SM90) | S2.1 |
| ❷ promotion | promote every N_C = 128 elements = 4 WGMMAs | S4.3, S3.2 |
| ❷ SM100 | block-scaled UMMA reads SF from tensor memory; BLOCK_K 128 (FP8) / 256 (FP4) | S3.3, S2.10 |
| ❸ layout | int32 packed, MN-major, stride(-1) = align(mn, 4), packed_sf_k = ceil_div(k, gran_k*4) | S2.5, S2.7, S2.8 |
| dtype paths | SM90 NT-only FP32 SF; SM100 all majors, packed UE8M0; BF16 via cuBLASLt | S1.1, S3.1, S3.4 |
| Results | 206-1426 TFLOPS, 1.1-2.7x; peak 1550 TFLOPS | S6, S1.4 |
| Results | 2.6 TB/s memory bandwidth at M=64, N=7168, K=16384 (bandwidth-bound) | S6 |
| Accuracy | FP8 vs BF16 relative loss error < 0.25% | S4.6 |
| Limits | transform kernel per call unless SFs are pre-packed; user does the casting | S2.4, S1.3 |

## Discrepancies (Notion row vs primary sources)

1. **Notion Core Mechanism** says "SM90 supports only the NT layout with FP32 scale factors, while
   SM100 supports all layouts with 4x UE8M0 packed into an int" — matches S1.1/S1.2 exactly. It omits
   that SM90 has two kernels (1d1d when B is also per-token scaled, 1d2d for 128x128 block scales,
   S3.1) and that SM100 accepts gran_k 32 as well as 128 (S2.1). Slides state both.
2. **Notion Impact** "Peaked at 1550 TFLOPS on H800" matches the README news (S1.4). The README states
   this as a peak across shapes; the per-shape table (S6) tops out at 1426 TFLOPS, and PR #74 reports
   1470 TF for the same shape after the large-block-tile change (S7). The slides therefore quote 1426
   TFLOPS for the measured shape and 1550 TFLOPS as the published peak, with dates.
3. The Notion row's title includes BF16, but BF16 dense GEMM usually runs through **cuBLASLt** inside
   DeepGEMM unless determinism is requested or the SM budget is capped (S3.4) — stated on the slides.
4. No numeric conflict was found between the Notion row and the repository.
