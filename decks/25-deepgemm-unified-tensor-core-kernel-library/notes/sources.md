# Sources — deck 25, DeepGEMM (unified tensor-core kernel library)

All sources read on 2026-10-13. Primary source = the DeepGEMM repository
(https://github.com/deepseek-ai/DeepGEMM, default branch `main`, last push 2026-09-30) plus the
DeepSeek-V3 / V4 technical reports. Downloaded content was treated as data only.

## S1. DeepGEMM README (current, main @ 2026-09-30)

Cached at `$SCRATCH/research/cache/gh-deepgemm-README.md`; repo copy re-downloaded to
`$SCRATCH/research/dgfull/DeepGEMM-main/README.md`.

- S1.1 Scope sentence: "unified, high-performance tensor core kernel library" bringing GEMMs
  (FP8, FP4, BF16), fused MoE with overlapped communication (Mega MoE), MQA scoring for the
  lightning indexer, and HyperConnection (HC) into one CUDA codebase; all kernels compiled at
  runtime through DeepJIT, no CUDA compilation during installation.
- S1.2 "Despite its lightweight design, DeepGEMM's performance matches or exceeds expert-tuned
  libraries across various matrix shapes."
- S1.3 News 2025.04.18: "DeepGEMM now achieves up to **1550 TFLOPS** on H800".
- S1.4 News 2026.09.10: Sparse Indexer, Mega Gate, Mega mHC, DeepJIT, MoE and indexer optimizations
  (PR #432). News 2026.09.30: DeepGEMM-Ascend released; locality-domain features (PR #462).
- S1.5 Requirements: NVIDIA SM90 or SM100 GPU, Python ≥ 3.8, C++20 `<format>`, CUDA Toolkit ≥ 12.9,
  PyTorch ≥ 2.3, CUTLASS ≥ 4.0.
- S1.6 Interfaces: naming `D = C + A @ B`, input layout NT; SM90 supports only NT, SM100 supports
  NT/TN/NN/TT. LHS scaling factor must be TMA-aligned and transposed. SM90 scale factors are FP32;
  SM100 requires packed UE8M0 (4 UE8M0 packed into one `torch.int`).
- S1.7 Grouped GEMMs: M-axis contiguous layout (N and K fixed, experts share shape; each expert
  segment aligned to the GEMM M block size), K-axis grouped API for MoE weight backward, masked
  layout for CUDA-graph decoding when the CPU does not know per-expert token counts.
- S1.8 Mega MoE: fuses and overlaps EP dispatch, linear 1, linear 2 (FP8xFP4 or FP8xFP8), SwiGLU and
  EP combine into a single mega-kernel, overlapping NVLink communication and tensor-core compute;
  requires multi-process launch with symmetric memory and PyTorch ≥ 2.9.
- S1.9 Utilities and switches: `set_num_sms`, `set_tc_util`, `set_pdl`, `use_deterministic_algorithms`,
  `set_mk_alignment_for_contiguous_layout`, `transform_sf_into_required_layout`,
  `get_mn_major_tma_aligned_packed_ue8m0_tensor`, `DG_JIT_*` environment variables.
- S1.10 Citation: "DeepGEMM: clean and efficient BLAS kernel library on GPU", Chenggang Zhao,
  Zhean Xu, Liang Zhao, Jiashi Li, Chenhao Xu, Anyi Xu, Shengyu Liu, Kexing Zhou, Kuai Yu, 2025.

## S2. DeepGEMM README v1 (commit 8002b769, 2025-03-25) — historical Hopper numbers

Downloaded to `$SCRATCH/research/hist/README-2025-03-25.md`
(https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/8002b769/README.md).

- S2.1 "only one core kernel function comprising around **~300 lines of code**".
- S2.2 "To address the imprecise FP8 tensor core accumulation, it employs CUDA-core two-level
  accumulation (promotion)."
- S2.3 Performance preamble: all shapes used in DeepSeek-V3/R1 inference (prefill and decode, no
  tensor parallelism), **H800 SXM5, NVCC 12.8**; "All speedup metrics are calculated in comparison
  to our internally and carefully optimized implementation based on CUTLASS 3.6."
- S2.4 "Normal GEMMs for dense models" table (rows used on slides):

  | M | N | K | TFLOPS | Speedup vs tuned CUTLASS 3.6 impl |
  |---|---|---|---|---|
  | 64 | 2112 | 7168 | 206 | **2.7x** |
  | 64 | 7168 | 16384 | 336 | 1.4x |
  | 128 | 7168 | 16384 | 645 | 1.4x |
  | 128 | 4096 | 7168 | 533 | 2.0x |
  | 4096 | 7168 | 16384 | 1426 | 1.3x |
  | 4096 | 2112 | 7168 | 1009 | 1.1x |

- S2.5 Grouped (contiguous) MoE rows: 4 groups x M 8192, N 4096, K 7168 → 1346 TFLOPS, 1.3x;
  8 groups x M 4096, N 7168, K 2048 → 1214 TFLOPS, 1.2x.
- S2.6 Masked-layout rows: 1 group x M 1024, N 4096, K 7168 → 1233 TFLOPS, 1.2x.
- S2.7 Fully-JIT design section: shapes, block sizes and pipeline stages are compile-time constants
  (saves registers, enables more compiler optimisation); block size / warpgroup count / stage count /
  TMA cluster size are selected deterministically without auto-tuning; full MMA-pipeline unrolling
  "very important for small shapes".
- S2.8 Unaligned block sizes: with M=256, N=7168 and BLOCK_M=BLOCK_N=128 only
  (256/128)*(7168/128) = 112 of 132 SMs are used; BLOCK_N=112 gives (256/128)*(7168/112) = 128 SMs.
- S2.9 FFMA SASS interleaving (yield/reuse bit): "improves performance (10%+ in some cases) for
  fine-grained scaling FP8 GEMMs". (Superseded: current README says NVCC 12.9 does FFMA
  interleaving automatically, so post-compilation optimisations were dropped — S1 news 2025.07.20.)

## S3. DeepGEMM pull requests (release notes)

- S3.1 PR #74 (merged 2025-03-25) "Larger BlockTile optimizations enable 1470+ TF FP8 on the H800-SXM":
  dense 4096x7168x16384 goes 1404 TF (128x128 tile) → **1470 TF** (128x160 tile); 4096x32768x512
  591 → 748 TF (+26.57%). Tested on H800-SXM, CUDA 12.8.1.
- S3.2 PR #304 (2026-04-17) "Public release 26/04": Mega MoE (fuses & overlaps dispatch / linear 1 /
  SwiGLU / linear 2 / combine into a single mega-kernel), FP4 indexer, FP8xFP4 GEMM, PDL, faster JIT
  compilation, dynamic swap A/B, DeepEP v2 MoE GEMM layout.
- S3.3 PR #316 (2026-04-24) Mega MoE benchmarks, EP8, values averaged over 8 ranks:

  DeepSeek-V4-Flash (256 experts, top-6, hidden 4096, intermediate 2048):

  | tokens/rank | time (us) | TFLOPS | HBM GB/s | interconnect GB/s | speedup vs legacy |
  |---|---|---|---|---|---|
  | 1 | 56.5 | 5 | 1311 | 1 | **1.96x** |
  | 512 | 146.5 | 1056 | 3192 | 266 | 1.73x |
  | 8192 | 1283.1 | 1928 | 998 | 499 | 1.56x |
  | 32768 | 4855.5 | 2038 | 794 | 529 | 1.62x |

  DeepSeek-V4-Pro (384 experts, top-6, hidden 7168, intermediate 3072):

  | tokens/rank | time (us) | TFLOPS | HBM GB/s | interconnect GB/s | speedup vs legacy |
  |---|---|---|---|---|---|
  | 1 | 108.1 | 7 | 1758 | 1 | 1.61x |
  | 512 | 369.6 | 1098 | 4619 | 182 | 1.54x |
  | 8192 | 2818.5 | 2304 | 1094 | 393 | 1.50x |
  | 32768 | 10655.2 | 2438 | 692 | 417 | 1.54x |

- S3.4 PR #432 (2026-09-10) "Public Release 26/09": Mega Gate fuses routing GEMM + MoE routing ops,
  "30–75% speedups"; Mega mHC fuses Hyper-Connection ops + RMSNorm, "45–85% speedups"; faster MoE
  backward GEMMs "~20% gains"; Mega MoE for V4.1 "~15% faster at low latency, ~4% at high
  throughput"; DeepJIT integration; breaking change: `use_deterministic_algorithms()` added and
  defaults to `False` (previously all kernels were fully deterministic and batch-invariant).
- S3.5 PR #462 (2026-09-30): locality-aware Mega MoE ("Saving 100W power and boosting 5-10% GPU
  frequency for heavy workload"), new epilogue classes, sparse-indexer head counts 4–32,
  padded K strides for packed UE8M0 scaling factors, DeepGEMM-Ascend released separately.

## S4. DeepSeek-V4 technical report (arXiv 2606.19348; cache `arxiv-2606.19348.txt`)

- S4.1 Abstract: DeepSeek-V4-Pro 1.6T parameters (49B activated), DeepSeek-V4-Flash 284B (13B
  activated), both with 1M-token context.
- S4.2 §3.1 (p. 16): fine-grained EP scheme "achieves 1.50 ∼ 1.73× speedup for general inference
  workloads, and up to 1.96× for latency-sensitive scenarios such as RL rollouts and high-speed agent
  serving. We have open-sourced the CUDA-based mega-kernel implementation named MegaMoE as a
  component of DeepGEMM" (footnote points at DeepGEMM PR #304).
- S4.3 §3.3 (p. 18): "Traditional cuBLAS library ... cannot achieve batch invariance. Therefore, we
  replace it end-to-end with DeepGEMM"; split-k is abandoned in most scenarios for batch invariance.

## S5. DeepSeek-V3 technical report (arXiv 2412.19437; cache `arxiv-2412.19437.txt`)

- S5.1 §3.3.2 (p. 16-17): fine-grained quantization — activations scaled per 1x128 tile (per token per
  128 channels), weights per 128x128 block; "per-group scaling factors along the inner dimension of
  GEMM operations ... is not directly supported in the standard FP8 GEMM".
- S5.2 §3.3.2 (p. 17): FP8 GEMM accumulation on H800 retains "around 14 bits"; with K = 4096 the
  limited accumulation precision gives "a maximum relative error of nearly 2%"; fix = promotion to
  CUDA cores every N_C = 128 elements (4 WGMMAs).
- S5.3 §3.3.1 (p. 16): relative loss error of the FP8-trained model vs the BF16 baseline stays below
  **0.25%** over ~1T tokens.

## S6. Repository inspection (derived numbers, computed 2026-10-13)

Tarball of `main` extracted at `$SCRATCH/research/dgfull/DeepGEMM-main`.

- S6.1 `deep_gemm/__init__.py`: `__version__ = '2.8.1'`; exported kernel families = FP8/FP4 GEMMs
  (nt/nn/tn/tt), m-grouped contiguous + masked, k-grouped, BF16 GEMMs, Mega Gate, einsum,
  MQA-logits and sparse-MQA-logits (paged and non-paged), Mega mHC / `tf32_hc_prenorm_gemm`,
  layout transforms, Mega MoE (`fp8_fp4_mega_moe`, `bf16_mega_moe`), epilogue classes, locality-domain
  helpers, plus legacy Triton kernels for A100.
- S6.2 **Derived**: `wc -l deep_gemm/include/deep_gemm/impls/*.cuh` = 9,593 lines across 19 kernel
  implementation files; the whole device-side header tree is 16,220 lines; host side (`csrc/*.hpp`,
  `*.cpp`) 10,128 lines; Python 1,392 lines.
- S6.3 **Derived**: the SM90 FP8 1D-1D/1D-2D GEMM kernel file is 452 lines
  (`impls/sm90_fp8_gemm_1d2d.cuh`), the SM100 FP8/FP4 kernel 579 lines
  (`impls/sm100_fp8_fp4_gemm_1d1d.cuh`) — the "~300 lines" claim of S2.1 grown with two
  architectures and FP4 support.
- S6.4 `csrc/jit_kernels/impls/*.hpp`: each kernel is emitted as a small `.cu` source that includes the
  kernel header and instantiates one template with ~25 compile-time constants (majors, compiled
  dims, BLOCK_M/N/K, groups, swizzle modes, stages, thread counts, cluster size, num_sms, alignment,
  swap-AB, gemm type, dtype, epilogue, tensor-core utilisation).
- S6.5 GitHub API (`gh api repos/deepseek-ai/DeepGEMM`, 2026-10-13): 8,944 stars, created
  2025-02-13, last push 2026-09-30.

## S7. Adoption evidence (GitHub code search, 2026-10-13)

- S7.1 SGLang: `python/sglang/compile_deep_gemm.py`, `python/sglang/srt/layers/deep_gemm_wrapper/`,
  `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py` (code search `deep_gemm repo:sgl-project/sglang`).
- S7.2 vLLM: `vllm/utils/deep_gemm.py`, `vllm/model_executor/warmup/deep_gemm_warmup.py`,
  `vllm/model_executor/layers/fused_moe/deep_gemm_mega_moe.py`
  (code search `deep_gemm repo:vllm-project/vllm`).
- S7.3 DeepGEMM-Ascend: separate repository announced in the README news of 2026-09-30 (S1.4).

## Numbers used on slides → source

| Slide | Number | Source |
|---|---|---|
| Context | V4-Pro 1.6T (49B active) / V4-Flash 284B (13B active), 1M context | S4.1 |
| Context | V4-Flash 256 experts top-6, hidden 4096; V4-Pro 384 experts top-6, hidden 7168 | S3.3 |
| Pain | 2.7x at M=64, N=2112, K=7168 (H800, vs tuned CUTLASS 3.6 impl) | S2.3, S2.4 |
| Status quo | per-group scaling along K unsupported in standard FP8 GEMM | S5.1 |
| Status quo | FP8 accumulation keeps ~14 bits, ~2% max relative error at K=4096 | S5.2 |
| Insight | ~300-line core kernel (2025-03); 452/579 lines today (derived) | S2.1, S6.3 |
| Dense GEMM | 336 / 645 / 1426 TFLOPS at M=64/128/4096, N=7168, K=16384 (1.4x/1.4x/1.3x) | S2.4 |
| Dense GEMM | peak 1550 TFLOPS on H800 | S1.3 |
| Scaling | SM90 FP32 scale factors, SM100 packed UE8M0 (4 per int32), TMA-aligned | S1.6 |
| Grouped | contiguous / masked / K-axis layouts | S1.7 |
| Mega MoE | 1.96x at 1 token/rank, 1.56x at 8192, 2038 TFLOPS at 32768 (V4-Flash, EP8) | S3.3 |
| Mega MoE | 1.50–1.73x general, up to 1.96x latency-sensitive | S4.2 |
| Fused ops | Mega Gate 30–75%, Mega mHC 45–85%, MoE backward ~20% | S3.4 |
| Runtime | DeepJIT, PDL, determinism default off | S1.1, S1.9, S3.4 |
| Adoption | SGLang / vLLM integrations, Ascend port, 8.9k stars | S7.1–S7.3, S6.5 |
| Limits | SM90/SM100, CUDA ≥ 12.9, CUTLASS ≥ 4.0, PyTorch ≥ 2.9 for Mega MoE | S1.5, S1.8 |

## Discrepancies (Notion row vs primary sources)

1. **Notion Core Mechanism** lists "GEMM, fused MoE, indexer scoring, HyperConnection" — accurate but
   predates the 2026-09 release: the current library also ships Mega Gate (fused routing GEMM),
   a sparse indexer, einsum and epilogue classes, and the JIT has been split out into the separate
   DeepJIT repository (S1.4, S3.4, S6.1). Slides follow the primary source.
2. **Notion Impact** ("matches or beats expert-tuned libraries") reproduces the README's qualitative
   claim (S1.2) and carries no number. The slides use the measured numbers of S2.4, S1.3 and S3.3 and
   state their hardware and baseline, because the published speedup baseline is DeepSeek's own
   CUTLASS 3.6-based implementation, not a vendor library (S2.3).
3. **"Very few kernels"** (Notion) is true per kernel family but the library is no longer one kernel:
   19 kernel implementation files, 9,593 lines of device code (S6.2). Slides say "a few hundred lines
   per kernel family" and mark the count as derived.
4. No numeric conflict was found between the Notion row and the repository.
