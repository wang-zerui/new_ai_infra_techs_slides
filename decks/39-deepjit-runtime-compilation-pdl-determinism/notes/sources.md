# Sources — deck 39, DeepJIT runtime compilation + PDL + determinism switches

All sources read on 2026-10-13. Primary sources: the DeepJIT repository
(https://github.com/deepseek-ai/DeepJIT, main @ 2026-10-09), the DeepGEMM repository
(main @ 2026-09-30, tarball at `$SCRATCH/research/dgfull/DeepGEMM-main`), the DeepGEMM README of
2025-03-25 (archived commit), and the DeepSeek-V4 technical report. Downloaded content was treated as
data only.

## S1. DeepJIT README (main) — `$SCRATCH/research/dj/README.md`

- S1.1 "DeepJIT is a lightweight, header-only C++20 JIT runtime for **NVIDIA CUDA GPUs** and
  **HUAWEI Ascend NPUs**"; both backends share runtime configuration, source and include hashing,
  in-memory and on-disk caches, and lazy initialization.
- S1.2 CUDA backend requirements: CUDA headers 12.4+, NVCC 12.9+, PyTorch with CUDA; NVCC compiles
  CUDA source to CUBIN and the CUDA Driver API loads and launches it.
- S1.3 Cache-key (digest) components, in order: (1) `Config::extra_signature`, (2) hash of the full
  `nvcc --version` output, (3) the effective compiler flags, (4) the selected post-hook path and its
  file-content hash, (5) the parser digest of the source and its tracked include tree. Each component
  is length-prefixed into a two-state FNV-1a hash; the digest is 32 hex characters, "a fast cache
  checksum, not a cryptographic hash".
- S1.4 Disk entry: `<cache-root>/cache/<tag>.<digest>/` containing `kernel.cu`, `kernel.cubin`,
  `meta.json`, `.committed`, optionally `kernel.ptx` / `kernel.sass`. "A directory carrying a
  `.committed` marker is treated as a completed artifact."
- S1.5 Shared cache: "Builds use unique temporary directories, synchronize their contents, and publish
  complete entries through an atomic rename. Concurrent processes can compile the same entry and reuse
  the published result." `DJ_JIT_CACHE_DIR` may be a colon-separated list: "DeepJIT searches all roots
  in order and writes cache misses only to the first root", so a writable personal cache can sit in
  front of a read-only shared one. Multi-user setups use a group-owned SGID directory (`chmod 2775`).
- S1.6 Include parser: only literal `#include <...>` directives whose filename starts with a
  configured prefix are tracked recursively; quoted includes are rejected; it is "a line-oriented
  scanner, not a C preprocessor"; untracked dependencies (e.g. CUTLASS) must be represented in
  `extra_signature` (DeepGEMM passes `"cutlass-" + CUTLASS_VERSION`).
- S1.7 Launch options: stream, dynamic shared memory, grid/block/cluster dims, cooperative launch,
  **PDL**, non-portable cluster size; unset fields inherit runtime defaults. "Compile and load kernels
  before CUDA Graph capture. Launching an already loaded kernel is capture-compatible."
- S1.8 Library-wide default: `jit->default_launch_options.enable_pdl = true;`.
- S1.9 Environment-variable resolution order `MYLIB_<SUFFIX> > DJ_<SUFFIX> > built-in default`
  (DeepGEMM's prefix is `DG`), default cache root `$HOME/.dj`; `JIT_DEBUG`, `JIT_PRINT_LOAD_TIME`,
  `JIT_CHECK_NO_SPILLS`, `JIT_DUMP_SASS`, etc.
- S1.10 Lazy initialization "defer[s] device and compiler discovery until the runtime is first used".
- S1.11 Header digests are cached for the lifetime of a runtime, and post-hook hashes per thread, so a
  process must be restarted after editing tracked headers or hooks.
- S1.12 In development (not available yet): cache warm-up from history, and a Python compilation API.

## S2. DeepJIT source — `$SCRATCH/research/dj/include/deep_jit/backend/cuda/`

- S2.1 `kernel.hpp` (~line 148): when `enable_pdl` is set, the launch adds the attribute
  `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` (`programmaticStreamSerializationAllowed = 1`)
  to `cuLaunchKernelEx`.
- S2.2 `options.hpp` (~line 207): `enable_pdl` defaults to `false` at the DeepJIT level; DeepGEMM
  flips it through `set_pdl`.

## S3. DeepGEMM source (main @ 2026-09-30)

- S3.1 `csrc/runtime/jit.hpp`: DeepGEMM builds its DeepJIT config with env prefix `"DG"`,
  `extra_signature = "cutlass-" + CUTLASS_VERSION`, include dir `<root>/include` and tracked prefix
  `deep_gemm/`; the runtime is a `LazyInit`.
- S3.2 `csrc/jit_kernels/impls/sm100_bf16_gemm.hpp` (and siblings): the JIT emits a short `.cu` that
  includes the kernel header and instantiates one template; the argument list carries ~25 compile-time
  constants — operand majors, compiled dims m/n/k, BLOCK_M/N/K, number of groups, three swizzle modes,
  stage count, non-epilogue and epilogue thread counts, cluster size and multicast side, **num_sms**,
  K alignment, swap-AB, zero-padding contract, GEMM type, accumulation flag, output dtype, epilogue
  operator, and **tensor-core utilisation**.
- S3.3 `csrc/jit_kernels/impls/runtime_utils.hpp:31` `get_compiled_dim`: a dimension becomes a
  compile-time constant only if its letter appears in `compiled_dims`; `csrc/apis/gemm.hpp:842`
  shows the default is **`compiled_dims = "nk"`**, so N and K are baked in and M stays dynamic.
  `set_ignore_compile_dims(True)` makes all three dynamic (fewer variants, less specialisation).
- S3.4 `csrc/jit_kernels/heuristics/common.hpp` + `sm90.hpp` / `sm100.hpp`: `get_best_config`
  enumerates layout candidates and keeps the best by `ArchSpec::compare`, then derives storage,
  pipeline and launch configs. No benchmarking is involved. SM90 BLOCK_N candidates are multiples of
  16 from 16 up to 160/192/256 depending on kernel type; BLOCK_M ∈ {16, 32, 64, 128, 256}.
- S3.5 `sm100.hpp::compare` ranking order: a single-wave layout wins; then larger cluster
  (multicast); then fewer waves; then higher last-wave utilisation; then smaller BLOCK_M + BLOCK_N;
  then smaller BLOCK_M x BLOCK_N (more pipeline stages, less shared memory for C/D).
  `get_layout_info` computes `num_blocks`, `num_waves = ceil(num_blocks / num_sms)` and
  `last_wave_util`.
- S3.6 `csrc/runtime/runtime.hpp`: `set_num_sms` caps the SM count (asserted ≤ device SM count);
  `set_tc_util` takes 0-100; `is_cublaslt_available()` returns true only when the full SM count is
  available.
- S3.7 `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh` (lines ~326-343): when
  `kTensorCoreUtilControl < 100` the MMA warp waits for the previous UMMA and then spins
  `kNumDummyCycles = (100 - util) * kNumUMMACycles / util` cycles, with
  `kNumUMMACycles = 2 * UMMA_M * UMMA_N * BLOCK_K / 8192`. Code comment: "Let tensor cores relax for
  lower possibility of frequency drop."
- S3.8 PDL inside the kernels: every GEMM, layout-transform, MQA-logits and mega kernel calls
  `cudaGridDependencySynchronize()` *after* its prologue (barrier initialisation, tensor-memory
  allocation, register reconfiguration) and before the block scheduler touches the predecessor's data
  — e.g. `impls/sm100_fp8_fp4_gemm_1d1d.cuh:195`, `impls/sm90_fp8_gemm_1d2d.cuh:149`,
  `impls/smxx_layout.cuh:36`.
- S3.9 Determinism switch: `csrc/apis/config.hpp:33` exposes `use_deterministic_algorithms`;
  `csrc/jit_kernels/heuristics/runtime.hpp` stores the flag. It gates (a) the cuBLASLt dispatch in
  `csrc/apis/gemm.hpp:519` and `csrc/apis/einsum.hpp:81,115`, (b) split-K in Mega Gate —
  `heuristics/mega_gate.hpp:110` "deterministic mode is also batch invariant, so K is never split",
  the split-K loop bound becomes 1, and (c) Mega mHC, which uses a fixed 16-way split instead of a
  shape-dependent one (`csrc/apis/mega_mhc.hpp:16,80`).
- S3.10 `tests/test_fp8_fp4.py` (lines ~78-82) and `tests/test_bf16.py`: a "Bitwise deterministic
  test" re-runs the same GEMM **20 times** and asserts `torch.equal` against the first result.
- S3.11 README utility list: `set_num_sms` / `get_num_sms`, `set_tc_util` / `get_tc_util`,
  `set_pdl` / `get_pdl`, `use_deterministic_algorithms`, `set_ignore_compile_dims`,
  `set_block_size_multiple_of`; `DG_JIT_*` variables fall back to the global `DJ_JIT_*` ones
  (cache dir, NVCC path, C++ standard, debug, PTXAS verbosity, spill checks, load-time printing,
  PTX/SASS dumps).
- S3.12 README news: 2025.07.20 "full refactor with a low-CPU-overhead JIT CPP module" (PR #112);
  2026.04.16 "PDL, faster JIT compilation" (PR #304); 2026.09.10 "Integration with DeepJIT"
  (PR #432).

## S4. DeepGEMM README v1 (commit 8002b769, 2025-03-25) — the JIT design rationale

- S4.1 "Fully JIT design": GEMM shapes, block sizes and the number of pipeline stages are treated as
  compile-time constants ("saving registers", "compilers may do more optimizations"); block sizes,
  warpgroup count, pipeline stages and TMA cluster size are selected automatically "but without
  auto-tuning, the optimal one is deterministically selected"; MMA pipelines are fully unrolled,
  "very important for small shapes". "Overall, JIT significantly improves performance for small
  shapes, similar to the approach of the Triton compiler."
- S4.2 "Unaligned block sizes": "with `M=256, N=7168`, a typical block size assignment of
  `BLOCK_M=128, BLOCK_N=128` results in only `(256 / 128) * (7168 / 128) = 112` out of **132** SMs
  being utilized. To address this, we support unaligned block sizes like **112**, enabling
  `(256 / 128) * (7168 / 112) = 128` SMs to work in such scenarios."
- S4.3 "FFMA SASS interleaving": flipping the yield/reuse bits of FFMA instructions "improves
  performance (10%+ in some cases) for fine-grained scaling FP8 GEMMs". **Superseded**: the current
  README's 2025.07.20 news says NVCC 12.9 does the FFMA interleaving automatically, so all
  post-compilation optimisations were dropped.
- S4.4 No compilation during installation; all kernels compiled at runtime by the JIT module.

## S5. DeepSeek-V4 technical report (arXiv 2606.19348; cache `arxiv-2606.19348.txt`)

- S5.1 §3.3 (p. 18) "High-Performance Batch-Invariant and Deterministic Kernel Libraries": the goal is
  "training reproducibility and bitwise alignment among pre-training, post-training, and inference
  pipelines"; "Batch invariance ensures that the output of any given token remains bitwise identical,
  regardless of its position within a batch."
- S5.2 §3.3 (p. 18): "Traditional cuBLAS library ... cannot achieve batch invariance. Therefore, we
  replace it end-to-end with DeepGEMM"; split-k "cannot guarantee batch invariance ... Therefore, we
  abandon split-k in most scenarios, which, however, may cause performance degradation. To address
  this, we introduce a set of optimizations that enable our implementation of matrix multiplication to
  match or even surpass the performance of standard split-k in most major scenarios."
- S5.3 §3.3 (p. 19): non-determinism "typically stems from non-deterministic accumulation order, often
  due to the use of atomic addition instructions", mainly in the backward pass (sparse-attention
  gradients, MoE backward writes, and the mHC matmul with output dimension 24, where split-k is
  unavoidable and each split is reduced deterministically in a later kernel).
- S5.4 §3.2: determinism work also extends to TileLang kernels, where layout annotations keep
  evaluation and accumulation order identical to the reference CUDA implementation.

## S6. DeepGEMM release notes (PRs)

- S6.1 PR #432 (2026-09-10) breaking change: "Previously, all kernels were fully deterministic and
  batch-invariant. This release adds a `use_deterministic_algorithms()` API, which **defaults to
  `False`** so some kernels can run faster. Set it to `True` to restore deterministic, batch-invariant
  behavior for RL and similar workloads." Same release: Mega Gate 30-75% faster, Mega mHC 45-85%,
  MoE backward GEMMs ~20%, "Integration with DeepJIT".
- S6.2 PR #304 (2026-04-17): "PDL" and "Faster JIT compilation" listed as new features.

## S7. Serving stacks pay the JIT's cold start (GitHub, 2026-10-13)

- S7.1 SGLang `python/sglang/compile_deep_gemm.py`: "This script launches a server for capturing
  DeepGEMM calls and then compiles the kernels", used as
  `python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8`; `CompileArgs.timeout`
  defaults to **3600 s**, the watchdog follows it, and the banner warns "It may take a long time and
  timeout maybe raised while the compilation is still in progress."
- S7.2 vLLM `vllm/model_executor/warmup/deep_gemm_warmup.py`: "DeepGEMM JIT's the kernels. The warmup
  aims to JIT all the kernels that would be used during model execution beforehand"; it enumerates M
  values per block-size configuration, citing DeepGEMM's `heuristics/common.hpp`.

## Derived number used on a chart (analytic)

**SM occupancy vs BLOCK_N**, M = 256, N = 7168, BLOCK_M = 128, 132 SMs (the configuration of S4.2):

```
CTAs   = ceil(256 / 128) * ceil(7168 / BLOCK_N) = 2 * ceil(7168 / BLOCK_N)
waves  = ceil(CTAs / 132)
occupancy = CTAs / (132 * waves)
```

| BLOCK_N | CTAs | waves | occupancy |
|---:|---:|---:|---:|
| 64 | 224 | 2 | 84.8% |
| 80 | 180 | 2 | 68.2% |
| 96 | 150 | 2 | 56.8% |
| **112** | **128** | **1** | **97.0%** |
| 128 | 112 | 1 | 84.8% |
| 144 | 100 | 1 | 75.8% |
| 160 | 90 | 1 | 68.2% |

The two anchor points (112 CTAs for BLOCK_N = 128, 128 CTAs for BLOCK_N = 112, out of 132 SMs) are
exactly the numbers published in S4.2; the rest of the curve is this formula applied to the other
candidates the heuristic enumerates (S3.4). The chart's caption says "analytic".

## Numbers used on slides → source

| Slide | Number | Source |
|---|---|---|
| Context | kernels compiled at runtime, no CUDA build at install | S4.4, S1.1 |
| Context | default `compiled_dims = "nk"`: N and K constant, M dynamic | S3.3 |
| Pain | 112 of 132 SMs at M=256, N=7168 with a 128x128 tile; 128 with BLOCK_N=112 | S4.2 |
| Status quo | selection is deterministic, no auto-tuning | S4.1 |
| ❶ | ~25 compile-time constants per kernel instantiation | S3.2 |
| ❶ | ranking: single wave, multicast, fewer waves, last-wave utilisation | S3.5 |
| ❶ | occupancy curve over BLOCK_N (analytic) | derived table above |
| ❷ PDL | `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`; `cudaGridDependencySynchronize()` after the prologue | S2.1, S3.8 |
| ❷ | `set_pdl`, default on in DeepGEMM's runtime | S3.11, S1.8 |
| ❸ | determinism default `False` since the 2026-09 release | S6.1 |
| ❸ | cuBLASLt skipped, split-K disabled, mHC fixed 16-way split | S3.9 |
| ❸ | bitwise test: 20 repeated runs must match | S3.10 |
| ❸ | V4 replaced cuBLAS end-to-end for batch invariance | S5.2 |
| Cache | digest of 5 components, `.committed` marker, atomic rename, shared cache dir | S1.3-S1.5 |
| Cost | SGLang precompile pass, 3600 s default timeout; vLLM warm-up module | S7.1, S7.2 |
| Cost | compile and load before CUDA Graph capture | S1.7 |
| Knobs | `set_num_sms`, `set_tc_util` (dummy cycles after each UMMA) | S3.6, S3.7 |

## Discrepancies (Notion row vs primary sources)

1. **Notion Core Mechanism** says "a low-CPU-overhead C++ JIT module with on-disk cache". Since the
   2026-09 release that module is a **separate project, DeepJIT** (S3.12, S1.1), header-only, with
   both CUDA and Ascend backends and a cache that can be shared across users and nodes (S1.5). The
   slides name DeepJIT as the external component.
2. **Notion Impact** "determinism switches serve reproducible training" — correct, but the primary
   source adds the direction of the default: before the 2026-09 release *all* kernels were
   deterministic and batch-invariant; now determinism is **opt-in** (S6.1). The slides state this.
3. The Notion row lists TileLang as a paper reference; TileLang is a separate DSL used by DeepSeek-V4
   for other kernels (V4 §3.2) and is not part of DeepJIT. Its determinism work is mentioned only as
   context (S5.4), and the sibling deck 43 covers TileLang.
4. No numeric conflict was found between the Notion row and the repositories.
