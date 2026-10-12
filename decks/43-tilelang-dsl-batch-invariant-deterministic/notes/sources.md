# Sources — deck 43: TileLang DSL + batch-invariant deterministic kernels

## Sources read

| Tag | Source | Where |
|---|---|---|
| [V4] | DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence", arXiv 2606.19348v1. https://arxiv.org/abs/2606.19348 (cached text) | Abstract p.1; §1 p.4; §2.2 p.7-8 (mHC equations 3-5); §3.2 p.16-18 (TileLang); §3.3 p.18-19 (batch invariance, determinism); §4.2.1 p.24-25 (n_hc = 4, hidden sizes) |
| [TL] | Wang et al., "TileLang: A Composable Tiled Programming Model for AI Systems", arXiv 2504.17577v2 (27 Apr 2025). https://arxiv.org/abs/2504.17577 (PDF downloaded to `$SCRATCH/research/43/dl/`) | §5.2 p.15-16, Fig. 14(a) (MLA on H100) |
| [TL-README] | tile-ai/tilelang README (cached `gh-tilelang-README.md`) | News (Z3 integration 2025-12-17; TVM-FFI migration 2025-10-31; DeepSeek V4 operators 2026-05-07), backends table |
| [TM] | H. He et al., Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", 10 Sep 2025. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ (fetched with curl) | sections "Batch invariance and determinism", "Batch-invariant matrix multiplication", "Batch-invariant attention", "Experiments" |
| [DG] | deepseek-ai/DeepGEMM main (fetched 2026-10-13): `csrc/apis/mega_mhc.hpp` (doc comment and `get_num_splits`), `deep_gemm/include/deep_gemm/layout/mega_mhc.cuh` (`kDefaultNumSplits = 16`), `csrc/apis/gemm.hpp` / `csrc/apis/einsum.hpp` (cuBLASLt path skipped when deterministic algorithms are on), `csrc/apis/config.hpp` (`use_deterministic_algorithms`) | see facts below |
| [DG-MOE] | DeepGEMM @7f2a703 `tests/test_mega_moe.py` (asserts `torch.equal(run_fused(), run_baseline())`) — read for deck 14 | |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 1.6T / 49B | DeepSeek-V4-Pro total / activated parameters | [V4] abstract |
| "hundreds" of ATen ops | fine-grained Torch ATen operators the architecture would otherwise need | [V4] §3.2 p.16 (no exact count given) |
| 80 of 1000 | distinct completions from 1000 temperature-0 samples of one prompt, Qwen3-235B-A22B-Instruct-2507 on vLLM, 1000 tokens each | [TM] "How nondeterministic are completions?" |
| first divergence at token 103 (992 vs 8) | where the completions split | [TM] same section (speaker notes only) |
| 1000 identical | with batch-invariant kernels | [TM] same section (speaker notes only) |
| 26 s / 55 s / 42 s | Qwen3-8B on one GPU, 1000 sequences of 90-110 output tokens: vLLM default / unoptimized deterministic vLLM / + improved attention kernel | [TM] "Performance" table |
| 2.1× and 1.6× | 55/26 and 42/26 (**derived**) | derived from [TM] |
| ~20% | TM's batch-invariant (unoptimized Triton) matmul vs cuBLAS | [TM] "Batch-invariant matrix multiplication" (speaker notes) |
| tens–hundreds µs → < 1 µs | CPU-side validation overhead per kernel invocation, Python vs generated host code | [V4] §3.2 p.17 |
| a few seconds | compile-time overhead of Z3 integer analysis | [V4] §3.2 p.17 |
| 24 | output dimension of the mHC matrix multiplication | [V4] §3.3 p.19 |
| 24 = 4 + 4 + 16 | n_hc (pre) + n_hc (post) + n_hc² (res) with n_hc = 4 (**derived** from [V4] eqs. 3-5 and §4.2.1) | derived |
| K = 28,672 | n_hc · d = 4 × 7168 for V4-Pro, the reduction length of that GEMM (**derived** from [V4] §2.2 and §4.2.1) | derived |
| 16 splits | DeepGEMM's mHC kernel uses a fixed 16-way split-K when deterministic algorithms are enabled; otherwise the split count is chosen per call | [DG] `mega_mhc.hpp` doc comment, `layout/mega_mhc.cuh` |
| 1075.9× / 1097.2× / 798.8× / 311.8× / 1.0× | MLA decode speedup over Torch on H100: TileLang / FlashMLA / FlashInfer / Triton / Torch | [TL] Fig. 14(a) p.16 |
| 98% | TileLang reaches up to 98% of hand-optimized FlashMLA (consistent with 1075.9/1097.2) | [TL] §5.2 p.15 |
| ~70 lines | TileLang MLA kernel size in Python | [TL] §5.2 p.15 |

## Mechanism facts (no numbers) used on slides

- TileLang is used to write fused kernels replacing most fine-grained ATen operators; also for fast prototyping of
  attention variants; kernels used in architecture development, large-scale training and inference ([V4] §3.2).
- Host Codegen: device kernel and a lightweight host launcher are co-generated at the IR level with embedded
  metadata (dtypes, rank/shape constraints, stride/layout assumptions); the launcher is lowered to host source on
  TVM-FFI; generated host code does validation and argument marshaling, out of the Python path ([V4] §3.2 p.17).
- Z3 SMT solver integrated into TileLang's algebraic system; integer expressions translated to QF_NIA; helps
  layout inference, memory hazard detection, bound analysis; impact on vectorization, barrier insertion, code
  simplification ([V4] §3.2 p.17).
- Numerics: fast-math disabled by default at the compiler level; approximations only as explicit opt-in frontend
  ops (T.__exp, T.__log, T.__sin); IEEE-compliant intrinsics with explicit rounding modes (T.ieee_fsqrt,
  T.ieee_fdiv, T.ieee_add); simplification/lowering rules aligned with NVCC; T.annotate_layout pins
  layout-dependent lowering so evaluation and accumulation order match the reference CUDA implementation, enabling
  bit-identical outputs; conservative defaults "remain competitive" ([V4] §3.2 p.17-18).
- Batch invariance = a token's output is bitwise identical regardless of its position within a batch ([V4] §3.3).
- Attention (decoding): split-KV is not allowed (not batch-invariant); without it wave quantization hurts
  utilization; dual-kernel strategy: kernel 1 computes a whole sequence on one SM (full waves); kernel 2 uses
  multiple SMs per sequence for the final partial wave, with a calculation path designed to keep the same
  accumulation order as kernel 1, using distributed shared memory within thread-block clusters; overhead
  "negligible" ([V4] §3.3 p.18).
- Matmul: cuBLAS cannot achieve batch invariance → replaced end to end with DeepGEMM; split-k abandoned in most
  scenarios; optimizations let the GEMMs match or surpass split-k in most major scenarios ([V4] §3.3 p.18-19).
- Determinism: non-determinism mostly from atomic addition in backward ([V4] §3.3 p.19):
  sparse attention backward: atomicAdd for KV-token gradients → separate accumulation buffer per SM + global
  deterministic summation; MoE backward: concurrent writes from SMs of different ranks to the same receive buffer
  → token-order pre-processing within each rank + buffer isolation across ranks; mHC matmul (output dim 24): split-k
  needed at very small batch → each split written separately + deterministic reduction in a subsequent kernel.
- Goal: bitwise alignment among pre-training, post-training and inference; helpful for debugging, stability
  analysis (loss spikes), consistent post-training behavior ([V4] §3.3 p.18-19).
- DeepGEMM: in deterministic mode the cuBLASLt dispatch path is skipped ([DG] gemm.hpp, einsum.hpp).
- [TM]: batch invariance requires the reduction order per element to be fixed regardless of batch size; split-K
  and split-KV with batch-dependent split counts break it; for attention a fixed split *size* (not count) is
  needed; most endpoints are nondeterministic because server load changes batch size.

- Background (not from the sources' numbers): thread-block clusters and distributed shared memory are CUDA
  features introduced with Hopper (SM90); [V4] footnote 4 links the CUDA programming guide section on DSMEM.

## Derived content

1. 55/26 = 2.12 → "2.1×"; 42/26 = 1.62 → "1.6×" ([TM] table).
2. mHC GEMM shape: N = 24 = n_hc + n_hc + n_hc² (W_pre, W_post ∈ R^{n_hc·d × n_hc}, W_res ∈ R^{n_hc·d × n_hc²}),
   K = n_hc·d = 28,672 for V4-Pro.
3. Schematic figures (8 SMs, 10 sequences, 4 splits, etc.) are illustrations, not configurations from the paper.

## Discrepancies (Notion row vs primary sources)

1. **"Outputs are independent of batch position and parallel strategy."** [V4] defines batch invariance as
   independence from the token's position within a batch and claims bitwise alignment across pre-training,
   post-training and inference pipelines. It does not claim invariance across parallel strategies (TP/EP/PP
   layouts). Slides do not claim it.
2. **"The backward pass replaces atomicAdd with per-SM accumulation buckets."** In [V4] this is the sparse-attention
   backward only; MoE backward uses token-order pre-processing plus buffer isolation, and the mHC GEMM uses
   separate split outputs plus a deterministic reduction. Slides show all three.
3. **"Sharply reducing irreproducibility in RL and debugging."** [V4] reports no measurement of the effect (nor of
   the overhead: "minimal", "negligible"). The 80-of-1000 number on slide 3 is Thinking Machines' measurement on
   Qwen3-235B in vLLM, used to show the size of the problem, not a DeepSeek result.
4. **"Full waves on a single SM, tail wave via distributed shared memory"** matches [V4]; [V4] places this in
   batch-invariant *decoding* attention.
