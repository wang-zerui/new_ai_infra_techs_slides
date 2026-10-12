# Story — deck 43: TileLang DSL + batch-invariant deterministic kernels

- **Where it lives** (prosperity): DeepSeek-V4 (1.6T/49B) with hybrid CSA/HCA attention, mHC and Muon; the same
  kernels serve pre-training, post-training (RL rollout + trainer) and inference. The architecture would have
  needed hundreds of fine-grained ATen ops; DeepSeek writes fused TileLang kernels instead, plus DeepGEMM for GEMMs.
- **Pain number** (betrayal): the same temperature-0 request sent 1000 times returns **80 distinct completions**
  (Qwen3-235B on vLLM, Thinking Machines). Cause: the output of a token depends on what else is in the batch.
  For RL this turns on-policy into off-policy; for debugging a loss spike, a rerun is not the same run.
- **Failed suitors**:
  1. Fast kernels (cuBLAS, split-K GEMM, split-KV / FlashDecoding, atomicAdd backward): pick their reduction
     strategy from the batch shape or let timing pick the order.
  2. Naive invariant kernels (one fixed config, no split): correct but slow: 26 s → 55 s in TM's vLLM test
     (2.1×, derived), 42 s with a better attention kernel.
  3. Hand-writing hundreds of exact CUDA kernels: too slow to iterate; Python glue costs tens–hundreds of µs per
     launch in host checks.
- **Villain**: floating-point addition is not associative, and the order of additions is chosen by runtime
  facts (batch size via split heuristics, thread timing via atomics) instead of by the token.
- **Turn / insight**: bits depend only on accumulation order; fix the order per token, then win back parallelism
  in ways that keep that order (spread a tail sequence across SMs in the same order; give each SM its own buffer
  and sum the buffers in a fixed order).
- **Mechanisms**:
  ❶ TileLang: one DSL with co-generated host launchers (<1 µs checks), Z3-assisted integer analysis, exact
     numerics by default and pinned layouts/accumulation order (answers: hundreds of kernels, host overhead,
     bit-identity against CUDA references).
  ❷ Batch-invariant forward: dual-kernel decode attention (full waves 1 SM/sequence; tail wave multi-SM via DSMEM,
     same order); GEMMs on DeepGEMM end to end, no split-K, fixed 16-way split where unavoidable (mHC, N = 24)
     (answers: split-KV / split-K / cuBLAS heuristics).
  ❸ Deterministic backward: per-SM buffers + fixed-order sum for sparse-attention dKV; token-order pre-processing +
     buffer isolation for MoE backward; separate split outputs + deterministic reduce for mHC (answers: atomics).
- **Release**: bitwise batch-invariant and deterministic kernels end to end in V4, with "minimal" overhead per
  DeepSeek (no number published); TileLang kernels within 98% of hand-written FlashMLA at ~70 lines; host checks
  < 1 µs.
- **Limits**: no published overhead or reproducibility measurement; split-K abandoned in most cases (needs other
  GEMM optimizations); cuBLAS replaced end to end; tail kernel needs thread-block clusters and DSMEM; invariance is
  across batch position, not across parallel layouts; Z3 adds a few seconds of compile time.

## Slide plan (15 logical slides)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Pre-training, RL and serving must agree bitwise | prosperity | fig_context |
| 3 | Same request, 80 different answers | pain | `%% big: 80 / 1000` |
| 4 | Reduction order follows the batch, not the token | villain | fig_order |
| 5 | Naive batch invariance costs up to 2.1× | failed suitors | chart tm_cost |
| 6 | Insight: fix the order, then add parallelism back | turn | text + build |
| 7 | Three layers make V4's kernels reproducible | overview | fig_stack |
| 8 | ❶ TileLang generates the host launcher too | mechanism | fig_codegen |
| 9 | ❶ TileLang keeps numerics exact by default | mechanism | two-col text |
| 10 | ❷ Decode attention splits only the tail wave | mechanism | fig_dualkernel |
| 11 | ❷ GEMMs keep one K order at every batch | mechanism | fig_gemm |
| 12 | ❸ Backward sums in a fixed order, not atomically | mechanism | fig_atomic |
| 13 | TileLang reaches 98% of FlashMLA in ~70 lines | result | chart mla |
| 14 | The cost: own kernels, lost split-K, no numbers | limits | two-col text |
| 15 | Summary | | QR |
