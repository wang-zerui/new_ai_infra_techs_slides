# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Kernels written in the TileLang domain-specific language replace hundreds of fine-grained ATen ops; batch-invariant attention uses two kernels (full waves on a single SM, tail wave via distributed shared memory) to preserve accumulation order; the backward pass replaces atomicAdd with per-SM accumulation buckets and deterministic reduction.
**After:** Fused kernels written in the TileLang DSL replace most of the hundreds of fine-grained ATen ops; batch-invariant decoding attention forbids split-KV and uses two kernels (full waves on a single SM, tail wave spread over SMs via distributed shared memory, same accumulation order); DeepGEMM replaces cuBLAS and drops split-k in most GEMMs; deterministic backward: per-SM buffers plus deterministic summation for sparse-attention KV gradients, token-order pre-processing plus per-rank buffer isolation for MoE, separate split outputs plus deterministic reduction for the mHC GEMM.
**Why:**
- TileLang kernels replace "most" of the fine-grained ATen operators the architecture would need; the paper gives no exact count ([V4] §3.2 p.16).
- The dual-kernel strategy is for batch-invariant decoding attention and exists because split-KV is not batch-invariant ([V4] §3.3 p.18).
- Batch-invariant GEMMs come from replacing cuBLAS with DeepGEMM and abandoning split-k in most scenarios ([V4] §3.3 p.18-19).
- Per-SM accumulation buffers apply only to the sparse-attention backward; MoE backward and the mHC GEMM use different fixes ([V4] §3.3 p.19).

## Impact / Measured Results
**Before:** Outputs are independent of batch position and parallel strategy and consistent between training and inference, sharply reducing irreproducibility in RL and debugging.
**After:** A token's output is bitwise identical regardless of its position in the batch, aiming at bitwise alignment among pre-training, post-training and inference; V4 names debugging and loss-spike analysis as benefits but reports no measurement of the effect or of the overhead.
**Why:**
- [V4] defines batch invariance as independence from the token's position within a batch; it makes no claim about invariance across parallel strategies ([V4] §3.3 p.18).
- Bitwise alignment among pre-training, post-training and inference is the stated goal; debugging and stability analysis are the stated benefits ([V4] §3.3 p.18-19).
- "Sharply reducing irreproducibility" is not measured in [V4]; overhead is only described as "minimal" / "negligible" ([V4] §3.3 p.18). The 80-of-1000 figure on the slides is Thinking Machines' Qwen3/vLLM measurement ([TM]), not a DeepSeek result, so it is not used here.

## Other properties
none
