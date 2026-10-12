# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Coarse-grained fusion tuned to domestic NPU memory behavior: Lightning Indexer fuses score+ReLU+TopK into one op, Sparse Flash Attention fuses sparse indexing with attention, MLAPO collapses 13 small ops into one super-operator, and MoE routing + permute + GroupedGEMM become a Mega-Fusion.
**After:** GLM-5's Ascend kernel fusion: Lightning Indexer fuses score + ReLU + TopK into one kernel so the NPU overlaps compute with memory access; Sparse Flash Attention does TopK selection from the KV cache and sparse attention in one kernel; MLAPO fuses 13 small pre-processing ops into one super-operator that runs Vector and Cube units in parallel. GLM-5 reports no MoE fusion (vLLM-Ascend separately fuses grouped matmul + SwiGLU + quant).
**Why:**
- The three kernels and their stated purposes are the report's own descriptions ([GLM5] §5 p.21-22); "tuned to domestic NPU memory behavior" replaced by the stated benefits (compute/memory overlap for the indexer, Vector/Cube parallelism for MLAPO).
- "MoE routing + permute + GroupedGEMM become a Mega-Fusion" is not in the GLM-5 report, which names only these three kernels ([GLM5] §5). The Ascend stack's MoE fusion that does exist is grouped matmul + SwiGLU + quantization in vLLM-Ascend ([VA-MOE]), not attributed to GLM-5; Pangu Ultra MoE's permute and GroupedMatMul fusions are for a different model's training ([PANGU] §4.5).

## Impact / Measured Results
**Before:** Sharply reduces kernel-launch and intermediate memory traffic - the key to domestic hardware matching GPUs.
**After:** No kernel-level numbers published; GLM-5 credits quantization, fusion and scheduling jointly for one Chinese-chip node comparable to dual-GPU international clusters, at 50% lower long-sequence deployment cost.
**Why:**
- The report publishes no launch counts, memory-traffic numbers or per-kernel speedups ([GLM5] §5); the "sharply reduces" claim has no measurement behind it.
- The only end-to-end claim (single node comparable to "dual-GPU international clusters", 50% lower deployment cost in long-sequence scenarios) is attributed to all hardware-level co-optimizations together, not to fusion ([GLM5] §5 p.22 last paragraph); the GPU setup is not specified in the report.

## Other properties
none changed. Title still lists "MoE Mega-Fusion", which the GLM-5 report does not describe (its third kernel is the Lightning Indexer); left for the user to decide.
