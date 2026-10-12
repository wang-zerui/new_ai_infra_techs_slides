# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Splits AllReduce into ReduceScatter + AllGather and pipelines it with tiled matrix computation to mask weaker interconnect bandwidth; layered with prefill/decode-disaggregated serving, prefix caching and MTP speculative decoding.
**After:** FlashComm splits AllReduce into ReduceScatter + AllGather so the MoE path sees 1/TP of the tokens and communication hides behind computation; vLLM Ascend can also fuse the matmul with its reduce-scatter; layered with prefill/decode-disaggregated serving, prefix caching and MTP speculative decoding.
**Why:**
- The GLM-5 report says only that FlashComm "splits AllReduce operations to hide communication latency behind computation" (GLM5 §5 p.22); the RS + AG split and the 1/TP token sharding are from vLLM Ascend (VA-GLM5 §9.1 Table 2; VA-SP).
- Tile-level fusion of matmul and collective is the engine's (fuse_gemm_comms, VA-SP) and the literature's (FLUX), not the report's, so it is now attributed to vLLM Ascend (sources.md discrepancy 1).
- "To mask weaker interconnect bandwidth" dropped: no source states it for GLM-5 on Ascend, and no Ascend communication share is published (discrepancy 4).

## Impact / Measured Results
**Before:** One domestic node matches a dual-GPU node; about 50% lower long-sequence serving cost.
**After:** Whole Ascend stack, no per-technique split: about 50% lower long-sequence deployment cost; one node claimed comparable to dual-GPU clusters, with no benchmark published.
**Why:**
- The 50% is the result of the whole Ascend adaptation stack (quantization + fused kernels + scheduling), with no per-pillar breakdown (GLM5 §5 p.22) (discrepancy 3).
- The report writes "comparable to dual-GPU international clusters" with no baseline, benchmark, metric or configuration; "matches" overstated it (discrepancy 2).

## Other properties
none
