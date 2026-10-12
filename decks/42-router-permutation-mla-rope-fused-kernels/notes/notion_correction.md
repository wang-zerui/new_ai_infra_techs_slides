# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** --moe-router-fusion fuses routing score/topk/bias, layered with permutation, MLA RoPE and FP8-cast fusion, plus cuDNN fused attention / FlashAttention integration.
**After:** --moe-router-fusion fuses score function, (group) top-k and expert bias into one kernel, with separate fused aux-loss kernels; --moe-permute-fusion builds the row-id map once and fuses permute/unpermute with padding and FP8-scale permutation; MLA RoPE fusion applies YaRN RoPE to Q and KV in Triton kernels.
**Why:**
- cuDNN fused attention is a separate feature in the README list, not one of these glue-kernel fusions, and no source read mentions a FlashAttention integration; dropped ([RD]; discrepancy 1).
- Router: one kernel for score function + top-k (naive or group) + bias + scaling, plus aux-loss-score and aux-loss kernels under a separate flag ([MC] §4.3.4 p.37-38, [TE-R], [TE-1883], [MLM-CFG] `moe_router_aux_loss_fusion`).
- Permutation: row-id map built once in three passes and reused; padding and FP8 scales handled inside the permute kernel ([MC] §4.3.3 p.36-37, §5.4.1; [TE-P] `FUSION_PAD`, `PERMUTE_SCALE`; [TE-1921]).
- MLA RoPE: Triton Q (in place) and KV-split kernels with a cached YaRN cos/sin table; YaRN only ([MLM-ROPE], [MLM-CFG] `apply_rope_fusion`).

## Impact / Measured Results
**Before:** Cuts the launch and memory-traffic overhead of many small kernels; pairs with sync-free dropless MoE for higher compute efficiency.
**After:** Cuts the launch and memory-traffic overhead of many small kernels; only permute+pad fusion has published gains (1.1x to 1.6x permute, 1.7x to 3x unpermute kernels; +0.4% MFU, about 1 GB less peak memory in FP8 training of a 600B-class MoE).
**Why:**
- No isolated number is published for router fusion or MLA-RoPE fusion; the paper says only that the fusions "reduce kernel launch count" in the GB200 configuration ([MC] §9.2.2 p.75). The measured numbers are for permute+pad fusion ([TE-1921] kernel benchmarks and end-to-end, about 600B-parameter FP8 Megatron setup).
- "Pairs with sync-free dropless MoE for higher compute efficiency" is not stated for these kernels in any source read; the sync-free fused expert kernels of [BLOG-F] are GroupGEMM epilogues, a different fusion. Dropped.

## Other properties
none
