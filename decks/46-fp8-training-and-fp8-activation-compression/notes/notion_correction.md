# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** TE provides DelayedScaling / blockwise FP8 recipes with automatic autocast; MCore supports FP8 parameter all-gather and fine-grained recompute under FP8; K2 compresses activations in FP8-E4M3.
**After:** TE offers per-tensor (delayed or current scaling) and blockwise FP8 recipes under autocast; blockwise uses 1x128 tiles for activations and gradients, 128x128 blocks for weights; MCore casts FP32 master weights straight to FP8 for the parameter all-gather; K2 caches MoE up-projection and SwiGLU inputs in FP8-E4M3 but computes in BF16.
**Why:**
- TE also ships Float8CurrentScaling, and Megatron-Core does not recommend DelayedScaling for precision reasons ([TE-primer], [TE-cur], [MCore-MoE] Sec. 5.3.1).
- Blockwise granularity from [TE-blk] "Block size" and [DSV3] Sec. 3.3.2.
- FP8 parameter all-gather casts FP32 master shards directly to FP8, eliminating the BF16 copy ([MCore-MoE] Sec. 5.2.2).
- "Fine-grained recompute under FP8" is not in the sources read; dropped.
- K2 compresses activations for storage only and explicitly does not compute in FP8 ([K2] Sec. 2.4.3); the old wording under "FP8 training" implied FP8 GEMMs.

## Impact / Measured Results
**Before:** FP8 parameter all-gather cuts roughly 50% of that communication; activation memory halved - the default recipe for DeepSeek-V3-class models on H100.
**After:** FP8 parameter all-gather moves 1 instead of 2 bytes per parameter (by design); FP8 caching halves each eligible tensor, saving about 16 of 131 GB of DeepSeek-V3 activations per GPU; MCore recommends blockwise FP8 on Hopper, validated on DeepSeek-V3.
**Why:**
- The 50% all-gather saving is a byte count (1 vs 2 B/param), not a measured result ([MCore-MoE] Sec. 5.2.2).
- Only FP8-eligible linear-layer inputs are halved; for DeepSeek-V3 that is about 16 GB of 131 GB, about 12% ([MCore-MoE] Sec. 4.1.3, Table 3). K2 reports no memory number ([K2] Sec. 2.4.3).
- Megatron-Core recommends blockwise FP8 on Hopper and validated it on DeepSeek-V3 ([MCore-README] "FP8 Recipes"); DeepSeek-V3 itself trained on H800, so "default recipe on H100" was replaced ([DSV3]).

## Other properties
none
