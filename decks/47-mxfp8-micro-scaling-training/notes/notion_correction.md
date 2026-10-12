# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Every 32 contiguous values share one E8M0 (pure power-of-two) scale, with E4M3 for weights/activations/gradients; natively supported by Blackwell Tensor Cores, requires keeping both transposed and non-transposed copies.
**After:** Every 32 contiguous values share one E8M0 (pure power-of-two) scale, with E4M3 for weights, activations and gradients; Blackwell Tensor Cores apply the scales natively; each tensor keeps rowwise (1x32) and columnwise (32x1) copies, both quantized from high precision, so parameters are all-gathered in BF16.
**Why:**
- The two copies are not transposes of each other: 1x32 and 32x1 blocks are numerically different and each is quantized from the high-precision tensor; Blackwell needs no physical transpose ([TE-mx] "Handling transposes"; [MX] Sec. 3.2.1).
- Because the copies cannot be derived from each other, MXFP8 parameters are all-gathered in BF16 ([MCore-MoE] Sec. 5.2.2).

## Impact / Measured Results
**Before:** 1.28-1.37x over BF16 on B200 (up to 2x on GEMMs), loss curves indistinguishable from BF16, and covers more layers than per-tensor FP8.
**After:** Within 0.50% of BF16 validation perplexity (8B model, 15T tokens, MX simulated on Hopper); 1.22x BF16 tokens/s for DeepSeek-V3 and Qwen3-235B on 256 GB200 (derived) and 1.44x GEMM speedup on B300; quantizes every transformer block, unlike per-tensor FP8.
**Why:**
- "1.28-1.37x over BF16 on B200" appears in none of the sources read; the MXFP8 paper reports no measured speedup. Replaced by 4,020/3,298 and 6,212/5,100 tokens/s/GPU = 1.22x ([MCore-MoE] Table 11, derived; each precision with its own best parallel config) and the 5B-model GEMM speedup of 1.44x on B300 ([TE-speed]).
- "Up to 2x on GEMMs" is the peak Tensor Core math rate vs BF16 ([MX] Table 1), not a measurement; dropped.
- "Indistinguishable" is quantified in the paper as under 0.50% validation-perplexity difference, with MX numerics simulated on 3,072 Hopper GPUs ([MX] Sec. 3.1).
- More layers: MXFP8 covers all transformer blocks while the per-tensor FP8 baseline keeps the first and last in BF16 ([MX-v1] Sec. 3.1, [MX] Sec. 3.2.1); kept.

## Other properties
none (the discrepancy notes that Megatron-LM / MCore is missing from Source; that is an omission, not a wrong value, so Source was left unchanged and the question is left to the user)
