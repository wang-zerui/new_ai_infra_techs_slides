# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Dispatch sends FP8 (data plus scale) and combine returns BF16; the backward dispatch is exactly the forward combine; the Hybrid-EP backend further supports NVFP4.
**After:** Dispatch sends FP8 (data plus per-128-channel scales) and combine returns BF16; in backward, dispatch's gradient is a combine and combine's a dispatch, so token-to-expert stays FP8 and expert-to-token BF16; NVFP4 is only listed for the Hybrid-EP branch, whose own docs show BF16 and FP8.
**Why:**
- The source states the mirror per operator ("The backward pass of MoE dispatch is actually a combine" and vice versa, [RD]); precisions follow the direction ([V3] §3.3.3).
- Scale granularity: 1x128 per-token groups ([V3] §3.3.2; [MATH]).
- NVFP4: main README lists it for the Hybrid-EP branch, but that branch's doc lists "BF16, FP8" and its combine is always 2-byte ([HE-DOC], [HE-CFG]).

## Impact / Measured Results
**Before:** About 3h bytes per token-expert - the premise behind DeepSeek V4's C/B balance point (about 6144 FLOPs/Byte).
**After:** About 3h bytes per token-expert (plus about 3% FP32 scales, derived); DeepSeek V4 uses 3h for its analytic hardware balance point, C/B at most 2d = 6144 FLOPs/Byte (V4-Pro, d = 3072), not a measured result.
**Why:**
- V4's 3h ignores scales; with DeepEP's FP32 scales the payload is 3h + h/32 = 3.03h (derived in sources.md).
- 6144 FLOPs/Byte is a design threshold from V4 §3.1 "Observations and Proposals" (p.16) for d = 3072 ([V4] §4.2.1), not a DeepEP measurement.

## Other properties
none
