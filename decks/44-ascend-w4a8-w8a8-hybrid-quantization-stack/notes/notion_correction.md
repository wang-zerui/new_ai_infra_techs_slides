# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** QuaRot rotation preprocessing removes activation outliers, Flex SmoothQuant searches per-layer migration strength, SSZ (scale-shift-zero) handles weight quantization; W4A8 for MoE experts, W8A8 kept for sensitive layers.
**After:** In msModelSlim, QuaRot rotation suppresses activation outliers and Flex_AWQ_SSZ calibrates scaling (named but not defined in the GLM-5 report); MoE experts use W4A8 (INT4 weights, INT8 activations) and standard attention and MLP blocks W8A8, a structural split since experts hold about 97% of weights (derived).
**Why:**
- The report names Flex_AWQ_SSZ for scaling calibration; SmoothQuant does not appear in it ([GLM5] §5 p.21; lineage in [AWQ] Eq. 5, [SQ] §4).
- "SSZ (scale-shift-zero)" is an interpretation; the report never defines SSZ, so it is not presented as a separate weight-quantization step ([GLM5] §5 p.21).
- The precision split is by block type ("standard Attention and MLP blocks use W8A8 (INT8), while the MoE experts are compressed to W4A8 (INT4)"), not by layer sensitivity ([GLM5] §5 p.21).
- 97% = 724.8B routed-expert parameters of 744B, derived from [GLM5] Table 10 (sources.md derivation).

## Impact / Measured Results
**Before:** Halves weight memory at near-lossless accuracy, with large throughput gains on domestic chips.
**After:** No published accuracy, memory or throughput numbers, only 'without significant accuracy loss'; weights about half of all-INT8 (derived); the 50% long-sequence cost cut is for all Ascend co-optimizations.
**Why:**
- [GLM5] §5 gives no accuracy, memory or throughput figures for the quantized model; "near-lossless" and "large throughput gains" are unsupported.
- Hybrid weights 381.6 GB vs 744 GB all-INT8 = 1.95x, derived from Table 10; "halves" holds only against INT8 (BF16 is 3.9x larger).
- The only end-to-end number, 50% lower deployment cost in long-sequence scenarios, is credited to all Ascend co-optimizations together ([GLM5] §5 p.22).

## Other properties
none
