# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Post-training MXFP4 QAT on MoE expert weights and the CSA indexer's QK path; stored as FP4 (E2M1) and dequantized to FP8 (E4M3) for compute (FP8's extra exponent bits make dequantization mathematically lossless), backward uses STE to update FP32 master weights, index scores drop from FP32 to BF16.
**After:** Post-training MXFP4 QAT on MoE expert weights and the CSA indexer's QK path; FP32 master weights are quantized to FP4 (E2M1, 1x32 sub-blocks) and dequantized to FP8 (E4M3, 128x128 blocks) so the FP8 training stack runs unmodified, lossless only while FP4 sub-block scale ratios inside each FP8 block stay under a threshold (verified empirically); gradients go straight to the FP32 master weights (STE); index scores drop from FP32 to BF16.
**Why:**
- Losslessness is conditional on the max/min ratio of FP4 sub-block scales within a 128x128 FP8 block staying under a threshold, which DeepSeek verified empirically for current weights; "mathematically lossless" overstates it ([V4] §5.2.1 p.33).
- Purpose of the FP4-to-FP8 path is to reuse the existing FP8 training framework unmodified; the backward pass computes gradients w.r.t. the FP8 weights and passes them to the FP32 master, equivalent to an STE ([V4] §5.2.1 p.33-34).

## Impact / Measured Results
**Before:** 2x faster top-k selector with KV recall held at 99.7%; deployment runs the FP4 weights directly, so training and serving behavior match.
**After:** Quantizing index scores from FP32 to BF16 within QAT makes the top-k selector 2x faster while recalling 99.7% of KV entries; inference and RL rollouts run native FP4 weights, so sampling matches online deployment.
**Why:**
- [V4] ties the 2x / 99.7% to the FP32-to-BF16 index-score quantization, not to FP4 QAT as a whole ([V4] §5.2.1 p.33).
- Native FP4 weights are stated for the inference and RL-rollout phases, so that sampling matches online deployment ([V4] §5.2.1 p.33-34); wording made precise.

## Other properties
none
