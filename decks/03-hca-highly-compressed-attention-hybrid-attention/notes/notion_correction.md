# Notion row correction (2026-10-13)

## Technique
**Before:** HCA highly-compressed attention + hybrid attention stacking
**After:** HCA heavily-compressed attention + hybrid attention stacking
**Why:** the paper names it Heavily Compressed Attention (V4 Abstract, §2.3.2).

## Core Mechanism
**Before:** High-compression attention layers with m'=128 interleaved with CSA; every layer remains attention - no SSM or Gated-DeltaNet-style linear attention is introduced.
**After:** Softmax pooling folds each 128-token block into one KV entry (m'=128), and queries attend densely to all entries plus a 128-token raw window (no indexer, no top-k); interleaved with CSA, so every layer stays softmax attention.
**Why:**
- Mechanism detail from V4 §2.3.2-2.3.3 and Eq. 20-23 (non-overlapping blocks, dense shared-KV MQA, 128-token window).
- The "no SSM / Gated-DeltaNet" framing is not argued in the paper (it even manages the window cache "as a state-space model", §3.5.1); kept only the supported fact that every layer is softmax attention.

## Impact / Measured Results
**Before:** KV cache falls to 10% (Pro) / 7% (Flash) of V3.2, making 1M context routine.
**After:** Whole V4 stack (CSA + HCA + low-precision KV storage): 1M-token KV cache is 10% (Pro) / 7% (Flash) of V3.2's; retrieval still fades past 128K (MRCR 8-needle 0.92 at 128K, 0.59 at 1M).
**Why:**
- V4 credits the 10% / 7% to the hybrid CSA + HCA design together with low-precision storage, not to HCA alone (§1 p.5, §2.3.4).
- "Routine 1M" glosses over the measured retrieval drop (V4 Fig. 9).

## Other properties
none
