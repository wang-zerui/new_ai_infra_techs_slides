# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Normalizes Q/K so the dynamic range of attention logits stays in a stable band (the same problem QK-Clip attacks from the other side).
**After:** Per-head RMSNorm with learnable gains on Q and K (before RoPE) bounds attention logits at sqrt(d) times the gains, independent of Q/K weight norms; same problem QK-Clip attacks from the other side, but needs per-head keys, which MLA does not materialize.
**Why:**
- "Stable band" overstates: the bound is soft, sqrt(d) x max gain_q x max gain_k, set by 2 x 128 learnable gains per layer (analytic, sources.md "sqrt(128) = 11.3"; [HF-CODE] RMSNorm, norm before RoPE, scale 128^-0.5).
- Form described from the code: per-head RMSNorm with a 128-dim weight, keeping the 1/sqrt(d) scale ([HF-CODE]; [CFG-GLM45]), not the l2 + learnable-scalar form of the cited Henry et al. paper ([QKN] eq. 2-5).
- MLA limit added because the row presents QK-Clip as the alternative: QK-Norm is "not applicable" to MLA since keys are not materialized ([K2] §2.1 p.3; [DSV3] §2.1.1).

## Impact / Measured Results
**Before:** Together with Muon, supports stable convergence at larger batch sizes.
**After:** Stabilizes the attention-logit range in GLM-4.5 (no isolating measurement published); the larger-batch tolerance is credited to Muon, not QK-Norm.
**Why:**
- [GLM45] §2.4 credits Muon with "accelerate convergence and tolerate larger batch sizes"; QK-Norm is credited only with stabilizing "the range of attention logits" (§2.1 p.3).
- GLM-4.5 publishes no logit curves or ablation isolating QK-Norm.

## Other properties
none
