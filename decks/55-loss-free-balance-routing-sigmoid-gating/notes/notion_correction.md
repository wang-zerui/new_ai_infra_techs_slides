# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** A learnable per-expert bias with fixed update rate regulates load instead of an auxiliary loss; scoring uses sigmoid.
**After:** A per-expert bias, used only for top-K selection, is moved by a fixed sign step after each batch from token counts (no gradient) instead of a tuned auxiliary loss, though production keeps a 1e-4 sequence-wise loss; scoring uses sigmoid.
**Why:**
- The bias is not learned by back-propagation: it is updated by b_i += u·sign(e_i) from per-expert token counts ([LF] Algorithm 1 p.4; [V3] p.9) and stored as a non-gradient buffer in MCore ([CODE] `expert_bias`).
- The bias affects only which experts are picked; gate weights use the unbiased scores ([LF] Eq. 3; [V3] Eq. 16).
- DeepSeek-V3, GLM-4.5 and DeepSeek-V4 all keep a complementary sequence-wise balance loss with weight 0.0001 ([V3] p.23; [GLM] §2.4; [V4] p.25-26); the original paper uses none.

## Impact / Measured Results
**Before:** Now the default routing setup for frontier open MoEs; MCore ships this router and its fused kernels.
**After:** Adopted by DeepSeek-V3/V4 and GLM-4.5; MCore ships this router and its fused kernels as opt-in (defaults: no expert bias, softmax).
**Why:**
- In Megatron-Core it is not a default: `--moe-router-enable-expert-bias` defaults to False and `--moe-router-score-function` to softmax ([RD] Router Arguments table).
- "Frontier open MoEs" narrowed to the models checked in the sources ([V3], [V4], [GLM]); fused router from MCore v0.14 ([RD]; [MC] §4.3.4).

## Other properties
none. For the user to decide (not changed, since neither value is plainly wrong):
- Reference is GLM-4.5 (arXiv 2508.06471), which adopts the technique in one sentence plus hyper-parameters; the method and controlled experiments are in DeepSeek's arXiv 2408.15664 and DeepSeek-V3 §4.5.2.
- Source lists "DeepSeek V3.2", which the deck did not read; the deck relies on DeepSeek-V3 and DeepSeek-V4.
