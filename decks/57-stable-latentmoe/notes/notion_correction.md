# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Experts down-project into a latent space (an MLA analogue), with SiTU-GLU to suppress activation blow-up and RMSNorm on routed experts for numerical stability.
**After:** A shared W-down projects each token from 7,168 to a 3,584 latent before dispatch and a shared W-up restores width after combine (from NVIDIA LatentMoE); RMSNorm on the weighted sum of routed outputs before W-up and SiTU-GLU (smoothly capped GLU) in every FFN suppress activation blow-up.
**Why:**
- The down-projection is one shared W-down applied before dispatch, not done by each expert ([K3] Eq. 11 p.7; [K3-cfg] `routed_expert_down_proj` 7168 to 3584, `routed_expert_up_proj`).
- K3 cites LatentMoE (arXiv 2601.18089) as the origin of the latent path and never calls it an MLA analogue; credited LatentMoE instead ([K3] §2.3; [LMoE] §3).
- RMSNorm is applied once to the aggregated routed latent u, not per routed expert ([K3] Eq. 11, §2.3.1 p.7; [K3-cfg] `routed_expert_norm`).
- SiTU-GLU described as a smooth cap on both GLU factors ([K3] §2.3.2, App. B Eq. 18-19).

## Impact / Measured Results
**Before:** Trains stably with 16 of 896 routed experts active per token (\~56x sparsity); the key enabler of K3's width scaling.
**After:** K3 (2.8T total, 104B active) trains with 16 of 896 routed experts active per token (sparsity 56) at K2's per-token all-to-all volume (derived); no isolated ablation of the RMSNorm or SiTU-GLU stabilizers is published.
**Why:**
- "Trains stably" and "key enabler" are inferences: K3 publishes no loss or activation curves isolating the stabilizers, and its 2.5x scaling-efficiency gain is credited jointly to KDA, AttnRes, Stable LatentMoE and recipes ([K3] §2.3.1, §3.2, Fig. 7).
- Sizes from [K3] abstract and Table 1; 896 / 16 = 56 exactly, so no "about".
- Equal all-to-all volume: 16 x 3,584 = 8 x 7,168 (derived, sources.md "2x all-to-all"; [LMoE] §2.2).

## Other properties
none (the page body lists DeepSeek-V2/MLA but not LatentMoE, the actual base design; body left untouched per brief)
