# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** One (MoE) MTP layer on top of the backbone provides multi-token supervision during training and serves as the speculative-decoding drafter at inference; MCore supports MTP + MLA + CP together.
**After:** One MTP module (a Transformer block, MoE in GLM-4.5) takes the backbone's final hidden state plus the next token's embedding, shares the embedding and output head, and predicts one token further; it adds a weighted extra loss in training and serves as the speculative-decoding drafter at inference; MCore supports MTP + MLA + CP together.
**Why:**
- The module is not just a layer stacked on the backbone: it projects RMSNorm(h) concatenated with RMSNorm(Emb(t_i+1)) through M_k, runs one Transformer block, and reuses the main embedding and output head ([V3] §2.2 Eq. 21-23, Fig. 3; [CODE] enorm/hnorm/eh_proj). The block is MoE in GLM-4.5 ([GLM] §2.1).
- Loss weighting: L_MTP scaled by lambda (0.3 then 0.1) and added to the main loss ([V3] Eq. 25, §4.2; [GLM] §2.4; [V4] §4.2.2).
- "MCore supports MTP + MLA + CP" kept: confirmed by [RD] v0.14.

## Impact / Measured Results
**Before:** Wins on both training efficiency and decode throughput; GLM-5.2's improved MTP raises accepted length by 20%; also raises per-step compute density in Ascend deployments.
**After:** Better quality at equal training tokens but extra training compute (DeepSeek-V3 ablation; adds 14B params to V3); 1.8x decode TPS as a speculative drafter with 85-90% second-token acceptance (DeepSeek-V3); GLM-5 accept length 2.76 vs DeepSeek-V3.2's 2.55 (4 steps, cross-model); raises compute density in Ascend deployments.
**Why:**
- "Training efficiency" overstates: MTP improves quality at equal tokens ([V3] §4.5.1, Table 4) but adds a Transformer block and an output-head pass; DeepSeek-V3 carries 14B MTP params on top of 671B ([MC] table note p.13).
- Decode throughput given its published numbers ([V3] §5.4.3).
- "GLM-5.2 ... +20% accepted length" appears in no source read; replaced by the GLM-5 report's published accept lengths, a cross-model comparison on a private prompt set, not an ablation ([GLM5] §2.1, Table 2).
- Ascend compute-density claim kept (qualitative, [GLM5] p.22).

## Other properties
none. Left for the user: the Reference (GLM-4.5, arXiv 2508.06471) does mention the MoE MTP layer, so it is not wrong, but the module design, ablation and acceptance numbers come from DeepSeek-V3 (arXiv 2412.19437).
