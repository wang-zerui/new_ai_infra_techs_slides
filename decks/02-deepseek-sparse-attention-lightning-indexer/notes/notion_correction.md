# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** A lightweight indexer scores history tokens (multi-head ReLU dot-product weighting), top-k (about 2048) picks a KV subset, and sparse attention then runs on top of MLA; migrated through dense-to-sparse continued pretraining.
**After:** A lightweight FP8 indexer (64 heads) scores every preceding token with head-weighted ReLU dot products; top-k (k = 2048) picks the KV entries, and sparse attention runs on MLA in its MQA mode; migrated from V3.1-Terminus by a dense warm-up, then sparse continued pretraining at 128K.
**Why:**
- k is exactly 2,048, not "about" (V32 §2.1.1; HF config index_topk) (sources.md discrepancy 1).
- 64 heads and FP8 from HF config / model.py and V32 Eq. 1 ("can be implemented in FP8"); DSA is instantiated on MLA's MQA mode (V32 §2.1).
- Warm-up (main model frozen, dense) then sparse training from V3.1-Terminus on 128K sequences (V32 §2.1.1).

## Impact / Measured Results
**Before:** Large drop in long-context training and serving cost with base quality on par with V3.1; GLM-5 adopts it wholesale and completed large-scale DSA-based RL.
**After:** Per-token serving cost at 128K about 3.6x lower (prefill) / 8.7x (decode) than V3.1-Terminus (derived from V3.2 Fig. 3, H800); training cost not quantified; quality on par. GLM-5 adopts it with changes (32 indexer heads, 20B-token adaptation, deterministic top-k, frozen indexer during RL).
**Why:**
- DeepSeek measures only inference cost (V32 Fig. 3, §2.3); training-cost reduction is not measured, so it is no longer claimed (discrepancy 2). The 3.6x / 8.7x are read from the Fig. 3 vector data (notes/digitize_fig3.py), hence "derived".
- Baseline is V3.1-Terminus, not "V3.1" (V32 §2.1, §2.2: on par, +4 on AA-LCR).
- "Wholesale" is wrong: GLM-5 uses 32 indexer heads (Table 10), 20B adaptation tokens vs 943.7B (§2.1.1), deterministic torch.topk and a frozen indexer during RL ("DSA RL insights") (discrepancy 3).

## Other properties
none. Not changed (page body, out of scope): the body links Native Sparse Attention (arXiv 2502.11089) as a paper for this row, though NSA is a different, natively trained method (sources.md discrepancy 4).
