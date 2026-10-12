# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Weighted-ReLU MQA logits for the lightning indexer: each query takes E4M3 dot products over KV in [start, end), then ReLU, then a head-weighted sum; separate non-paged (prefill) and paged (decode) versions, now extended to an FP4 indexer.
**After:** Weighted-ReLU MQA logits for the lightning indexer: each query takes dot products over its KV range, E4M3 on SM90 or MXFP8 / MXFP4 on SM100, then ReLU, then a head-weighted sum; non-paged (prefill) and paged (decode) versions, plus a sparse variant (2026-09) that scores only selected KV blocks.
**Why:**
- E4M3 with a per-token scale is SM90-only; on SM100 the current API requires MXFP8 or MXFP4 inputs with packed UE8M0 block scales ([API]; PR #462) (sources.md discrepancy 1). This also covers the FP4 indexer (PR #304).
- A third family exists since 2026-09: the Sparse Indexer (fp8_fp4_sparse_mqa_logits and its paged variant) that scores only selected KV blocks for V4.1 ([RD] News; PR #432) (discrepancy 2).
- "[start, end)" rewritten as "its KV range" to avoid bracket characters in Notion markdown; meaning unchanged ([RD] ks / ke).

## Impact / Measured Results
**Before:** The prerequisite for DSA sparsity to actually save compute - the indexer itself must not become the new bottleneck.
**After:** unchanged
**Why:**
- Qualitative and not contradicted: IndexCache measures the indexer at up to 81% of prefill time at 200K on a 30B DSA model ([IC]). DeepSeek publishes no benchmark for these kernels (discrepancy 4), and the row claims none, so no number is added.

## Other properties
none
