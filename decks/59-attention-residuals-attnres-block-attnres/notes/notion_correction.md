# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Per-layer learnable pseudo-queries aggregate the embedding and all preceding layer outputs with softmax-attention weighting instead of equal-weight residual addition; the Block variant groups layers to cut memory from O(Ld) to O(Nd), with two-stage computation and cache-based pipelined communication.
**After:** unchanged
**Why:** every statement is supported: pseudo-query softmax over the embedding and preceding outputs ([AR] Eq. 1-4, §3); Block AttnRes memory O(Ld) to O(Nd) ([AR] §3.2); two-phase inference ([AR] Alg. 1, §4.2); cross-stage caching under pipeline parallelism ([AR] §4.1, Fig. 3).

## Impact / Measured Results
**Before:** Relieves PreNorm depth dilution - output magnitude and gradients stay uniform with depth; improves every task on Kimi Linear 48B / 1.4T tokens.
**After:** Relieves PreNorm depth dilution: output magnitudes stay bounded and gradients become substantially more uniform with depth; on Kimi Linear 48B / 1.4T tokens it matches or beats the baseline on all 15 benchmarks (MMLU-Pro tied at 52.2, GPQA-Diamond 36.9 to 44.4).
**Why:**
- Magnitudes are bounded with a periodic pattern within blocks, and gradients are "substantially more uniform", not uniform ([AR] §5.2, Fig. 5).
- "Improves every task" is contradicted by Table 3 (MMLU-Pro 52.2 vs 52.2); the paper's own text says "matches or outperforms the baseline on all benchmarks" ([AR] §5.2, Table 3 p.10).
- GPQA-Diamond added as the largest published gain, to keep a concrete number ([AR] Table 3; +7.5 stated in §5.2).

## Other properties
none (Source "Kimi K3" kept: K3 adopts Block AttnRes with 8 x 12-layer blocks, [K3] §2.2, Table 1; the measured results come from the AttnRes report, which the Reference already points to)
