# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Decomposes each sequence chunk into two locally computable terms - the cumulative transition applied to the entry state, and the state generated locally from zero - so only fixed-size recurrent states need an all-gather.
**After:** unchanged
**Why:** matches the sources: S_out = M S_in + S~, both terms computed locally, then an all-gather of fixed-size states and a prefix-scan merge ([K3] §5.1.2, Eq. 17; [FLA-CP] "Pre-Process Forward", `chunk_delta_h.py`).

## Impact / Measured Results
**Before:** Communication is independent of sequence length (softmax-attention CP must ship KV blocks that grow with length); compute scales linearly to 1M context.
**After:** Communication is independent of sequence length (softmax-attention CP must ship KV blocks that grow with length); K3 publishes no KCP timing for its linear-compute claim; FLA's open implementation reaches 86% fwd+bwd scaling efficiency vs 55% for head-split all-to-all CP (KDA, 32K, CP 4, H800).
**Why:**
- "Compute scales linearly to 1M context" reads as measured; K3 only states that KCP "achieves linear compute scaling" and that CP makes 1M-token training tractable, with no KCP measurement at any length ([K3] §5.1.2, §3.4).
- The only public timing is FLA PR #691 at 32K tokens, CP = 4, 64 heads, head dim 128, H800: 86.26% vs 55.12% fwd+bwd ([PR691] "Performance").
- Labeled "FLA's open implementation" because FLA's README says its CP code is a separate codebase from Moonshot's internal KCP ([FLA-CP] "Acknowledgments").

## Other properties
none (the page body describes the Kimi K3 repo as "architecture and training notes", but its README has nothing on KCP; body left untouched per brief)
