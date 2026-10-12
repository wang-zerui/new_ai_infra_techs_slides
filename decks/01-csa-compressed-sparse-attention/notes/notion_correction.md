# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Every m=4 consecutive tokens are compressed into one KV entry by a learned overlapping-window weighted average (1M -> 250K), followed by top-k (512/1024) selection plus a 128-token sliding window for local detail; MQA at the core.
**After:** Each m=4-token block becomes one KV entry: a learned softmax-weighted sum over 8 token projections (this block and the previous one, overlapping), cutting length to 1/4; an indexer then picks the top-k compressed entries (1024 Pro / 512 Flash), plus a 128-token sliding window; shared-KV MQA at the core.
**Why:**
- Compression detail from V4 §2.3.1 Eq. 9-12 (softmax over 2m = 8 projections, C^a of block i and C^b of block i-1, learned positional biases).
- "1M -> 250K" is not a number in V4; the paper states the factor (length to 1/m, m = 4), so the row now says 1/4 (sources.md discrepancy 2).
- Top-k is over compressed entries, chosen by the indexer (Eq. 13-17); 1024 for Pro, 512 for Flash (§4.2.1); MQA is shared-KV (Eq. 18-19).

## Impact / Measured Results
**Before:** Per-token FLOPs at 1M context are only 27% (Pro) / 10% (Flash) of V3.2.
**After:** Whole V4 stack (CSA + HCA + low-precision compute), no CSA-only ablation: single-token FLOPs at 1M context are 27% (Pro) / 10% (Flash) of V3.2's.
**Why:**
- V4 credits the 27% / 10% to the hybrid CSA + HCA design together with low-precision computation and storage, not to CSA alone; no CSA-only ablation is published (Abstract; §1 p.5; §2.3.4).

## Other properties
none
