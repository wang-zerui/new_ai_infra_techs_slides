# Story — deck 01, CSA (Compressed Sparse Attention)

- **Where it lives** (prosperity): DeepSeek-V4 (Pro 1.6T / 49B active, Flash 284B / 13B active), native
  1M-token context. CSA is the "sparse" half of V4's hybrid attention stack: 30 of 61 Pro layers, 21 of 43
  Flash layers; the other half is HCA (deck 03).
- **Pain number** (betrayal): at 1M tokens, DeepSeek Sparse Attention (V3.2) spends **30×** more
  multiply-adds in its lightning indexer than in the attention it selects for (analytic, per query per
  layer: 8.6 G vs 0.29 G MAC), because the indexer still scores every one of the 1,048,576 preceding
  tokens to keep 2,048.
- **Failed suitors**: (1) dense MLA attention (V3.1): reads all L entries per query, O(L²) attention;
  (2) DSA (V3.2): attention becomes O(L·k) but the indexer stays O(L²) and the KV cache still holds one
  entry per token per layer; V3.2 stops at 128K context.
- **Villain**: token granularity. Both the selector and the cache treat the single token as the unit of
  retrieval, so their cost grows one-for-one with context length.
- **Turn / insight**: retrieval does not need token granularity — select among learned 4-token summaries
  instead of tokens, and keep exact tokens only for the last 128 positions. Every term downstream of the
  compressor (index scan, cache, attended set) shrinks by m = 4 or more.
- **Mechanisms**:
  ❶ Token-level compressor — challenge: compress without cutting dependencies at block edges →
     overlapping 2m-token softmax pooling with positional biases (n → n/4 entries).
  ❷ Lightning indexer over compressed keys + top-k — challenge: pick the right blocks cheaply → 64-head
     ReLU scorer on a shared low-rank query latent, top-1024 (Pro) / 512 (Flash); FP4 QK path (deck 45).
  ❸ Shared-KV MQA with a sliding-window branch — challenge: causality hides the query's own block and
     compression blurs local detail → 128 uncompressed recent entries concatenated with the selected
     blocks; K = V = one 512-d entry; grouped output projection.
- **Release**: whole V4 stack at 1M: **27%** (Pro) / **10%** (Flash) of V3.2's single-token FLOPs
  (3.7× / 9.8× lower); analytic per-CSA-layer work 3.9× below a DSA layer; LongBench-V2 rises 40.2 → 51.5.
- **Limits**: indexer is still O(L²/m) in prefill and dominates the layer (14× core attention, analytic);
  needs dense warm-up (1T tokens) and staged sparsification; breaks PagedAttention assumptions (state
  cache, lcm block size); CP needs boundary exchange; no isolated CSA ablation published; the paper
  calls the architecture "relatively complex".

## Storyboard (14 logical slides, 8 minutes)

| # | Beat | Headline (as built) |
|---|---|---|
| 1 | Title | CSA: Compressed Sparse Attention |
| 2 | Hook | V4 serves 1M tokens with two attention types |
| 3 | Pain (30×, analytic) | At 1M, choosing costs more than attending |
| 4 | Status quo | DSA selects tokens, so its cost tracks context |
| 5 | Insight | Key idea: select among blocks, not tokens |
| 6 | Overview | CSA compresses, indexes, then attends |
| 7 | ❶ (2 build steps) | ❶ Learned pooling folds 4 tokens into one |
| 8 | ❷ | ❷ A light indexer keeps the top 1,024 blocks |
| 9 | ❸ | ❸ A raw 128-token window restores locality |
| 10 | Result (27% / 10%) | V4 needs 27% of V3.2's per-token FLOPs at 1M |
| 11 | Result (3.9×, analytic) | Per layer, CSA does 3.9× less attention work |
| 12 | Result (40.2 → 51.5) | Long-context quality rises instead of falling |
| 13 | Limits | Cost: a quadratic indexer and harder serving |
| 14 | Summary | Summary |
