# Story — deck 04: IndexShare / IndexCache (cross-layer index reuse for DSA)

Naming: the arXiv paper (2603.12201, Tsinghua + Z.ai) calls the method **IndexCache**; Z.ai's GLM-5.2 release (GitHub
README and blog, mid-2026) calls the production feature **IndexShare** and links to the same paper; the GLM-5 technical
report (arXiv 2602.15763v2, Feb 2026) uses neither name (it predates both); the HF config / transformers code call the
roles `indexer_types: full | shared`. The deck title uses the Notion name and says "IndexCache" wherever it cites the
paper.

- **Where it lives** (prosperity): DSA (deck 02) in GLM-5 (744B, 40B active, 78 layers, a 32-head indexer in every
  layer). GLM-5.2 extends the context from 200K to 1M tokens on the same DSA backbone.
- **Pain number** (betrayal): DSA's per-layer indexer still scores all L keys: on a 30B DSA model the indexer is
  **81%** of prefill time at 200K tokens (27% at 10K) and 41% of decode time (IndexCache Fig., §1).
- **Failed suitors**:
  1. Cross-layer top-k reuse (TidalDecode, OmniKV, Kascade, HySparse) needs full-attention anchor layers as the
     oracle; DSA has no full attention left.
  2. Skipping indexers uniformly without retraining loses long-context quality: Long Avg 50.2 → 43.0 at 1/4 retention
     (−7.2), 35.3 at 1/8; choosing layers by local similarity (DP over cosine similarity) is no better than uniform.
- **Villain**: every DSA layer re-derives its own top-k from scratch, as if layers were independent: total indexer
  work O(N·L²).
- **Turn / insight**: adjacent layers' indexers select **70–100%** of the same 2,048 tokens (measured pairwise overlap,
  30B DSA model, 200K tokens); so the indexer's own output is a cheap oracle to share.
- **Mechanisms**:
  ❶ F/S layers with an index cache: F layers run the indexer and overwrite a top-k buffer; S layers reuse it; one
     branch, no extra memory; S layers carry no indexer weights or indexer key cache (answers: reuse without an
     attention oracle).
  ❷ Training-free greedy layer selection by end-to-end LM loss on a calibration set (answers: which indexers are
     expendable; uniform and similarity-based choices fail).
  ❸ Training-aware multi-layer distillation: each retained indexer is distilled toward the average attention of all
     layers it serves (gradient-equivalent to one averaged target) → uniform patterns match full DSA (answers: an
     indexer trained for one layer must serve four). GLM-5.2 ships this shape: 21 of 78 layers keep an indexer, and its
     MTP layer reuses the first step's indices and KV.
- **Release**: with 1/4 of indexers, 1.82× faster prefill and 1.48× faster per-request decode at 200K (30B, H100,
  SGLang); quality within 0.4 Long-Avg points with a searched or trained pattern; GLM-5 (744B) ≥ 1.3× beyond 100K;
  GLM-5.2: **2.9×** lower per-token FLOPs at 1M; MTP acceptance length 4.56 → 5.10 with IndexShare + KVShare
  (5.47, +20%, with two more techniques).
- **Limits**: training-free search costs N(N−1)/2 forward passes (1,081 for 47 layers; 3,003 for 78, derived) and is
  pattern-sensitive; 1/8 retention loses 4.1 points even when searched; the remaining indexers are still O(L²) and the
  KV cache barely shrinks (GLM-5.2 blog); production use needs (re)training and engine support for "shared" layers.

## Slide plan (15 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Every DSA layer runs its own indexer | prosperity | fig_stack (SVG) |
| 3 | At 200K, the indexer is 81% of prefill | pain | `%% big: 81%` |
| 4 | The indexer's share grows with context | escalation / villain | chart_share (line) |
| 5 | Prior reuse needs a full-attention oracle | failed suitors | two-col text |
| 6 | Insight: adjacent layers share 70–100% of top-k | turn | fig_overlap (SVG, schematic) |
| 7 | ❶ F layers index, S layers reuse the cache | overview + mechanism | fig_overview (SVG) |
| 8 | ❷ Greedy search keeps the indexers that matter | mechanism | fig_patterns (SVG, real patterns) |
| 9 | ❸ Distil each indexer for every layer it serves | mechanism | fig_distill (SVG) |
| 10 | GLM-5.2's MTP drafts reuse indices and KV | mechanism (production) | chart_mtp (bars) |
| 11 | 1/4 indexers: 1.82× prefill, 1.48× decode | release | chart_speedup (2-panel bars) |
| 12 | Searched or trained patterns lose ≤ 0.4 points | result | chart_quality (Δ bars) |
| 13 | GLM-5.2 ships it: 2.9× fewer FLOPs at 1M | adoption | `%% big: 2.9×` (tone: ours) |
| 14 | The cost: search, retraining, a residual O(L²) | limits | two-col text |
| 15 | Summary | | QR |
