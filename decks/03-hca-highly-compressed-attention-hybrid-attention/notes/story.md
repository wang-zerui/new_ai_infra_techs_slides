# Story — deck 03, HCA (Heavily Compressed Attention) + hybrid stacking

Sibling of deck 01 (CSA). Deck 01 tells the **compute** story (who selects, at what cost per query);
this deck tells the **memory** story (what each layer keeps, and why mixing layer types pays).

- **Where it lives** (prosperity): DeepSeek-V4 (Pro 1.6T / 49B active, Flash 284B / 13B active), 1M-token
  context. Every layer is softmax attention at one of three resolutions: SWA (raw last 128), CSA (1 per 4
  tokens, selected), HCA (1 per 128 tokens, read densely). HCA is 31 of 61 Pro layers, 20 of 43 Flash.
- **Pain number** (betrayal): one 1M-token request needs **262 GB** of KV cache with a common BF16 GQA8
  layout at V4-Pro's depth (analytic, the paper's own baseline configuration).
- **Failed suitors**: (1) narrower entries (GQA, MQA, MLA: V3.2 stores a 576-dim latent) shrink each cell
  but still keep one per token per layer; (2) CSA everywhere (deck 01) divides the count by 4 but stays
  linear with a 4× constant, plus indexer keys; (3) window-only attention is constant-size but forgets
  far context.
- **Villain**: uniform resolution — every layer stores the whole context at the same granularity, so the
  cache is layers × tokens × bytes per entry.
- **Turn / insight**: not every layer needs fine detail. At 128× compression a 1M-token context is just
  8,192 entries per layer: few enough to read all of them densely, with no selector. Interleave such
  coarse dense layers with CSA's selective layers.
- **Mechanisms**:
  ❶ Heavy compressor — challenge: a 128-token summary must keep what matters → per-channel softmax
     pooling over 128 tokens with a learned positional bias, no overlap.
  ❷ Dense shared-KV MQA over all entries + the 128-token window — challenge: no selector and no local
     detail at 128× → read every entry, add raw recent tokens; same MQA/sink/grouped output as CSA.
  ❸ Hybrid stack + two-part KV cache — challenge: layers with different compression rates break
     PagedAttention → state cache (window + tails, fixed per request) + 128-token blocks holding 32 CSA
     entries and 1 HCA entry per layer.
- **Release**: whole V4 stack at 1M: KV cache **10%** (Pro) / **7%** (Flash) of V3.2 (9.5× / 13.7×
  smaller); ≈ **2%** of BF16 GQA8 (paper), reproduced analytically (1.98%).
- **Limits**: retrieval fades beyond 128K (MRCR 8-needle 0.92 → 0.59 at 1M, Pro-Max); custom cache
  manager and on-disk prefix cache (SWA KV ≈ 8× compressed volume); CP boundary exchange; no ablation
  of the CSA:HCA mix or m′; "relatively complex".

## Storyboard (14 logical slides, 8 minutes)

| # | Beat | Headline (as built) |
|---|---|---|
| 1 | Title | HCA: Heavily Compressed Attention |
| 2 | Hook | V4 layers see context at three resolutions |
| 3 | Pain (262 GB, analytic) | A 1M-token request outgrows per-token caches |
| 4 | Status quo | Narrower entries still grow with every token |
| 5 | Insight | Key idea: some layers need only the gist |
| 6 | Overview | HCA compresses 128×, reads all, and interleaves |
| 7 | ❶ | ❶ Softmax pooling folds 128 tokens into one |
| 8 | ❷ | ❷ No selector: read every entry plus a window |
| 9 | ❸ | ❸ Interleaving needs a two-part KV cache |
| 10 | Result (10% / 7%) | V4's 1M-token KV cache is 10% of V3.2's |
| 11 | Result (≈2%, analytic) | The mix lands at 2% of a GQA8 cache |
| 12 | Result / limit | Retrieval holds to 128K, then fades |
| 13 | Limits | Cost: coarse memory and custom cache plumbing |
| 14 | Summary | Summary |
