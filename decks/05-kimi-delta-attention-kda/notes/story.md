# Story: Kimi Delta Attention (KDA)

Deck 05 of the KDA chain: **05 model-level attention** -> 08 FlashKDA kernel -> 69 KDA Context Parallelism.
This deck owns the math (state update, chunkwise form, hybrid layout); kernel scheduling belongs to deck 08,
cross-device communication to deck 69.

- **Where it lives** (prosperity): Kimi K3, 2.8T-parameter MoE (104B active), 1M-token context; 69 of its 93
  attention layers are KDA, 24 are Gated MLA (K3 Table 1, §2.1).
- **Pain number** (betrayal): full attention decodes at **11.48 ms per output token** at 1M context
  (MLA baseline, Kimi Linear 48B-A3B setting, Kimi Linear Fig. 1b). The KV cache and the 2T^2 d_h attention
  FLOPs grow with every token.
- **Failed suitors**:
  1. Softmax attention (K2 used 61 MLA layers): exact, but cache O(T) and FLOPs O(T^2).
  2. Vanilla linear attention (S += k v^T): fixed d_k x d_v state but never forgets; with d_k = 128 at most
     128 orthogonal keys fit, beyond that writes collide (GDN paper §1; Kimi Linear §2.2).
  3. Scalar gates (Mamba2 / GDN): forget a whole head at one rate; Mamba2 has no delta rule and fails the
     synthetic recall tasks in the Kimi Linear tests.
- **Villain**: a fixed-size memory (128 x 128 per head) facing a 1M-token stream with additive, unconditional
  writes: nothing decides what to keep, what to overwrite, and at which per-feature rate to forget.
- **Turn / insight**: the delta rule is one step of online gradient descent on ||S^T k - v||^2, so after a
  per-channel decay the write is only the prediction error: S_t = S' + beta k (v - S'^T k)^T, S' = Diag(alpha) S.
  The state stays d_k x d_v regardless of length.
- **Mechanisms**:
  1. ❶ Gated delta rule with channel-wise decay (challenge: memory collisions / uniform forgetting).
  2. ❷ Chunkwise parallel form + lower-bounded decay (challenge: the recurrence is serial; 1/Gamma overflows
     in BF16 so diagonal tiles fell off Tensor Cores). K3 bounds g >= -5 so all tiles are dense matmuls.
     Kernel detail: deck 08.
  3. ❸ 3:1 hybrid with NoPE Gated MLA (challenge: a finite state cannot do exact long-range retrieval; RoPE
     would need retuning for 1M). Ablation: 3:1 has the best validation PPL (5.65).
- **Release**: in the pain's currency: **1.84 ms vs 11.48 ms per output token at 1M (6.3x)**, with better
  quality at equal 1.4T tokens (MMLU-Pro 51.0 vs 47.2, RULER@128K 84.3 vs 81.3) and up to 75% less KV cache.
  At K3 scale: approx. 2.5x overall scaling efficiency over K2 (KDA + AttnRes + Stable LatentMoE + recipes,
  not KDA alone), 1M context with no positional-encoding change.
- **Limits**: speed-ups appear only at long context (prefill comparable to MLA at 4K-16K); 24 MLA layers still
  keep a KV cache; serial state needs dedicated kernels (deck 08) and a new CP scheme (deck 69); prefix caching
  needs state checkpoints at hash boundaries and speculative decoding needs state replay (K3 §5.4); the
  measured efficiency numbers come from the 48B Kimi Linear model, K3 publishes no KDA-only ablation.

## Slide plan (final, 16 logical slides incl. title)

| # | Headline |
|---|---|
| 1 | Title |
| 2 | Kimi K3 runs 69 of 93 attention layers on KDA |
| 3 | Full attention pays for every past token |
| 4 | Growing caches and colliding writes both fail |
| 5 | Insight: write only what memory gets wrong |
| 6 | One KDA layer: gated inputs, one fixed state |
| 7 | ❶ Each token: decay, predict, correct |
| 8 | ❶ Per-channel decay also encodes position |
| 9 | ❷ Chunks: parallel inside, recurrent across |
| 10 | ❷ Bounded decay: all tiles on Tensor Cores |
| 11 | ❸ Three KDA layers per global MLA layer |
| 12 | Same 1.4T tokens: KDA hybrid beats full MLA |
| 13 | Decodes 6.3× faster than MLA at 1M tokens |
| 14 | At K3 scale: 2.5× scaling efficiency over K2 |
| 15 | Costs: long context only, new kernels and caches |
| 16 | Summary |
