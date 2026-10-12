# Story: KDA Context Parallelism (KCP)

Deck 69 of the KDA chain: 05 model-level attention -> 08 FlashKDA kernel -> **69 KDA Context Parallelism**.
Deck 05 owns the recurrence, deck 08 the single-GPU kernel (and the intra-device SM split that uses the same
decomposition); this deck owns what crosses GPUs. Softmax CP background: deck 68.

- **Where it lives** (prosperity): Kimi K3 pre-training at up to 1M tokens (context grows 8K -> 64K -> 256K -> 1M,
  K3 §3.4) with PP + EP + ZeRO + CP (K3 §5.2); 69 of 93 attention layers are KDA. The sequence dimension is split
  across GPUs, so each KDA layer needs the state that enters its local segment.
- **Pain number** (betrayal): the obvious CP for linear attention, head-split all-to-all, reaches only
  **55% scaling efficiency** on 4 GPUs for KDA forward+backward (37% forward), 32K tokens, H800
  (FLA PR #691 benchmark). Every GPU still runs the full-length recurrence for a subset of heads.
- **Failed suitors**:
  1. Softmax-style CP (ring / all-gather KV): ships KV blocks that grow with sequence length (K3 §5.1.2;
     Megatron CP doc). Linear attention has no KV cache to ship.
  2. Head-split all-to-all CP (Ulysses-style): CP degree <= number of heads, and the serial recurrence is not
     shortened; 55% / 37% efficiency at CP = 4 (PR #691).
  3. LASP-style state summation: compute each rank's state from S = 0 and sum the preceding ranks' states. Exact
     for vanilla linear attention, wrong for KDA because the delta rule multiplies the incoming state by
     M_t = (I - beta k k^T) Diag(alpha) (K3 §5.1.2).
- **Villain**: the delta rule makes a segment's effect depend on the state entering it.
- **Turn / insight**: the dependence is affine: S_out = M S_in + S~, with M (d_k x d_k) and S~ (d_k x d_v)
  computable from the local tokens alone (K3 Eq. 17). Affine maps compose associatively, so a prefix scan over
  fixed-size fragments recovers every rank's entry state exactly.
- **Mechanisms**:
  1. ❶ Local fragments: each rank builds M (chained from I over its chunks) and S~ (the normal chunk recurrence
     from 0) before any communication.
  2. ❷ One all-gather + in-order merge: gather [S~, M] (H x d_k x (d_k + d_v), FP32 in FLA), then rank i applies
     S <- M_j S + S~_j over preceding fragments; backward does the same in reverse with transposed M.
  3. ❸ Real batches: packed documents (merge only over fragments of the same document; only the first local
     sequence continues), ShortConv halo (tail tokens from the previous rank), FP32 M chain.
- **Release**: same currency as the pain: **86% scaling efficiency** (fwd+bwd) and 76% (forward) at CP = 4, vs
  55% / 37% for all-to-all (PR #691: 36.19 ms -> 10.49 ms vs 16.42 ms). Communication per layer is independent
  of T: analytic 512x less than a KV all-gather at 1M tokens on 8 ranks (MHA-equivalent comparator).
- **Limits**: payload and serial merge grow with CP degree; small numerical differences ("lossy" in PR #691),
  FP32 chain required; extra pre-process kernels + conv halo; the 24 MLA layers still need KV-exchanging CP;
  public numbers come from FLA's independent implementation (32K, CP = 4); K3 publishes no KCP measurements.

## Slide plan (final, 13 logical slides incl. title)

| # | Headline |
|---|---|
| 1 | Title |
| 2 | K3 splits million-token sequences across GPUs |
| 3 | Head-split CP gets 55% out of four GPUs |
| 4 | Three ways to split a sequence fall short |
| 5 | Insight: a segment is an affine map of S |
| 6 | KCP: local fragments, one all-gather, local merge |
| 7 | ❶ Each rank builds both fragments on its own |
| 8 | ❷ One all-gather, then a local in-order merge |
| 9 | ❸ Packed documents and convolution halos |
| 10 | KCP keeps 86% efficiency on 4 GPUs |
| 11 | Analytic: 512× less traffic than KV at 1M |
| 12 | Costs: grows with GPU count, MLA still ships KV |
| 13 | Summary |
