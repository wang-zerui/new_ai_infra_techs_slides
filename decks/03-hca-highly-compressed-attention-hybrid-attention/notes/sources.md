# Sources and number trace — deck 03 (HCA + hybrid attention stacking)

## Sources read

| Tag | Source | Where |
|---|---|---|
| [V4] | DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence", arXiv:2606.19348v1 (26 Apr 2026). https://arxiv.org/abs/2606.19348 (cached text `$CACHE/arxiv-2606.19348.txt`; PDF pages rendered for Fig. 1, 4, 6, 9) | Abstract; §1 p.4-5; Fig. 1; §2.3 p.9; §2.3.2 HCA p.11-12 (Eq. 20-26, Fig. 4); §2.3.3 p.12-13; §2.3.4 p.13; §3.4.3 p.20-21; §3.5.1 p.21-23 (Fig. 6); §3.5.2 p.23-24; §4.2.1 p.24-25; Table 6 p.38; §5.3.2 + Fig. 9 p.39-40; §6 p.44 |
| [V3.2] | DeepSeek-AI, DeepSeek-V3.2, arXiv:2512.02556 (cached) | §2.1 (DSA under MLA); p.10 (128K max context) |
| [V3] | DeepSeek-V3 Technical Report, arXiv:2412.19437 (cached) | §2.1.1 (MLA caches c^KV and k^R only); §4.2 (d_c = 512, d_h^R = 64) |
| [cfg-Pro], [cfg-Flash] | `config.json` of huggingface.co/deepseek-ai/DeepSeek-V4-Pro and -Flash (downloaded 2026-10-13 to `$SCRATCH/research/01/`) | `compress_ratios` (layer order), `sliding_window` 128, `head_dim` 512, `qk_rope_head_dim` 64 |
| [code] | V4 reference inference code (`inference/model.py`, linked from [V4] footnote 1) | `Compressor` (overlap only when ratio == 4; HCA layers use `get_compress_topk_idxs` = all preceding compressed entries; no `Indexer` when ratio == 128); `Attention` keeps a 128-entry window ring buffer in every layer |

## Numbers on slides

| Slide | Number | Source |
|---|---|---|
| 2 | Every V4 layer is softmax attention over KV entries: sliding window (SWA), CSA or HCA | [V4] §2.3, §4.2.1; [cfg-*] `compress_ratios` ∈ {0, 4, 128} |
| 2, 7 | SWA reads the last 128 raw tokens; CSA reads top-k of n/4 + 128; HCA reads all n/128 + 128 | [V4] §2.3.1, §2.3.2, §2.3.3 (n_win = 128, §4.2.1) |
| 3 | **262 GB**: KV cache of one 2²⁰-token request for BF16 GQA8 (8 KV heads × 128 dims × K,V × 2 B = 4,096 B per token per layer) at 61 layers — **analytic** | `notes/analytic.py`; baseline configuration named by [V4] §2.3.4 ("BF16 GQA8 ... head dimension of 128"); 61 layers from [V4] §4.2.1 |
| 4 | V3.2 caches one 576-dim MLA entry per token per layer | [V3] §2.1.1, §4.2; [V3.2] §2.1 |
| 4 | CSA alone stores 1 entry per 4 tokens per layer | [V4] §2.3.1 |
| 5 | 2²⁰ / 128 = 8,192 entries per HCA layer at 1M | Derived from m′ = 128 ([V4] §4.2.1) |
| 6-8 | m′ = 128, no overlap, no indexer; c = 512; 128 query heads (Pro) / 64 (Flash); same shared-KV MQA, partial RoPE, sink and grouped output projection as CSA | [V4] §2.3.2 Eq. 20-26; §2.3.3; §4.2.1 |
| 7 | Each entry = per-channel softmax over 128 token projections + learned positional bias B ∈ R^{m′×c} | [V4] Eq. 22-23 |
| 8 | Per query at 2²⁰ tokens (analytic): HCA layer 8,192 + 128 entries → 1.09 G multiply-adds (V4-Pro: 128 heads × (512 + 512)); CSA layer 2.30 G; V3.2 DSA layer 8.88 G | `notes/analytic.py` (same counting rules as deck 01) |
| 9 | Pro layer order: L0-L1 HCA, then CSA/HCA alternate (30 CSA + 31 HCA); Flash: L0-L1 SWA, then alternate (21 CSA + 20 HCA) | [V4] §4.2.1; exact counts [cfg-Pro]/[cfg-Flash] |
| 9 | State cache (fixed per request): SWA window + uncompressed tail; block cache: each block covers lcm(m, m′) = lcm(4, 128) = 128 tokens → k1 = 32 CSA entries and k2 = 1 HCA entry per layer | [V4] §3.5.1, Fig. 6 caption |
| 9 | Bar widths in the layout figure ∝ bytes: 32 × 576 B (CSA main) vs 32 × 64 B (FP4 indexer keys) vs 1 × 576 B (HCA) | Derived; byte model below |
| 10 | KV cache at 1M: V4-Pro 10% of V3.2 (9.5× smaller), V4-Flash 7% (13.7× smaller) | [V4] Abstract; §1 p.5; Fig. 1 right (labels "9.5× smaller", "13.7× smaller") |
| 11 | V4 KV ≈ 2% of a BF16 GQA8 (head dim 128) baseline at 1M | [V4] §2.3.4 p.13 ("approximately 2%") |
| 11 | Analytic reproduction: V4-Pro mix 5.2 GB per 2²⁰-token request (4,944 B/token) vs GQA8 262 GB (249,856 B/token) = 1.98%; Flash 1.96%; hypothetical all-CSA 10.2 GB, all-HCA 0.3 GB; CSA layers hold 97% of the mix | `notes/analytic.py`. Byte model: entry = 64 RoPE dims × 2 B (BF16) + 448 dims × 1 B (FP8) = 576 B ([V4] §2.3.3 RoPE on last 64 dims; §2.3.4 BF16 RoPE dims / FP8 rest); indexer key 128 dims in FP4 = 64 B ([V4] §4.2.1 c_I = 128; §5.2.1 QK cached in FP4); CSA = 576/4 + 64/4 = 160 B/token/layer; HCA = 576/128 = 4.5 B; window 128 × 576 B per layer (fixed). Excludes quantization scale factors, tail buffers and the MTP layer |
| 12 | MRCR 8-needle average MMR, V4-Pro-Max: 0.90 / 0.85 / 0.94 / 0.90 / 0.92 / 0.82 / 0.66 / 0.59 at 8K…1M; V4-Flash-Max: 0.91 / 0.84 / 0.87 / 0.85 / 0.87 / 0.76 / 0.60 / 0.49 | [V4] Fig. 9 p.40 (series identity checked on the rendered page) and text p.39-40 ("highly stable within a 128K context window", degradation "beyond the 128K mark") |
| 12 | MRCR 1M (MMR): V4-Pro-Max 83.5, Gemini-3.1-Pro 76.3, Claude Opus 4.6 92.9 | [V4] Table 6 p.38 |
| 13 | On-disk prefix cache: SWA KV volume ≈ 8× the compressed CSA/HCA entries; strategies: full SWA caching, periodic checkpointing every p tokens, zero SWA caching (recompute last n_win · L tokens) | [V4] §3.5.2 p.23-24 |
| 13 | PagedAttention assumptions violated (diverse cache policies; kernel alignment) | [V4] §3.5.1 |
| 13 | CP: two-stage exchange because compression windows straddle rank boundaries | [V4] §3.4.3 |
| 13 | Paper's own limitation: architecture "relatively complex"; no published ablation of the CSA:HCA ratio or of m′ | [V4] §6 p.44; absence checked in §2.3 and §4 |
| 14 | FLOPs at 1M: V4-Pro 27%, V4-Flash 10% of V3.2 (context for the summary) | [V4] Abstract |

## Discrepancies (Notion row vs primary source; the primary source wins on the slides)

1. **Name.** Notion calls it "highly-compressed attention"; [V4] names it **Heavily Compressed Attention
   (HCA)** (Abstract, §2.3.2). Slides use the paper's name.
2. **"every layer remains attention - no SSM or Gated-DeltaNet-style linear attention".** True of the
   released configs (every layer is SWA, CSA or HCA), but [V4] does not argue against SSMs or linear
   attention; it actually describes managing the SWA window and uncompressed tails "as a state-space
   model" for cache purposes (§3.5.1). Slides state only that every layer is softmax attention.
3. **Attribution of 10% / 7% KV.** Notion credits HCA alone; [V4] credits the hybrid CSA + HCA design
   together with low-precision storage (BF16 RoPE dims, FP8 elsewhere) (§1 p.5; §2.3.4). The slides say
   "whole V4 stack". Analytic model: CSA layers hold ~97% of the mixed cache, so HCA's contribution is to
   make half the layers nearly free, not to shrink the cache by itself.
4. **"interleaved with CSA".** Correct, with detail Notion omits: V4-Pro's first two layers are HCA and
   V4-Flash's first two are pure sliding-window ([V4] §4.2.1).
5. **"making 1M context routine".** The paper's wording is "routinely support one-million-token contexts"
   (Abstract). Retrieval quality is not flat to 1M: MRCR 8-needle drops from 0.92 at 128K to 0.59 at 1M
   (Fig. 9). The slides show this limit.
