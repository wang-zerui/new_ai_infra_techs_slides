# Sources and number trace — deck 01 (CSA, Compressed Sparse Attention)

## Sources read

| Tag | Source | Where |
|---|---|---|
| [V4] | DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence", arXiv:2606.19348v1 (26 Apr 2026). https://arxiv.org/abs/2606.19348 (cached text `$CACHE/arxiv-2606.19348.txt`, PDF pages rendered for Fig. 1/3/6/9) | Abstract; §1 (p.4-5); Fig. 1 (p.1); §2.3 (p.9); §2.3.1 CSA (p.9-11, Eq. 9-19, Fig. 3); §2.3.3 (p.12-13); §2.3.4 (p.13); §3.4.3 (p.20-21); §3.5.1 (p.21-23, Fig. 6); §4.2.1 (p.24-25); §4.2.2 (p.25-26); Table 1 (p.28); Table 6 (p.38); §6 (p.44) |
| [V3.2] | DeepSeek-AI, "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models", arXiv:2512.02556 (cached `$CACHE/arxiv-2512.02556.txt`) | §2.1 DSA (p.3, Eq. 1-2); §2.1.1 (p.4, top-k 2048); §2.3 Inference costs (p.5); p.10 (128K max context) |
| [V3] | DeepSeek-AI, DeepSeek-V3 Technical Report, arXiv:2412.19437 (cached) | §2.1.1 MLA cache (c^KV and k^R cached); §4.2 hyper-parameters: 61 layers, d_c = 512, d_h^R = 64, 128 heads |
| [cfg-Pro] | `config.json` of huggingface.co/deepseek-ai/DeepSeek-V4-Pro (downloaded 2026-10-13 to `$SCRATCH/research/01/`) | `compress_ratios`, `index_*`, `sliding_window`, `num_attention_heads`, `head_dim` |
| [cfg-Flash] | `config.json` of huggingface.co/deepseek-ai/DeepSeek-V4-Flash (same date) | same fields |
| [cfg-V3.2] | `config.json` of huggingface.co/deepseek-ai/DeepSeek-V3.2 (same date) | `index_n_heads` 64, `index_head_dim` 128, `index_topk` 2048, `kv_lora_rank` 512, `qk_rope_head_dim` 64, 61 layers, 128 heads |
| [code] | V4 reference inference code, huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference (`model.py`, linked from [V4] footnote 1) | `Compressor` (overlap only when ratio == 4), `Indexer`, `Attention.forward` (window indices + top-k indices concatenated; inverse RoPE on output; grouped `wo_a`/`wo_b`) |

## Numbers on slides

| Slide | Number | Source |
|---|---|---|
| 1, 2 | DeepSeek-V4-Pro 1.6T params (49B activated); V4-Flash 284B (13B activated); 1M-token context | [V4] Abstract; §1 p.4 |
| 2 | V4-Pro: 61 layers = 30 CSA + 31 HCA (layers 0-1 HCA, then CSA/HCA alternate); V4-Flash: 43 layers = 21 CSA + 20 HCA + 2 pure sliding-window | [V4] §4.2.1 p.24-25 (first two layers HCA in Pro / pure SWA in Flash, then interleaved); exact counts from [cfg-Pro]/[cfg-Flash] `compress_ratios` (4 = CSA, 128 = HCA, 0 = SWA; last entry is the MTP layer) |
| 2, 14 | V3.2 supports at most 128K context | [V3.2] p.10 ("supports a maximum context length of only 128K"); §2.1.1 |
| 3 | **30×**: DSA indexer multiply-adds vs core-attention multiply-adds per query per layer at 1M tokens (8.59 G vs 0.285 G) — **analytic** | Derived in `notes/analytic.py`: indexer = 64 heads × 128 dims × L; core = 128 heads × (576 QK + 512 PV) × 2048 entries; L = 2^20. Inputs: [cfg-V3.2], [V3.2] §2.1 (indexer scores every preceding token; top-k 2048), [V3] (MLA entry = 512 + 64 dims; DSA runs in MLA's MQA mode, [V3.2] §2.1) |
| 3, 4 | DSA keeps top-k = 2048 entries per query; indexer is O(L²) overall (O(L) per query) | [V3.2] §2.1.1 p.4; §2.3 p.5 |
| 4 | 1,048,576 entries scored per query at 1M (= L) vs 2,048 kept → 512× | Analytic, `notes/analytic.py` |
| 4 | Dense MLA in V3.1-Terminus reads all L entries (DSA's indexer "requires much less computation compared with MLA in DeepSeek-V3.1-Terminus") | [V3.2] §2.1 (V3.2 = V3.1-Terminus + DSA), §2.3 |
| 4 | V3.2 caches one 576-dim entry per token per layer (MLA latent 512 + decoupled RoPE key 64) | [V3] §2.1.1 (only c^KV and k^R are cached) and §4.2 (d_c = 512, d_h^R = 64); [cfg-V3.2] kv_lora_rank 512, qk_rope_head_dim 64 |
| 5 | Insight figure: 16-token toy example, DSA scores 16 units, CSA scores 16/4 = 4 block summaries | Illustration of m = 4 ([V4] §2.3.1); not a measurement |
| 6-9 | m = 4; indexer heads n_h^I = 64; indexer head dim c_I = 128; top-k 1024 (Pro) / 512 (Flash); query heads 128 (Pro) / 64 (Flash); head dim c = 512; d_c = 1536 (Pro) / 1024 (Flash); output groups g = 16 (Pro) / 8 (Flash); d_g = 1024; n_win = 128; hidden d = 7168 (Pro) / 4096 (Flash) | [V4] §4.2.1 p.24-25; confirmed by [cfg-Pro]/[cfg-Flash] |
| 7 | Each compressed entry mixes 2m = 8 tokens (current block via C^a, previous block via C^b), softmax over 2m weights with learned positional biases B^a, B^b ∈ R^{m×c}; sequence shrinks to n/m entries | [V4] §2.3.1 Eq. 9-12, p.9-10 |
| 8 | Index score I_{t,s} = Σ_h w_{t,h} · ReLU(q^I_{t,h} · K^IComp_s); indexer queries from the shared latent c^Q_t = h_t W^DQ; top-k over I_{t,:} | [V4] §2.3.1 Eq. 13-17 |
| 8 | Indexer QK path in FP4 (post-training QAT); index scores FP32 → BF16; top-k selector 2× faster at 99.7% KV-entry recall | [V4] §2.3.4 p.13; §5.2.1 p.33 (details: deck 45) |
| 9 | Shared-KV MQA: each selected 512-d entry is both key and value; 128 uncompressed sliding-window entries concatenated; attention sink; RMSNorm on q and KV; RoPE on last 64 dims + RoPE(−i) on outputs; grouped output projection (g groups → d_g each → d) | [V4] §2.3.1 Eq. 18-19 + "Grouped Output Projection" p.11; §2.3.3 p.12-13 |
| 9 | 1024 + 128 = 1152 entries attended per query (Pro); 512 + 128 = 640 (Flash) | Derived from §4.2.1 values |
| 8 | 64 heads × 128 dims = 8,192 multiply-adds per scored block | Derived from n_h^I = 64, c_I = 128 ([V4] §4.2.1) |
| 9 | Grouped output projection for Pro: 128 heads × 512 = 65,536-dim output in 16 groups → 1,024 each → 7,168 | [V4] §2.3.1 "Grouped Output Projection" + §4.2.1 (g = 16, d_g = 1024, d = 7168); [code] `wo_a`, `wo_b` |
| 10 | Single-token inference FLOPs at 1M context: V4-Pro 27% of V3.2 (3.7× lower), V4-Flash 10% (9.8× lower); FLOPs in equivalent FP8 FLOPs; V4-Pro activates 49B vs V3.2 37B | [V4] Abstract; §1 p.5; Fig. 1 right (3.7× / 9.8× labels); Table 1 (V3.2 37B activated) |
| 11 | Per query, per layer, at 1M (analytic): V3.2 DSA 8.59 G indexer + 0.285 G core = 8.88 G MAC; V4-Pro CSA 2.15 G + 0.151 G = 2.30 G; V4-Flash CSA 2.15 G + 0.042 G = 2.19 G; DSA/CSA-Pro = 3.86× (≈3.9×); CSA-Pro indexer/core = 14× | `notes/analytic.py` (inputs above). Ignores projections, compressor, softmax, top-k and per-op precision (the CSA indexer runs in FP4, which this count does not credit) |
| 11 | Entries attended: 2048 (V3.2) vs 1152 (V4-Pro) = 1.78× fewer | Derived |
| 12 | LongBench-V2 (EM, 1-shot): V3.2-Base 40.2, V4-Flash-Base 44.7, V4-Pro-Base 51.5 | [V4] Table 1 p.28 |
| 12 | MRCR 1M (MMR): V4-Pro-Max 83.5, Gemini-3.1-Pro 76.3, Claude Opus 4.6 92.9 | [V4] Table 6 p.38 |
| 13 | Flash: dense attention for the first 1T tokens (Pro: a longer dense stage); sparse attention introduced at 64K sequence length after a short indexer warm-up; sequence length 4K → 16K → 64K → 1M | [V4] §4.2.2 p.25-26 |
| 13 | Indexer cost O(L²/4) over a prefill: each of L queries scores L/4 compressed keys | Derived from [V4] §2.3.1 (I_{t,s} for s < ⌊t/m⌋) |
| 13 | KV block = lcm(m, m′) = lcm(4, 128) = 128 tokens; tail tokens not yet compressible kept in a per-request state cache | [V4] §3.5.1 p.21-23, Fig. 6 |
| 13 | CP: rank i sends its last m uncompressed KV entries to rank i+1, then all-gather of compressed entries | [V4] §3.4.3 p.20-21 |
| 13 | Paper's own limitation: architecture "relatively complex" | [V4] §6 p.44 |

## Discrepancies (Notion row vs primary source; the primary source wins on the slides)

1. **Attribution of the 27% / 10% FLOPs result.** Notion lists it as CSA's impact. [V4] attributes the
   1M-token FLOPs (and KV) reduction to the whole design: hybrid CSA + HCA together with low-precision
   computation and storage (Abstract; §1 p.5; §2.3.4). No CSA-only ablation is published. The slides say
   "whole V4 stack" when quoting 27% / 10%.
2. **"1M → 250K".** [V4] states the compression factor (sequence length to 1/m with m = 4), not a "250K"
   figure. 1M/4 = 250K (decimal) or 262,144 (with 1M = 2^20). The slides use 1M = 2^20 → 262,144 and say
   "analytic".
3. **"learned overlapping-window weighted average".** Correct in spirit; precisely, each entry is a
   softmax-weighted sum (per channel) over 2m = 8 token projections (C^a of block i, C^b of block i−1)
   with learned positional biases ([V4] Eq. 11-12). The overlap is CSA-only; HCA does not overlap.
4. Notion omits that CSA's top-k (512/1024) is *smaller* than V3.2's DSA top-k (2048) ([V4] §2.3.4) and
   that each selected entry is a compressed block, not a token. Not a contradiction; added on the slides.
