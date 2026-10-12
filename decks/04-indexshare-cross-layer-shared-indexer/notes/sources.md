# Sources — deck 04: IndexShare / IndexCache (cross-layer index reuse)

## Sources read

| Tag | Source | Where |
|---|---|---|
| [IC] | Y. Bai, Q. Dong, T. Jiang, X. Lv, Z. Du, A. Zeng, J. Tang, J. Li (Tsinghua University, Z.ai), "IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse", arXiv 2603.12201v1 (12 Mar 2026). https://arxiv.org/abs/2603.12201 (cached text + PDF) | Abstract + Fig. 1 caption (p.1); §1 + indexer-time figure (p.2); Fig. 2 inference loops (p.3); §2 (p.3-4); §3.1 + Alg. 1 + greedy-loss figure (p.4-5); §3.2 Eq. 1-3, Prop. 1 (p.6); Table 1 + §4.1-4.2 (p.7-8); Fig. 3 (p.8); Table 2 (p.8); Table 3, §4.3-4.4 (p.9); Table 4, §4.5 (p.10); §5.2 related work (p.11); App. A overlap (p.16); App. B searched patterns (p.17); App. C similarity search (p.17-18) |
| [G52R] | zai-org/GLM-5 GitHub README (fetched 2026-10-13; GLM-5.2 section) | "Improved Architecture" bullet: IndexShare, every four sparse attention layers, 2.9× per-token FLOPs at 1M, MTP acceptance length up to +20% |
| [G52B] | Z.ai blog "GLM-5.2" (https://z.ai/blog/glm-5.2; text extracted from the page's JS bundle, fetched 2026-10-13) | "Architecture for 1M Context": IndexShare for DSA (indexer at the first of every 4 layers, 3/4 of layers skip indexer dot product and top-k, trained with IndexShare from mid-training at 128K, beats GLM-5.1 on long-context benchmarks); figure "Lower FLOPs with IndexShare" (single-token FLOPs vs position, 2.9× lower at 1M); "MTP with IndexShare and KVShare" + acceptance-length table; "Efficiently Serving 1M Context Length" (context 200K → 1M; KV cache not proportionally reduced) |
| [HF52] | HuggingFace `zai-org/GLM-5.2` config.json (copy: notes/hf_config_GLM-5.2.json) | num_hidden_layers 78; indexer_types (21 "full", 57 "shared": FFF then FSSS × 18); index_topk 2048; index_topk_freq 4; index_skip_topk_offset 3; index_n_heads 32; index_head_dim 128; index_share_for_mtp_iteration true; max_position_embeddings 1,048,576 |
| [HF5] | HuggingFace `zai-org/GLM-5` config.json (copy: notes/hf_config_GLM-5.json) | 78 layers, every layer an indexer (no indexer_types), max_position_embeddings 202,752 |
| [TF] | huggingface/transformers `models/glm_moe_dsa/modular_glm_moe_dsa.py` (main, fetched 2026-10-13) | `GlmMoeDsaConfig.indexer_types` ("full" runs the indexer, "shared" reuses the previous full layer's top-k); `GlmMoeDsaAttention`: `self.indexer = None` on shared layers, comment "Refer: https://arxiv.org/abs/2603.12201", `prev_topk_indices` passed layer to layer |
| [GLM5] | GLM-5 technical report, arXiv 2602.15763v2 (24 Feb 2026) | adopts DSA (§1, §2.1.1); Table 10 (744B, 40B active, 32 indexer heads × 128); no mention of IndexShare or IndexCache |
| [V32] | DeepSeek-V3.2 report, arXiv 2512.02556 | DSA definition, k = 2,048 (context only; deck 02) |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 744B / 40B active; 78 layers; 32-head × 128 indexer | GLM-5 (and GLM-5.2) backbone | [GLM5] Table 10; [HF5]; [HF52] |
| 200K → 1M | GLM-5.2 max context vs GLM-5 / 5.1 | [G52B] "Efficiently Serving 1M"; [HF52] 1,048,576 vs [HF5] 202,752 |
| 2,048 | tokens selected per query (top-k) | [IC] §2.1, §3 ("k = 2048 throughout"); [HF52] |
| 81% (and 27%, 50%, 68% at 10K / 60K / 120K) | indexer share of prefill time, 30B DSA model | [IC] §1 figure "Indexer Time Proportion" (values printed on the figure) |
| 27%, 31%, 38%, 41% | indexer share of decode time at 10K / 60K / 120K / 200K | same figure |
| 30B-A3B, 47 layers, MLA; H100 node, SGLang, DP attention 8 | test model and setup | [IC] §4.1-4.2 |
| 70–100% (0.7–1.0) | top-k overlap between adjacent layers; ≤ 0.4 between early and late layers | [IC] §1 and App. A (768 samples × 200K) |
| Long Avg 50.2 → 47.4 / 43.0 / 35.3 | uniform interleave, training-free, at 1/2 / 1/4 / 1/8 retention (Δ −2.8 / −7.2 / −14.9) | [IC] Table 2 (Δ derived) |
| searched 50.3 / 49.9 / 46.1 | greedy-searched pattern at 1/2 / 1/4 / 1/8 (Δ +0.1 / −0.3 / −4.1) | [IC] Table 2 (Δ derived) |
| training-aware uniform 51.6 / 50.6 vs 51.0 | 1/2 / 1/4 retention vs its own DSA baseline (Δ +0.6 / −0.4); w/o cross-layer loss 49.8 (−1.2) | [IC] Table 3 (Δ derived) |
| G&R within 1 point (73.7–74.9 vs 74.6) | general & reasoning average, training-free configs except uniform 1/8 | [IC] §4.3 |
| N(N−1)/2 forward passes; 1,081 (N = 47); 3,003 (N = 78) | cost of the full greedy search | [IC] §3.1.2; counts derived |
| 0.57 / 3.38 / 8.57 / 19.5 s → 0.45 / 2.59 / 5.66 / 10.7 s | prefill time DSA vs IndexCache 1/4 at 10K / 60K / 120K / 200K (1/2: 0.47 / 2.86 / 6.57 / 13.7) | [IC] Table 1 |
| 1.27 / 1.31 / 1.51 / 1.82× (1/4); 1.21 / 1.18 / 1.30 / 1.42× (1/2) | prefill speedup | [IC] Fig. 3(a); equals Table 1 ratios |
| 73.5 / 67.0 / 63.0 / 58.0 → 91.0 / 89.5 / 88.0 / 86.0 tok/s | per-request decode DSA vs 1/4 (1/2: 84.5 / 80.0 / 77.0 / 73.0) | [IC] Table 1 |
| 1.24 / 1.33 / 1.40 / 1.48× (1/4); 1.15 / 1.19 / 1.22 / 1.26× (1/2) | per-request decode speedup | [IC] Fig. 3(b) |
| 197 → 297 tok/s (1.51×); +22–51% | full-KV decode throughput at 200K, range across lengths (1/4) | [IC] Table 1, §4.2 |
| ≥ 1.3× | GLM-5 (744B), IndexCache 1/4, prefill latency and decode throughput beyond 100K | [IC] §4.2 |
| ~1.2× end-to-end, 50% of indexers removed | GLM-5 + IndexCache 1/2 | [IC] Fig. 1 caption |
| GLM-5 Long Avg 78.4 → 72.7 (1/4 uniform) / 78.0 (1/4 searched); 78.1 / 78.7 at 1/2 | 744B training-free | [IC] Table 4 |
| 21 of 78 (57 shared); FFF + (FSSS) × 18 | GLM-5.2 indexer layout | [HF52] indexer_types (counted) |
| 22 of 78 | GLM-5 greedy-searched 1/4 pattern | [IC] App. B (counted) |
| 2.9× | lower per-token FLOPs at 1M context, GLM-5.2 vs GLM-5.1 | [G52R]; [G52B] figure label |
| 4.56 → 5.10 → 5.29 → 5.47 (+20%) | MTP acceptance length: baseline, + IndexShare + KVShare, + rejection sampling, + end-to-end TV loss (7 MTP steps, coding, GLM-5.1 backbone) | [G52B] table |
| +12% | 5.10 / 4.56 = 1.118, IndexShare + KVShare alone | derived from [G52B] |
| ≈ 2.7× (notes only) | analytic per-token FLOPs ratio at 2²⁰ tokens: (78 vs 21 indexers × 32 × 128 × L × 2) + 78 × 64 × (576 + 512) × 2,048 × 2 + 2 × 40B; consistent with the published 2.9× | analytic, notes only (`notes/analytic.py`) |

## Mechanism facts (no numbers) used on slides

- IndexCache inference loop: F layers compute I, top-k, write T_cache; S layers set T ← T_cache; T_cache is a temporary
  buffer overwritten at each F layer, "no additional GPU memory" ([IC] Fig. 2 caption). The first layer is always F.
- S layer inherits from the nearest preceding F layer f(ℓ) = max{j < ℓ : c_j = F} ([IC] §3).
- In GLM-5.2 / transformers, shared layers are built without an indexer module (no indexer weights, no indexer key
  cache written), and the top-k indices are passed upward layer by layer ([TF]).
- Greedy search: start all-F, at each step tentatively flip every remaining F layer (except layer 1) to S, keep the flip
  with the lowest calibration LM loss; with P pipeline stages, search per block for ~P× fewer passes ([IC] §3.1.2,
  Alg. 1). Calibration: SFT data, batch 768, 200K context ([IC] §4.1).
- Multi-layer distillation: L_multi = Σ_{j=0..m} 1/(m+1) Σ_t KL(p^(ℓ+j)_t ‖ q^(ℓ)_t); its gradient equals that of KL
  against the averaged target p̄ (Proposition 1) ([IC] §3.2).
- Uniform interleave in the training-free setting removes critical layers; searched patterns place F layers densely in
  early/transition layers ([IC] §3.1.1, App. A-B).
- GLM-5.2: indexer at the first of every 4 layers; trained with IndexShare from mid-training at 128K; MTP: indexer on
  the first MTP step, its top-k indices and KV cache reused by later steps, removing the mixed target/draft KV in later
  steps ([G52B]).

## Discrepancies

1. **Name.** Notion calls the technique "IndexShare" and cites arXiv 2603.12201, whose title says **IndexCache**. The
   GLM-5.2 README and blog use **IndexShare** and link to the same arXiv paper. The GLM-5 technical report
   (2602.15763v2, the "GLM-5 series" source) mentions neither; IndexCache's paper only reports *preliminary* GLM-5
   experiments. Slides: "IndexShare (paper: IndexCache)".
2. **"Every four sparse-attention layers"** describes the GLM-5.2 production layout (uniform, trained). The paper's
   headline training-free results use a *searched, non-uniform* pattern, and the GLM-5.2 config keeps the first three
   layers all Full (21 indexers in 78 layers, not 78/4 = 19.5). Slides show both.
3. **"Improved MTP adds up to 20% more accepted length"**: the +20% (4.56 → 5.47) combines IndexShare + KVShare with
   rejection sampling and an end-to-end TV loss, measured on the GLM-5.1 backbone with 7 MTP steps; IndexShare +
   KVShare alone gives 4.56 → 5.10 (+12%, derived). Slides attribute it this way.
4. **"2.9× lower per-token FLOPs at 1M"** is a GLM-5.2 vs GLM-5.1 model-level FLOPs count from Z.ai's blog figure, not
   a measured speedup; the paper's measured speedups are 1.82× (prefill) / 1.48× (decode) at 200K on a 30B model.
   Slides keep the two separate.
