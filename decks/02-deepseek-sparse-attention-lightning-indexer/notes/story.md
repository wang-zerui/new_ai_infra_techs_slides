# Story — deck 02: DeepSeek Sparse Attention (DSA) + lightning indexer

- **Where it lives** (prosperity): DeepSeek-V3.2 (V3 architecture: 671B MoE, 37B active, MLA with 128 heads,
  61 layers, 128K context). DSA is the *only* architectural change from V3.1-Terminus, added by continued
  training. GLM-5 (744B, 40B active) adopts it; GLM-5.2 (IndexShare, deck 04) and DeepSeek-V4 (CSA, deck 01)
  build on it.
- **Pain number** (betrayal): with dense MLA, the serving cost of a decoded token grows linearly with its
  position: ≈ $0.07 → ≈ $2.15 per million tokens from position 0 to 128K, **31×** (H800 service, DeepSeek-V3.2
  Fig. 3, read from the figure's vector data). Same currency as the release.
- **Failed suitors**:
  1. Fixed sparsity (sliding windows) and linear attention: even keeping half the layers full, GLM-9B loses up
     to 30.35 RULER points at 128K (SWA interleave), 5.69 with a searched layer pattern, 8–11 with linear attention
     (GLM-5 Table 5). Fixed patterns cannot follow content.
  2. Natively sparse pretraining (NSA-style): needs training from scratch; GLM-5 calls that cost "astronomical".
- **Villain**: dense attention uses the expensive 128-head attention itself to decide which keys matter, so
  every cached key costs full price: per key, 139,264 MACs in MQA mode (analytic).
- **Turn / insight**: scoring a key is 17× cheaper than attending to it (64 × 128 FP8 MACs vs 128 × 1,088 BF16
  MACs, analytic): let a tiny FP8 scorer, trained to imitate the attention, read every key, and let the attention
  read only the top 2,048 (1.6% of a 128K context).
- **Mechanisms**:
  ❶ Lightning indexer: I(t,s) = Σ_j w(t,j)·ReLU(q(t,j)·k(s)), 64 heads × 128 dims, one shared key per token, FP8
     (answers: scoring all L keys must cost far less than attention).
  ❷ Top-k + sparse MLA in MQA mode: one 576-d latent per token shared by all 128 query heads, so one gathered
     entry serves every head (answers: a per-query gather must stay kernel-efficient). Prefill (contiguous
     [ks, ke) ranges) and decode (paged cache) take different indexer kernels (deck 10).
  ❸ Dense → sparse continued training: indexer-only warm-up by KL distillation to the head-summed attention
     (2.1B tokens), then sparse training of everything with a detached indexer (943.7B tokens)
     (answers: an untrained indexer selects garbage; training from scratch is unaffordable).
- **Release**: at 128K, ≈ 8.7× lower decode cost per token (≈ $2.15 → ≈ $0.25 per M tokens) and ≈ 3.6× lower
  prefill cost; API prices cut by 50%+; quality on par (V3.2-Exp +4 on AA-LCR; GLM-4.7-Flash RULER@128K 79.21 →
  78.86 after DSA).
- **Limits**: indexer is still O(L²) and dominates DSA's per-query work at 128K (79% of MACs, analytic → deck 04,
  deck 10); below ≈ 5K (decode) / ≈ 10K (prefill) tokens DSA costs slightly more (short prefill uses a masked MHA
  mode instead); needs a 944B-token continued pretraining (GLM-5: 20B); +132 B/token/layer indexer key cache
  (+20% on MLA's 656 B); RL needs a deterministic top-k and a frozen indexer (GLM-5).

## Slide plan (16 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | DSA is V3.2's only architectural change | prosperity | fig_lineage (SVG) |
| 3 | Dense attention makes late tokens 31× costlier | pain | `%% big: 31×` |
| 4 | Every token's attention reads every cached latent | villain | fig_dense (SVG) |
| 5 | Fixed patterns and linear attention lose recall | failed suitors | two-col text |
| 6 | Insight: scoring is 17× cheaper than attending | turn | chart_perkey (analytic) + boxed sentence |
| 7 | DSA puts a cheap scorer in front of MLA | overview | fig_overview (SVG) |
| 8 | ❶ The lightning indexer scores every key in FP8 | mechanism | fig_indexer_1/2 (SVG build) |
| 9 | ❷ 2,048 shared latents serve all 128 heads | mechanism | fig_select (SVG) |
| 10 | Prefill and decode call different indexer kernels | mechanism detail | fig_paths (SVG) |
| 11 | ❸ Warm up the indexer, then train sparse | mechanism | fig_training (SVG) |
| 12 | DSA cuts 128K decode cost 8.7× | release | chart_cost (2-panel line) |
| 13 | Quality holds: 128K RULER within 0.35 points | result | chart_ruler (line) |
| 14 | GLM-5 adopts DSA with 47× fewer tokens | adoption | chart_glm5 (Table 3) |
| 15 | The cost: an O(L²) indexer and new state | limits | two-col text |
| 16 | Summary | | QR |
