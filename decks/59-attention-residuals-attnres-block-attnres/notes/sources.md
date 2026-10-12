# Sources — deck 59: Attention Residuals (AttnRes / Block AttnRes)

## Sources read

| Tag | Source | Where |
|---|---|---|
| [AR] | Kimi Team, "Attention Residuals", arXiv 2603.15031v1 (cached text + PDF pages 10-11 rendered to check Fig. 5-6) | Abstract + Fig. 1 (p.1); §1 (p.2); §2 (p.3); §3, Eq. 1-4 (p.3-4); §3.2 Eq. 5-6 + Fig. 2 pseudocode (p.4-5); §4.1 Eq. 7-8 + Fig. 3 (p.5-7); Alg. 1 + §4.2 (p.7); Table 1 + prefill (p.8); §5 + §5.1 + Table 2 + Fig. 4 (p.8-9); §5.2 + Fig. 5 + Table 3 (p.9-10); §5.3 Table 4 + Fig. 6 (p.11-12); §5.4.1 Fig. 7 (p.12-13); Fig. 8 (p.13); Table 5 + §6 (p.14); Fig. 9 + §6.2 (p.14-15) |
| [AR-gh] | github.com/MoonshotAI/Attention-Residuals README (master, downloaded 2026-10-13) | same pseudocode; "1.25x more compute"; downstream table |
| [K3] | Kimi Team, "Kimi K3: Open Frontier Intelligence", arXiv 2607.24653 (cached text) | Abstract; Fig. 2 caption (p.3); §2 intro (p.3); §2.2 Eq. 8-10 (p.6); Table 1 (p.11); §5.2.2 "Memory-efficient Attention residual" (p.20); §5.4.2 "Block AttnRes" kernels (p.24) |
| [K3-cfg] | Kimi-K3 HF `config.json` (`attn_res_block_size` 12, `num_hidden_layers` 93) and `modeling_kimi_linear.py` (`KimiDecoderLayer._forward_attn_residual`, `_apply_attn_res`, `KimiLinearModel._apply_output_attn_res`) | AttnRes before attention and before MLP in every decoder layer; block boundary when `layer_idx % 12 == 0`; final output AttnRes |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 1.25× | compute a PreNorm baseline needs to match Block AttnRes's loss (fitted scaling curves) | [AR] §1 p.2, §5.1 p.8-9, Fig. 4; [AR-gh] |
| 1.692 vs 1.714 at 5.6 PFLOP/s-days | Block AttnRes vs baseline loss on the fitted curves | [AR] §5.1 p.9 |
| L = 1.891·C^−0.057 (baseline), 1.870·C^−0.058 (Block), 1.865·C^−0.057 (Full) | fitted laws | [AR] §5.1 p.8 |
| Table 2 losses: 194M / 241M / 296M / 436M / 528M active; Baseline 1.931 / 1.895 / 1.829 / 1.766 / 1.719; Block 1.909 / 1.875 / 1.809 / 1.746 / 1.693; Full 1.899 / 1.874 / 1.804 / 1.737 / 1.692; mHC(-lite) 1.906 / 1.869 / 1.807 / 1.747 / 1.694 | scaling-law runs (8192 context, baseline-tuned hyperparameters) | [AR] Table 2 p.9 |
| Gap Full vs Block 0.001 at the largest size | | [AR] §5.1 p.9 |
| Table 4 (16-layer model): Baseline 1.766; DenseFormer 1.767; mHC 1.747; Full 1.737; w/ input-dependent query 1.731; w/ input-independent mixing 1.749; w/ sigmoid 1.741; Full w/o RMSNorm 1.743; SWA (W = 1+8) 1.764; Block (S = 4) 1.746; w/ multihead 1.752; Block w/o RMSNorm 1.750 | ablations | [AR] Table 4 p.11 |
| Fig. 6: block size S = 32 / 16 / 8 / 4 / 2 → 1.757 / 1.753 / 1.748 / 1.746 / 1.746; Full (S = 1) 1.737; baseline 1.766 | block-size sweep | [AR] Fig. 6 p.11 (data labels, checked on the rendered page) |
| N ≈ 8 blocks recover most of the benefit | | [AR] §3.2 p.5; [K3] §2.2 p.6 |
| Table 3 (Kimi Linear 48B/3B, 1.4T tokens) baseline → AttnRes: MMLU 73.5 → 74.6; MMLU-Pro 52.2 → 52.2; GPQA-Diamond 36.9 → 44.4; BBH 76.3 → 78.0; ARC-C 64.6 → 65.7; HellaSwag 83.2 → 83.4; TriviaQA 69.9 → 71.8; GSM8K 81.7 → 82.4; MGSM 64.9 → 66.1; Math 53.5 → 57.1; CMath 84.7 → 85.1; HumanEval 59.1 → 62.2; MBPP 72.0 → 73.9; CMMLU 82.0 → 82.9; C-Eval 79.6 → 82.5 | downstream | [AR] Table 3 p.10 |
| +7.5 GPQA-Diamond, +3.6 Math, +3.1 HumanEval, +1.1 MMLU, +1.9 TriviaQA | deltas stated in text | [AR] §5.2 p.11 |
| Kimi Linear 48B: 27 Transformer blocks (54 layers), 8 of 256 routed + 1 shared expert, 48B total / 3B active; 6 layers per block → 9 blocks + embedding = 10 sources; 1T WSD + ≈ 400B mid-training tokens; 8M-token batch | final model setup | [AR] §5.2 p.9 |
| Table 1 per-layer residual I/O (typical L = 128, N = 8, S = 16, m = 4): standard 3d; mHC 34d; Full AttnRes 24d; Block AttnRes 5.5d | memory access per token per layer | [AR] Table 1 p.8 |
| < 4% end-to-end training overhead under pipeline parallelism; negligible without PP | | [AR] §4.1 p.7 |
| < 2% inference latency overhead on typical workloads | | [AR] §1 p.2, §4.2 p.8 |
| 15 GB block cache for a 128K-token prefill with 8 blocks → ≈ 1.9 GB per device with sequence sharding → < 0.3 GB with 16K chunked prefill | | [AR] §4.2 p.8 |
| P = 4, V = 2: caching eliminates 6 redundant block transmissions in the second virtual stage; peak per-transition cost O(C) → O(P), a V× improvement | cross-stage caching | [AR] §4.1 Eq. 7-8, Fig. 3 p.6 |
| 25 architecture configs: AttnRes lower in all, by 0.019–0.063; optimum d_model/L_b 60 (baseline, 1.847) → 45 (AttnRes, 1.802) | fixed-compute sweep | [AR] §5.4.1 p.12-13 |
| Kimi K3: 93 layers; 8 blocks of 12 layers; partial final block; 9 sources with the embedding | K3 configuration | [K3] §2.2 p.6, Table 1; [K3-cfg] `attn_res_block_size` 12 |
| L < 100 for the full form's O(L²d) arithmetic to be affordable (K3); "L < 1000" (AR) | depth vs sequence | [K3] p.6; [AR] §3 p.3 |

### Derived numbers (written out)

- **K3 sources**: decoder layers 0-92, a new block starts when layer_idx % 12 == 0 → blocks start at 0, 12, …, 84
  → 8 blocks (7 × 12 + 1 × 9 layers); plus the embedding b0 → 9 sources ([K3-cfg]; matches [K3] §2.2).
- **Δloss at every size** (slide 11): Block − Baseline = −0.022, −0.020, −0.020, −0.020, −0.026 (Table 2).
- **I/O ratios** (slide 13 notes): mHC 34d / Block 5.5d ≈ 6.2×; Block 5.5d vs standard 3d = +2.5d per layer.
- **Toy α values in fig_op**: illustrative only (rows sum to 1), not from a model.

## Mechanism facts used on slides

- Standard residual unrolled: h_l = h_1 + Σ_{i<l} f_i(h_i) → depth-mixing matrix M is all-ones lower-triangular
  ([AR] §6.2 p.15); PreNorm magnitude grows O(L), diluting each layer ([AR] §1).
- AttnRes: h_l = Σ_{i=0}^{l−1} α_{i→l} v_i, v_0 = h_1, v_i = f_i(h_i); α_{i→l} = φ(w_l, k_i)/Σ_j φ(w_l, k_j),
  φ(q, k) = exp(qᵀ RMSNorm(k)), k_i = v_i; w_l ∈ R^d learned, one per layer (attention and MLP sublayers are
  separate "layers"); RMSNorm stops large-output layers from dominating; w_l initialized to zero → uniform
  weights at start ([AR] Eq. 1-4, §5).
- Block AttnRes: b_n = Σ_{j∈B_n} f_j(h_j); b_n^i partial sum; b_0 = h_1; V = [b_0..b_{n−1}] for the first layer in
  block n, [b_0..b_{n−1}, b_n^{i−1}] afterwards; final output aggregates all blocks; memory O(Ld) → O(Nd),
  compute O(L²) → O(N²); N = L is Full, N = 1 is standard residual with embedding isolated ([AR] §3.2).
- Cross-stage caching: blocks received in earlier virtual stages stay cached; each transition sends only new
  blocks ([AR] §4.1, Fig. 3). Activation checkpointing keeps per-layer activation memory equal to the standard
  architecture ([AR] §4.1; [K3] §5.2.2).
- Two-phase inference: Phase 1 batches all S pseudo-queries of a block against cached block reps (returns LSE
  stats); Phase 2 adds the intra-block partial sum per layer and merges with online softmax ([AR] Alg. 1, §4.2).
  K3: inter-block kernel on a side stream in decode; intra-block merge + RMSNorm fused into the TP all-reduce;
  sequence parallelism in prefill ([K3] §5.4.2).

## Discrepancies (Notion row vs primary source)

1. **"improves every task on Kimi Linear 48B / 1.4T tokens"**: Table 3 shows MMLU-Pro unchanged (52.2 vs 52.2);
   the paper's text says "matches or outperforms the baseline on all benchmarks" (§5.2) while its abstract says
   "improves ... across all evaluated tasks". The deck says "matches or beats on all 15".
2. **"output magnitude and gradients stay uniform with depth"**: [AR] says output magnitudes stay *bounded* with
   a periodic pattern within blocks and gradients are "substantially more uniform" (Fig. 5), not uniform.
3. **Notion source is Kimi K3, reference is arXiv 2603.15031**: the measured results all come from the AttnRes
   report (Kimi Linear 48B and the scaling runs); K3 adopts Block AttnRes (8 × 12-layer blocks) but publishes
   no AttnRes ablation of its own. The deck labels results with their model.
