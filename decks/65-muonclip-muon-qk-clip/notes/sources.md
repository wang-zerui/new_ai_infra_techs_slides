# Sources — deck 65: MuonClip (Muon + QK-Clip)

## Sources read

| Tag | Source | Where |
|---|---|---|
| [K2] | Kimi Team, "Kimi K2: Open Agentic Intelligence", arXiv 2507.20534. https://arxiv.org/abs/2507.20534 | Abstract + §1 (p.1-2); §2.1 MuonClip, Algorithm 1, Fig. 2 (p.3-4); Fig. 3 (p.5); §2.3 + Table 2 (p.6); §2.5 (p.9); App. D (p.29-30, Fig. 12); App. E (p.30-31) |
| [ML] | Liu et al. (Moonshot AI), "Muon is Scalable for LLM Training" (Moonlight), arXiv 2502.16982. https://arxiv.org/abs/2502.16982 (downloaded to `$SCRATCH/research/65/dl/`) | Abstract and Fig. 1a (p.1-2); §1 contributions (p.2); §2.2 weight decay and update-RMS matching (p.3-4) |
| [DSV3] | DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv 2412.19437 | §2.1.1 MLA, Eq. 1-11 (p.7-8): only c_KV and the decoupled k_R are cached; k_R is produced by one W_KR shared by all heads |
| [GLM] | GLM-4.5 Team, arXiv 2508.06471 | §2.1 (p.3): Muon optimizer (§2.4) with QK-Norm "to stabilize the range of attention logits" (GQA attention) |
| [K3] | Kimi Team, "Kimi K3", arXiv 2607.24653 | §2.5 Per-Head Muon (p.10); §3.3 (p.11): K3 keeps "the weight-clipping mechanism introduced in Kimi K2" |
| [N] | Notion row (notes/notion_row.md) | properties + body links |

The Keller Jordan Muon blog listed on the Notion row was not fetched; the Muon update used on the slides
is the one written in [K2] Algorithm 1.

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 1.04T total / 32.6B active | Kimi K2 parameters | [K2] Table 2 p.6 |
| 61 layers, 64 attention heads, MLA | K2 attention | [K2] Table 2 p.6, §2.3 |
| 15.5T tokens | K2 pretraining tokens | [K2] abstract, §2.5 p.9 |
| ~52% of training FLOPs (Fig. 1a: "0.519x FLOPs"); "~2×" compute efficiency | Muon vs AdamW to equal loss, compute-optimal scaling law | [ML] abstract, §1 p.2, Fig. 1a |
| > 1000 | max attention logit with vanilla Muon, 53B-total / 9B-active MoE ("quickly exceed a magnitude of 1000") | [K2] §2.1 p.3, Fig. 2 left |
| loss spikes, occasional divergence | consequence of max logits at that level | [K2] §2.1 p.3 |
| τ = 100 | K2 QK-Clip threshold; max logit "capped at 100" | [K2] §2.1 p.4, Fig. 2 right |
| ~30% of training steps | when K2's max logits decay to a stable range | [K2] Fig. 2 caption p.4 |
| α = 0.5 | balancing exponent in the naive clip (equal scaling of Q and K) | [K2] §2.1 p.3 |
| γ_h = min(1, τ / S_max^h) | per-head scaling factor | [K2] §2.1 p.3 |
| √γ on q_C, k_C; γ on q_R; k_R untouched | MLA split | [K2] §2.1 p.3, Algorithm 1 lines 13-15 |
| 0.2 · √max(n, m) | Muon update RMS matched to AdamW | [K2] Algorithm 1 line 5; [ML] §2.2 (AdamW update RMS "around 0.2 to 0.4") |
| 0 loss spikes | K2 per-step training loss (no smoothing or sub-sampling) "shows no spikes" | [K2] abstract ("without a single loss spike"), Fig. 3 p.5 |
| 70,000 steps; 12.7% of attention heads | QK-Clip triggered at least once by 12.7% of heads in the first 70K steps; after 70K steps all heads dropped below 100 and QK-Clip was inactive | [K2] App. D p.30 |
| 0.5B active / 3B total MoE, τ = 30 | small-scale ablation: negligible effect on loss; "no statistically significant degradation" downstream | [K2] App. D p.30, Fig. 12 |
| 67M tokens | K2 global batch size | [K2] §2.5 p.9 |

## Derived numbers (labelled "derived" on the slides)

1. **K2 total steps.** 15.5T tokens / 67M tokens per step = 231,343 ≈ **231K** steps (excluding the 460B-token
   anneal, which adds ≈ 6.9K steps).
2. **70K steps in tokens.** 70,000 × 67M = 4.69T ≈ **4.7T** tokens = 70K / 231K = **30%** of the 15.5T
   pretraining steps. This matches [K2] Fig. 2's "approximately 30% of the training steps".
3. **Heads.** 61 layers × 64 heads = **3,904** attention heads; 12.7% of them ≈ **496** heads (speaker
   notes only).
4. **Logit scaling (algebra, no data).** logit_h = q_C·k_C + q_R·k_R. With q_C, k_C × √γ and q_R × γ:
   (√γ q_C)·(√γ k_C) + (γ q_R)·k_R = γ · logit_h. So for the same inputs every logit of head h is scaled by
   exactly γ, and the batch max becomes min(S_max^h, τ). Naive variant: W_q × γ^α, W_k × γ^(1−α) gives
   the same factor γ for any α.
5. **Bound (from [K2] App. E Eq. 2).** |q_i·k_j| ≤ ‖x_i‖‖x_j‖‖W_q‖‖W_k‖; RMSNorm keeps the x terms
   bounded, so growth comes from the spectral norms of W_q and W_k.

## Charts

- `charts/chart_muon.py`: relative training FLOPs to reach equal loss, AdamW = 1.00, Muon = 0.52 ([ML]).
- `charts/chart_maxlogit.py`: max attention logit, log scale: vanilla Muon (53B-A9B MoE) "> 1000" drawn as a
  bar to 1000 with an up-arrow (lower bound, not a measured peak); Kimi K2 with MuonClip: 100 (the cap τ).
  Different model scales; both from [K2] Fig. 2.
- `charts/chart_activity.py`: share of attention heads that triggered QK-Clip: 12.7% in steps 0-70K, 0%
  afterwards ([K2] App. D); step ranges converted to tokens with the 67M batch (derived 1-2).

## Mechanism facts (no numbers) used on slides

- Max logit definition: S_max^h = (1/√d) max over samples X in batch B, max over token pairs i,j of
  Q_i^h K_j^hᵀ; a per-head scalar already computed in the forward pass ([K2] §2.1, Algorithm 1 line 10).
- QK-Clip rescales W_q, W_k "post-update"; it "does not alter the forward/backward computation in the
  current step" ([K2] §2.1 p.3).
- Soft-cap critique: "the dot products between queries and keys can still grow excessively before capping
  is applied"; QK-Norm "is not applicable to MLA, because its Key matrices are not fully materialized during
  inference" ([K2] §2.1 p.3). MLA caches only c_KV and k_R ([DSV3] §2.1.1).
- Why Muon: Muon's msign update has all singular values equal (full effective rank), Adam's update a skewed,
  low-rank spectrum; full-rank updates more likely align with W's singular vectors and grow singular values
  additively; W_q W_kᵀ "squares the spectral norm" ([K2] App. E, stated as a hypothesis; Moonlight 16B
  weights trained with Muon show higher singular-value entropy).
- Per-head rather than per-layer/global, "to minimize potential over-regularization on other heads"
  ([K2] App. D).
- Muon step: M_t = μM_{t−1} + G_t; O_t = Newton-Schulz(M_t)·√max(n,m)·0.2; W_t = W_{t−1} − η(O_t + λW_{t−1})
  ([K2] Algorithm 1).

## Discrepancies (Notion row vs primary sources)

1. **"after every step the Q/K projection weights are rescaled against threshold tau".** QK-Clip runs after
   every step but rescales only heads whose max logit exceeds τ (γ_h = min(1, τ/S_max^h)); in K2 it acted on
   12.7% of heads during the first 70K steps and was inactive afterwards ([K2] §2.1, App. D). Slides say
   "only heads above τ".
2. **"killing attention-logit explosion at the source".** QK-Clip bounds the weights' contribution after the
   fact: the step in which S_max is measured runs unclipped, and the clip acts from the next step
   ([K2] §2.1). The logits are capped at τ, not prevented from rising to τ (Fig. 2 right). Slides say "caps".
3. **"no downstream degradation from clipping".** The evidence is a small-scale ablation (0.5B active / 3B
   total MoE, τ = 30: negligible loss change, no statistically significant downstream degradation) plus the
   fact that QK-Clip self-deactivates in K2; there is no full-scale K2 ablation ([K2] App. D). Slides
   attribute it accordingly.
4. **"Zero loss spikes across K2's full 15.5T-token pretraining"** matches [K2] (abstract, Fig. 3). No
   discrepancy.
