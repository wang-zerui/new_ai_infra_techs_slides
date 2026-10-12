# Sources — deck 66: QK-Norm

## Sources read

| Tag | Source | Where |
|---|---|---|
| [GLM45] | GLM-4.5 Team, "GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models", arXiv 2508.06471v1. https://arxiv.org/abs/2508.06471 | Abstract (p.1); §2.1 + Table 1 (p.2-3: "We also incorporate QK-Norm [15] to stabilize the range of attention logits"; QK-Norm row: GLM-4.5 yes, GLM-4.5-Air / DeepSeek-V3 / Kimi K2 no); §2.4 (p.5: Muon for all parameters except word embedding, bias and RMSNorm weights; N = 5, µ = 0.95, update RMS 0.2; "can accelerate convergence and tolerate larger batch sizes"; batch 16M → 64M tokens over the first 500B tokens); ref. [15] = Henry et al. 2020 |
| [HF-CODE] | Hugging Face transformers `src/transformers/models/glm4_moe/modeling_glm4_moe.py` (main, fetched 2026-10-13) | `Glm4MoeAttention.__init__` (q_norm, k_norm = Glm4MoeRMSNorm(head_dim) when use_qk_norm); `forward` (q/k/v proj with bias → q_norm/k_norm → transpose → partial RoPE → attention with scaling head_dim^-0.5); `Glm4MoeRMSNorm` (x / sqrt(mean(x²) + eps) × weight) |
| [CFG-GLM45] | `config.json` of `zai-org/GLM-4.5` | use_qk_norm true, head_dim 128, num_attention_heads 96, num_key_value_heads 8, rms_norm_eps 1e-5, num_hidden_layers 92, partial_rotary_factor 0.5, attention_bias true |
| [QKN] | Henry, Dachapally, Pawar, Chen, "Query-Key Normalization for Transformers", Findings of EMNLP 2020, arXiv 2010.04245 | Abstract (p.1); §2 eq. (2) (p.2): ℓ2-normalize each q and k along the head dimension; §3 (p.2): softmax shift-invariance example; eq. (3)-(5) (p.3): learnable scalar g replaces 1/√d, g₀ = log2(L² − L) |
| [VIT22B] | Dehghani et al., "Scaling Vision Transformers to 22 Billion Parameters", ICML 2023, arXiv 2302.05442 | §2 "QK Normalization" (p.2): divergence after a few thousand steps at ~8B parameters, caused by extremely large attention logits → almost one-hot attention with near-zero entropy; LayerNorm on queries and keys; Fig. 1; App. B (p.23): without normalization logits grow to over 50,000 at Adam LR 1e-3; ViT LR was reduced from 1e-3 to 4e-4 for ViT-H; with QK norm LR 1e-3 stays stable |
| [PROXY] | Wortsman et al., "Small-scale proxies for large-scale Transformer training instabilities", arXiv 2309.14322 (ICLR 2024) | Fig. 1 (p.1): qk-layernorm stable across 3 orders of magnitude of LR; §3.1.1 (p.4): logit z_ij = ⟨q_i, k_j⟩/√d_h; entropy collapse; without qk-layernorm the divergence LR shrinks with model size; qk-layernorm trains a 1.2B model at LR 0.3; logit growth comes from growth of q and k norms, not alignment (App. E.1); hypothesis: quadratic dependence of logits on parameter norms |
| [CHAM] | Chameleon Team, "Chameleon: Mixed-Modal Early-Fusion Foundation Models", arXiv 2405.09818 | §2.3 (p.6-7): QK-Norm "directly controls the norm growth of input to the softmax"; Chameleon-7B without QK-Norm diverges after ~20% of a training epoch; 34B also needed norm re-ordering; QK-Norm does not solve logit shift in the final softmax, so z-loss is added |
| [OLMO2] | Team OLMo, "2 OLMo 2 Furious", arXiv 2501.00656 | §3 bullet list (p.5): QK-norm with RMSNorm "avoids attention logits being too large, which can lead to training loss divergence"; §3.3.2 + Fig. 7 (p.15-16): reordered norm and QK-norm, "in isolation, neither … yield good results"; together they reduce the gradient spike score from 0.108 to 0.069 |
| [GEMMA3] | Gemma Team, "Gemma 3 Technical Report", arXiv 2503.19786 | §2 (p.2): "we replace the soft-capping of Gemma 2 with QK-norm" |
| [K2] | Kimi Team, "Kimi K2: Open Agentic Intelligence", arXiv 2507.20534 | §2.1 (p.3): logit explosion "occurs more frequently with Muon"; logit soft-cap insufficient (q·k can grow before capping); QK-Norm "not applicable to multi-head latent attention (MLA), because its Key matrices are not fully materialized during inference"; QK-Clip definition; mid-scale 9B-activated / 53B-total MoE with vanilla Muon: max logits "quickly exceed a magnitude of 1000", causing loss spikes and occasional divergence; Fig. 2 (p.4); τ = 100 for K2; App. D (p.29-30): 12.7% of heads triggered QK-Clip in the first 70,000 steps; App. E (p.30): |q_i·k_j| ≤ ‖x_i‖‖x_j‖‖W_q‖‖W_k‖, RMSNorm bounds ‖x‖, so explosion is driven by spectral norms of W_q/W_k; Muon updates have full effective rank |
| [DSV3] | DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv 2412.19437 | §2.1.1 (p.7-8): MLA, keys k^C = W^UK c^KV, only c^KV (512) and k^R (64) are cached; §4.2 (p.22): AdamW; extra RMSNorm after the compressed latents |
| [DSV4] | DeepSeek-AI, "DeepSeek-V4", arXiv 2606.19348 | §2.3.3 (p.12): RMSNorm on each query head and on the single head of the compressed KV entries before core attention, "avoids exploding attention logits"; Muon section (p.14): this "effectively prevents attention logits from exploding. Consequently, we do not employ the QK-Clip technique" |
| [GLM5] | GLM-5 Team, "GLM-5: from Vibe Coding to Agentic Engineering", arXiv 2602.15763 | §2.1 (p.4-5): switches to MLA; with Muon Split "the scale of attention logits of GLM-5 remains stable during pre-training without any clipping strategy"; Appendix (p.36): follows GLM-4.5's Muon setting |
| [CFG-GLM5] / [HF-GLM5] | `config.json` of `zai-org/GLM-5`; transformers `glm_moe_dsa/modeling_glm_moe_dsa.py` | no `use_qk_norm` key; main attention normalizes only the low-rank latents (q_a_layernorm, kv_a_layernorm); the only k_norm is in the DSA indexer |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 355B total / 32B active, 23T tokens | GLM-4.5 | [GLM45] Abstract, Table 1 |
| Muon on all matrices except embeddings, biases, RMSNorm weights | GLM-4.5 optimizer scope (so the QK-Norm gains are not Muon-updated) | [GLM45] §2.4 p.5 |
| 16M → 64M tokens | GLM-4.5 batch-size warmup over the first 500B tokens | [GLM45] §2.4 p.5 |
| > 1,000 | max attention logit in a 9B-active / 53B-total MoE trained with vanilla Muon (mid-scale run) | [K2] §2.1 p.3, Fig. 2 (left) |
| > 50,000 | attention logits of an 8B ViT without QK normalization (Adam, LR 1e-3), loss diverging within ~2,000 steps | [VIT22B] App. B p.23 |
| ~2,000 steps | the 8B ViT's loss started increasing within 2,000 steps without QK normalization | [VIT22B] App. B p.23 |
| LR 1e-3 → 4e-4 | original ViT reduced its LR for ViT-H to avoid instability; with QK norm 1e-3 stays stable | [VIT22B] App. B p.23 |
| τ = 100 | K2's QK-Clip threshold (cap on per-head max logit) | [K2] §2.1, Fig. 2 (right) |
| 12.7% | share of K2 heads that triggered QK-Clip at least once in the first 70,000 steps | [K2] App. D p.30 |
| d = 128 | GLM-4.5 head dim (QK-Norm normalizes over these 128 dims) | [CFG-GLM45]; [GLM45] Table 1 |
| √128 = 11.3 | **analytic** bound on |logit| at unit gains. Derivation: RMSNorm gives u = q / sqrt(mean(q²) + ε), so ‖u‖² = d·mean(q²)/(mean(q²)+ε) ≤ d; q̂ = g_q ⊙ u has ‖q̂‖ ≤ max|g_q|·√d (same for k); RoPE applied after the norm is a rotation and preserves norms; Cauchy-Schwarz: |q̂·k̂|/√d ≤ ‖q̂‖‖k̂‖/√d ≤ √d · max|g_q| · max|g_k|; with d = 128 and unit gains (initialisation, weight = ones) the bound is √128 = 11.31 | [HF-CODE] (RMSNorm, order norm → RoPE, scaling 128^-0.5) |
| e^22.6 ≈ 6.7 × 10⁹ | **analytic**: largest possible ratio of two attention weights in one row at unit gains, exp(2 × 11.31); bounded is not flat | derived from the bound above |
| 256 gains per layer; 23,552 over 92 layers | **derived**: q_norm + k_norm weights, 2 × 128 per layer, × 92 layers | [HF-CODE]; [CFG-GLM45] |
| 1.2B at LR 0.3 | largest model shown training stably at LR 0.3 with qk-layernorm | [PROXY] §3.1.1 p.4 |
| 3 orders of magnitude | LR range over which qk-layernorm gives stable training | [PROXY] Fig. 1 p.1 |
| ~20% of an epoch | Chameleon-7B without QK-Norm diverges after approximately 20% of a training epoch | [CHAM] §2.3 p.7 |
| 0.108 → 0.069 | OLMo 2 gradient spike score with reordered norm + QK-norm applied together | [OLMO2] §3.3.2, Fig. 7 p.16 |
| +0.928 BLEU | QKNorm average gain on 5 low-resource translation pairs | [QKN] Abstract |
| g₀ = log2(L² − L) | QKNorm's learnable-scale initialisation | [QKN] eq. (3) p.3 |
| 512 + 64 | MLA cache per token per layer (latent c^KV + shared RoPE key k^R) | [DSV3] §2.1.1 p.8, §4.2 |
| 92 layers, 96 Q / 8 KV heads | GLM-4.5 shape (context) | [GLM45] Table 1 |

## Mechanism facts (no numbers) used on slides

- Attention logit z_ij = ⟨q_i, k_j⟩/√d_h; logit growth collapses attention to one-hot ("attention entropy collapse") ([PROXY] §3.1.1; [VIT22B] §2).
- Growth comes from the norms of q and k, not from their alignment ([PROXY] §3.1.1, App. E.1); |q_i·k_j| ≤ ‖x_i‖‖x_j‖‖W_q‖‖W_k‖ and pre-norm RMSNorm bounds ‖x‖, so the driver is the spectral norm of W_q / W_k; Muon's update has all singular values equal (full effective rank), Adam's is low-rank, which K2 hypothesises makes Muon more prone ([K2] App. E).
- Scale invariance: RMSNorm(c·q) = RMSNorm(q) for c > 0 up to ε (follows from the RMSNorm formula in [HF-CODE]); so normalizing q and k after the projection removes ‖W_q‖ and ‖W_k‖ from the logit.
- Where it sits in GLM-4.5 ([HF-CODE]): pre-attention RMSNorm on x → q/k/v projections with bias → per-head RMSNorm on q and on k (learnable 128-dim weight, eps 1e-5) → partial RoPE on the first 64 dims → keys/values to the KV cache → scaled dot-product attention (scale 128^-0.5) → o_proj. V is not normalized.
- Gain sharing: q_norm and k_norm are `Glm4MoeRMSNorm(head_dim)` applied to tensors shaped (…, heads, 128), so the
  normalization statistics are per head and per token while the 128 gains are shared by all heads of a layer
  ([HF-CODE]).
- Soft-capping (Gemma 2) caps logits but "dot products between queries and keys can still grow excessively before capping" ([K2] §2.1); Gemma 3 replaced soft-capping with QK-norm ([GEMMA3] §2).
- QK-Clip rescales W_q and W_k after the optimizer step when a head's max logit S_max exceeds τ; it uses S_max from the forward pass as a signal ([K2] §2.1). Deck 65.
- MLA: per-head keys are k^C_h = W^UK_h c^KV; only c^KV and k^R are cached ([DSV3] §2.1.1), and at inference the keys are "not fully materialized" ([K2] §2.1), so a per-head RMSNorm on k has no place; the up-projection is folded into the query side (the folding is the standard MLA decode identity q_hᵀ(W^UK_h c) = (W^UK_hᵀ q_h)ᵀ c, written out from [DSV3] eq. for k^C).
- QK-Norm does not cover the output softmax (logit drift): Chameleon and OLMo 2 add z-loss ([CHAM] §2.3; [OLMO2] §3.3.3).

## Discrepancies

1. **Notion "Impact: together with Muon, supports stable convergence at larger batch sizes."** [GLM45] §2.4 credits *Muon* with "accelerate convergence and tolerate larger batch sizes"; it credits QK-Norm only with stabilizing "the range of attention logits" (§2.1). GLM-4.5 publishes no measurement isolating QK-Norm (no logit curves, no ablation). Slides attribute the batch-size claim to Muon and present QK-Norm's effect with the analytic bound plus measurements from other labs.
2. **Notion "keeps logits in a stable band".** The bound is soft: it scales with the learned gains (max|g_q|·max|g_k|), so it is "bounded by construction at a level set by 2 × 128 learnable gains per layer", not a fixed band. Slides say "√d × gains".
3. **Implementation vs cited paper.** [GLM45] cites Henry et al. 2020 ([QKN]), which uses ℓ2 normalization plus one learnable scalar instead of 1/√d. GLM-4.5's code uses per-head RMSNorm with a 128-dim learnable weight and keeps the 1/√d scale ([HF-CODE]), i.e. the Dehghani/OLMo-2 form. Slides describe the code.
4. **"The same problem QK-Clip attacks from the other side"** (Notion) is accurate, with one limit the row omits: QK-Norm needs materialized per-head keys, which MLA does not have ([K2] §2.1). That is why K2 (MLA) uses QK-Clip and GLM-5 (MLA) dropped QK-Norm.
