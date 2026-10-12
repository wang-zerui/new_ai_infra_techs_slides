# Sources — deck 62: MTP (multi-token prediction) layer

## Sources read

| Tag | Source | Where |
|---|---|---|
| [GLM] | GLM-4.5 Team, "GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models", arXiv 2508.06471v1 (cached) — Notion reference | §2.1 (p.3: "we add an MoE layer as the MTP layer ... to support speculative decoding"); Table 1 (# MTP layers 1); §2.4 (λ schedule) |
| [V3] | DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv 2412.19437 (cached) | §2.2 Eq. 21-25 + Fig. 3 (p.10-11); §3.2 "Shared Embedding and Output Head for MTP" (p.13-14); §4.2 hyper-parameters (p.22-23); §4.5.1 Table 4 (p.26); §5.4.3 MTP evaluation (p.35) |
| [V4] | DeepSeek-AI, DeepSeek-V4 technical report, arXiv 2606.19348 (cached) | §2 intro + §2.1 "Multi-Token Prediction" (p.5); Fig. 2 (MTP modules, MTP loss); §4.2.1 (MTP depth 1, p.24-25); §4.2.2 (MTP loss weight, p.25-26) |
| [GLM5] | GLM-5 Team, GLM-5 technical report, arXiv 2602.15763 (cached) | §2.1 "Multi-token Prediction with Parameter Sharing" + Table 2 (p.5); §2.4.1 "Flexible MTP placement" (p.9); RL rollouts "Tail-latency reduction with FP8 rollouts and MTP" (p.14-15); Ascend inference "Multi-Token Prediction" bullet (p.22); Table 10 (# MTP layers) |
| [K3] | Kimi Team, Kimi K3 technical report, arXiv 2607.24653 (cached) | architecture table (Number of MTP Layers: 1); §4.1 "Draft Model Fine-Tuning" (MTP layer → EAGLE-3 draft, 7-step unroll, LK loss) |
| [SD] | Y. Leviathan, M. Kalman, Y. Matias, "Fast Inference from Transformers via Speculative Decoding", arXiv 2211.17192 (downloaded to `$SCRATCH/research/62/dl/`) | §3.1 Definition 3.1 + Eq. 1 (expected tokens per run = (1−α^{γ+1})/(1−α)); Theorem 3.8 (walltime factor) |
| [MC] | NVIDIA, "Scalable Training of Mixture-of-Experts Models with Megatron Core", arXiv 2603.07685 (cached) | Table note p.13 (DeepSeek-V3 = 671B main + 14B MTP = 685B); §7.6 Flexible asymmetric VPP + Table 10 (p.66); §7.7 MTP (p.67); §9.2 layouts (p.75) |
| [RD] | Megatron-LM `megatron/core/transformer/moe/README.md` (cached) | "What's New" v0.12 (`--mtp-num-layers 1`), v0.14 (CP support for MTP and MLA); model support list (DeepSeek-V3 incl. MTP) |
| [CODE] | Megatron-LM main, fetched 2026-10-13 into `$SCRATCH/research/62/dl/`: `megatron/core/transformer/multi_token_prediction.py`, `megatron/core/transformer/transformer_config.py` | `MultiTokenPredictionLayer` (enorm, hnorm, eh_proj, transformer layer); `process_mtp_loss` (`mtp_loss_scale = mtp_loss_scaling_factor / mtp_num_layers`; `MTPLossAutoScaler` attaches the MTP loss to the hidden states; logs `mtp_k_acceptance_rate` = argmax match on valid tokens); `roll_tensor` (labels shifted, CP-aware); config: `mtp_loss_scaling_factor` default 0.1, `mtp_use_repeated_layer`, `mtp_detach_heads`; assert "MTP supports at most one layer when enabling overlap_moe_expert_parallel_comm" |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 1 MTP layer | DeepSeek-V3 (D = 1), DeepSeek-V4 (depth 1), GLM-4.5, GLM-5, Kimi K3 | [V3] §4.2 p.22; [V4] §4.2.1; [GLM] Table 1; [GLM5] Table 10; [K3] arch. table |
| 61 layers, 37B active, 671B | DeepSeek-V3 main model | [V3] §4.2 p.22 |
| 14B of 685B → ~2% | DeepSeek-V3 MTP module weights; 14/685 = 2.0% (**derived**) | [MC] table note (671B main + 14B MTP) |
| 355B / 32B active | GLM-4.5 size incl. MTP | [GLM] abstract, Table 1 |
| d×2d projection M_k; RMSNorm on both inputs | MTP module structure | [V3] Eq. 21; [CODE] enorm/hnorm/eh_proj |
| λ = 0.3 for the first 10T tokens, 0.1 for the remaining 4.8T (14.8T total) | DeepSeek-V3 MTP loss weight | [V3] §4.2 p.23 |
| λ = 0.3 for the first 15T tokens, 0.1 afterwards (23T total) | GLM-4.5 | [GLM] §2.4; abstract (23T) |
| λ = 0.3 for most of training, 0.1 from the start of LR decay | DeepSeek-V4 Flash and Pro | [V4] §4.2.2 |
| 0.1 | MCore default `mtp_loss_scaling_factor` | [CODE] transformer_config.py |
| 85–90% | acceptance rate of the second (MTP-drafted) token, across generation topics | [V3] §5.4.3 |
| 1.8× | DeepSeek-V3 decoding TPS with MTP speculative decoding | [V3] §5.4.3 |
| 1.85–1.90 tokens per main-model run | **analytic**: Eq. 1 of [SD] with γ = 1 draft gives E = 1 + α; α = 0.85–0.90 | [SD] Eq. 1 + [V3] §5.4.3 |
| 228.7B total / 20.9B active, 540B tokens; 15.7B / 2.4B, 1.33T tokens | DeepSeek-V3 MTP ablation models | [V3] §4.5.1, Table 4 |
| Large MoE baseline → w/ MTP: Pile BPB 0.658 → 0.657; BBH 70.0 → 70.7; MMLU 67.5 → 66.6; DROP 68.5 → 70.6; TriviaQA 67.0 → 67.3; NQ 27.2 → 28.5; HumanEval 44.5 → 53.7; MBPP 61.6 → 62.2; GSM8K 72.3 → 74.0; MATH 38.6 → 39.8 | Table 4 large-MoE columns | [V3] Table 4 p.26 |
| deltas +0.7, −0.9, +2.1, +0.3, +1.3, +9.2, +0.6, +1.7, +1.2; "8 of 9 better" | **derived**: w/ MTP minus baseline, large MoE | derived from [V3] Table 4 |
| small MoE deltas +2.4, +3.3, +2.1, +0.8, −0.4, +6.1, +1.0, +6.0, +1.9 ("8 of 9", notes only) | **derived**: small-MoE columns (39.0→41.4, 50.0→53.3, 39.2→41.3, 56.9→57.7, 22.7→22.3, 20.7→26.8, 35.8→36.8, 25.4→31.4, 10.7→12.6) | derived from [V3] Table 4 |
| 2.55 vs 2.76 | accept length, DeepSeek-V3.2 vs GLM-5, 4 speculative steps, GLM's private prompt set | [GLM5] Table 2 + §2.1 text |
| +8% | 2.76 / 2.55 = 1.082 (**derived**) | derived from [GLM5] Table 2 |
| 3 shared MTP layers | GLM-5 trains with the parameters of 3 MTP layers shared | [GLM5] §2.1 |
| PP = 16, VPP = 2; rank 14 / VPP 1 = MTP, rank 15 / VPP 1 = loss | MCore DeepSeek-V3 flexible asymmetric layout | [MC] Table 10 |
| 7 steps | Kimi K3 unrolls the EAGLE-3-style draft (initialized from the MTP layer) for seven steps in training | [K3] §4.1 |
| ≤ 1 MTP layer | when MCore's EP all-to-all overlap is enabled | [CODE] transformer_config.py assert |

## Mechanism facts (no numbers) used on slides

- k-th MTP module = shared embedding Emb, shared output head OutHead, one Transformer block TRM_k, projection M_k ∈ R^{d×2d}; input h'^k_i = M_k[RMSNorm(h^{k−1}_i); RMSNorm(Emb(t_{i+k}))]; h^0 is the main model's final representation; prediction P^k_{i+k+1} = OutHead(h^k_i). Keeps the complete causal chain at each depth; differs from Gloeckle et al. (2024), whose independent heads predict D tokens in parallel ([V3] §2.2).
- Loss: L^k_MTP = cross-entropy of depth-k predictions; L_MTP = (λ/D) Σ_k L^k_MTP, added to the main loss ([V3] Eq. 24-25).
- In GLM-4.5 the MTP layer's transformer block is an MoE layer ([GLM] §2.1).
- Inference: MTP modules can be discarded (inference cost identical to the baseline in the ablation) or repurposed for speculative decoding ([V3] §2.2 "MTP in Inference", §4.5.1).
- Speculative decoding (lossless): the draft proposes; the target model verifies the draft in one parallel run and emits one extra token on rejection or after the last draft ([SD] §2-3).
- Train/inference mismatch: DeepSeek-V3 trains one MTP layer and predicts the next 2 tokens at inference; drafting further steps lowers acceptance of later tokens; GLM-5 shares parameters across 3 MTP layers in training, keeping draft memory at one layer ([GLM5] §2.1).
- Placement: DeepSeek-V3 (DualPipe) puts embedding and output head on the same PP rank so MTP shares their parameters and gradients physically ([V3] §3.2); MCore places the MTP layer in its own virtual stage ([MC] Table 10; layout string with `m`, §9.2); GLM-5 co-locates the MTP output layer with the main output layer on the final stage and puts MTP embedding + transformer on the preceding stage ([GLM5] §2.4.1).
- MCore integration: MTP loss is attached to hidden states via `MTPLossAutoScaler` so its gradient flows in the same backward; the training-time "acceptance rate" metric is the argmax match of the MTP prediction with the label ([CODE]); CP support for MTP since v0.14 ([RD]).
- GLM-5 uses MTP in RL rollouts (small-batch decode, long-tail stragglers) and on Ascend NPUs to raise computation density ([GLM5] p.14-15, p.22) — qualitative, no numbers.
- Kimi K3 fine-tunes its pre-trained MTP layer into an EAGLE-3-style draft (target frozen; LK loss on the acceptance rate) ([K3] §4.1).

## Discrepancies (Notion row vs primary sources; slides follow the sources)

1. **"GLM-5.2's improved MTP raises accepted length by 20%."** Not found in any source read. The GLM-5 report (arXiv 2602.15763, Table 2) gives accept length 2.76 for GLM-5 vs 2.55 for DeepSeek-V3.2 at 4 speculative steps on a private prompt set, i.e. +8% (derived), and it is a cross-model comparison, not an ablation. No GLM-5.2 source was available; the 20% figure is not used on slides.
2. **"Wins on ... training efficiency."** The primary sources show better quality at equal training tokens and identical inference cost ([V3] Table 4; "may improve data efficiency"), but MTP *adds* training compute and memory (an extra Transformer block, extra output-head pass; 14B extra parameters for DeepSeek-V3). The slides say "quality per token", not "efficiency".
3. **"One (MoE) MTP layer on top of the backbone."** Correct for GLM-4.5 (an MoE layer); DeepSeek's MTP module is "a Transformer block" whose block type follows the model; it shares the main model's embedding and output head and is fed the next token's embedding, so it is not just a layer stacked on the backbone.
4. **"MCore supports MTP + MLA + CP together."** Confirmed by the README (v0.14: CP support for MTP and MLA); additionally MCore limits MTP to ≤ 1 layer when EP all-to-all overlap is enabled ([CODE]).
5. **Reference.** The Notion reference (GLM-4.5) states only that an MoE MTP layer is added and gives λ; the module design, ablation and acceptance numbers come from DeepSeek-V3 (arXiv 2412.19437).
