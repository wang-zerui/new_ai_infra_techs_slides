# Sources — deck 55: Loss-free balance routing + sigmoid gating

## Sources read

| Tag | Source | Where |
|---|---|---|
| [LF] | L. Wang, H. Gao, C. Zhao, X. Sun, D. Dai (DeepSeek-AI / PKU), "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts", arXiv 2408.15664v1 (28 Aug 2024). https://arxiv.org/abs/2408.15664 (downloaded to `$SCRATCH/research/55/dl/`) | Abstract; §2.2 Eq. 2 + Fig. 2 (p.3-4); §3 Eq. 3 + Algorithm 1 (p.3-4); Table 1 (p.5); §4.1 (p.5); Table 2, Fig. 3, Eq. 4 (p.6); Fig. 4, Table 3 (p.7); Table 4 (p.8); §5.1-5.2, Fig. 5 (p.8-9); App. A Table 5 (p.11); App. B (p.11); App. C, Table 6 (p.12); App. D (p.13-14) |
| [GLM] | GLM-4.5 Team, "GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models", arXiv 2508.06471v1 (cached) | §2.1 (p.2-3) + Table 1 (p.3); §2.4 Hyper-Parameters (p.4-5) |
| [V3] | DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv 2412.19437 (cached) | §2.1.2 Eq. 12-20 (p.8-9); "No Token-Dropping" (p.10); §4.2 hyper-parameters (p.22-23); §4.5.2 Table 5 (p.26-27); §4.5.3 + Fig. 9 (p.27-28) |
| [V4] | DeepSeek-AI, "DeepSeek-V4" technical report, arXiv 2606.19348 (cached) | §2.1 "Mixture-of-Experts" (p.5); §4.2.1 model configs (p.24-25); §4.2.2 training setups (p.25-26) |
| [K3] | Kimi Team, "Kimi K3" technical report, arXiv 2607.24653 (cached) | §2.3 (p.6); §2.3.3 Quantile Balancing, Eq. 13 (p.7); App. "Relation to sign-based loss-free updates" |
| [MC] | NVIDIA, "Scalable Training of Mixture-of-Experts Models with Megatron Core", arXiv 2603.07685 (cached) | §2.1.2 + Fig. 2 (p.9-10); §4.3.4 router fusion + Fig. 21 (p.37-38); §7.1 + Fig. 39 (p.63-64) |
| [RD] | Megatron-LM `megatron/core/transformer/moe/README.md` (cached copy) | "What's New" v0.14 (router fusion); "Router and Load Balancing" tables; "Router Arguments" table (defaults) |
| [CODE] | Megatron-LM main, fetched 2026-10-13 into `$SCRATCH/research/55/dl/`: `megatron/core/transformer/moe/router.py`, `moe_utils.py`, `megatron/core/distributed/finalize_model_grads.py` | `TopKRouter.__init__` (buffers `expert_bias` fp32, `local_tokens_per_expert` int64); `_apply_expert_bias` (counts only when `torch.is_grad_enabled()`: evaluation is skipped, and with activation checkpointing a micro-batch is counted once, in its grad-enabled pass); `topk_routing_with_score_function` (sigmoid branch: `scores_for_routing = scores + expert_bias`, gather unbiased `scores`, normalize over top-k); `get_updated_expert_bias` (all-reduce over TP×CP×DP, `sign(total − count·E)·rate`); `finalize_model_grads` → `_update_router_expert_bias`; `reset_model_temporary_tensors` (zero counters); fused path passes `expert_bias` to TE `fused_topk_with_score_function` |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 0.72 → "+72%" | MaxVio_global of the aux-loss baseline (α = 0.001), 1B MoE; MaxVio = (max load − mean load)/mean load, so the busiest expert carries 1.72× its fair share (derived from Eq. 4), averaged over layers | [LF] Table 2 (p.6), Eq. 4 (p.6) |
| 0.04 | MaxVio_global with Loss-Free Balancing, 1B and 3B | [LF] Table 2 |
| 9.56 → 9.50 | validation perplexity, 1B, aux loss (α = 0.001) vs Loss-Free | [LF] Table 2 |
| 7.97 → 7.92; 0.52 → 0.04 | 3B: perplexity and MaxVio_global, aux loss vs Loss-Free | [LF] Table 2 |
| 1B on 100B tokens, 3B on 200B tokens | training budgets | [LF] §1 p.2, §4.1 p.5, App. B |
| 64 routed + 2 shared experts, top-6 | 1B/3B experiment architecture | [LF] App. A Table 5 |
| α ∈ {1e-2, 1e-3, 1e-4, 0}; (MaxVio_global, perplexity) ≈ (0.10, 9.69), (0.72, 9.56), (2.23, 9.53), (5.07, 9.54) | aux-loss sweep, 1B; the 1e-3 point equals Table 2; the other three are **digitized from Fig. 2** (raster image extracted from the PDF; pixel positions mapped through the figure's own gridlines at x = 0..5 and y = 9.50..9.70; check: the two Table-2 points are recovered as (0.725, 9.557) and (0.045, 9.499)). Slides/charts mark these "≈ read off Fig. 2" | [LF] Fig. 2 (p.4); script `$SCRATCH/tools/digitize.py`, `find_a0.py` |
| u = 0.001 | bias update rate in [LF] main experiments | [LF] §4.3, Table 3 |
| u = 1e-4 / 1e-3 / 1e-2 → perplexity 9.51 / 9.50 / 9.51 | update-rate study: 1e-4 slow to converge early; 1e-2 fluctuates late | [LF] Fig. 4 legend + §4.3 (p.7) |
| sign rule 9.50 / 0.044; proportional (u·e_i, u = 0.01) 9.53 / 0.028 | update-rule variants (perplexity / MaxVio_global) | [LF] Table 3 |
| additive 9.50 vs multiplicative 9.52-9.54 | bias form | [LF] Table 4 |
| softmax gate: 9.604 / 0.937 (aux loss, α = 0.0003) vs 9.599 / 0.027 (loss-free, proportional update u = 1e-3) | loss-free with softmax | [LF] App. C.2, Table 6 |
| ~10% abnormal loss drop; > 50 bits/token | Expert Choice leakage: chunk 512 experiment; theoretical bound for 9 MoE layers, 16 experts, 2 per token | [LF] §5.2 (p.8-9), App. D |
| 671B total / 37B active; 256 routed + 1 shared expert; top-8; ≤ 4 nodes; 64 GPUs on 8 nodes per layer | DeepSeek-V3 config | [V3] §4.2 p.22-23 |
| γ = 0.001 for the first 14.3T tokens, 0.0 for the remaining 500B; α = 0.0001; 14.8T tokens | DeepSeek-V3 bias update speed and balance-loss weight | [V3] §4.2 p.23; p.1 |
| 355B / 32B active; 160 routed experts, top-8, 1 shared; 89 MoE layers | GLM-4.5 config | [GLM] abstract, Table 1 |
| bias update rate 0.001 for the first 15T tokens, 0.0 afterwards; sequence-level aux loss weight 0.0001; 23T tokens | GLM-4.5 routing hyper-parameters | [GLM] §2.4 p.5; abstract |
| DeepSeek-V4: Sqrt(Softplus) affinity; bias update speed 0.001; balance-loss weight 0.0001; Flash 284B (13B act., 256 routed, top-6), Pro 1.6T (49B act., 384 routed, top-6) | DeepSeek-V4 routing | [V4] §2.1 p.5; §4.2.1-4.2.2 p.24-26 |
| 896 routed experts per layer, 16 active | Kimi K3, where the fixed-step sign rule is replaced by Quantile Balancing | [K3] §2.3 p.6, §2.3.3 p.7 |
| 228.7B total / 20.9B active, 578B tokens; 15.7B / 2.4B, 1.33T tokens | DeepSeek-V3 ablation models | [V3] §4.5.2, Table 5 |
| Large MoE, aux-loss-based → aux-loss-free: Pile BPB 0.656 → 0.652; BBH 66.7 → 67.9; MMLU 68.3 → 67.2; DROP 67.1 → 67.1; TriviaQA 66.7 → 67.7; NQ 27.1 → 28.1; HumanEval 40.2 → 46.3; MBPP 59.2 → 61.2; GSM8K 70.7 → 74.5; MATH 37.2 → 39.6 | Table 5 large-MoE columns | [V3] Table 5 p.27 |
| deltas +1.2, −1.1, 0.0, +1.0, +1.0, +6.1, +2.0, +3.8, +2.4; "7 of 9 benchmarks better, 1 tie, 1 worse" | **derived**: loss-free minus aux-loss-based, large MoE | derived from [V3] Table 5 |
| small MoE (15.7B) deltas: BBH +2.0, MMLU +0.8, DROP +0.9, TriviaQA +0.2, NQ +0.2, HumanEval +0.6, MBPP −0.8, GSM8K +2.5, MATH +0.2 → "8 of 9 better" (speaker notes only) | **derived** from Table 5 small-MoE columns (37.3→39.3, 51.0→51.8, 38.1→39.0, 58.3→58.5, 23.2→23.4, 22.0→22.6, 36.6→35.8, 27.1→29.6, 10.9→11.1) | derived from [V3] Table 5 |
| "MaxVio ≈ 5 → about six times its fair share" (speaker notes) | **derived**: max load = (1 + MaxVio) × mean, with the digitized α = 0 point | [LF] Fig. 2 (digitized), Eq. 4 |
| 2.258 / 2.253 / 2.253 (1B); 2.085 / 2.080 / 2.080 (3B) | validation loss: sequence-wise aux loss / aux-loss-free / batch-wise aux loss | [V3] §4.5.3 p.27 |
| 1e-3 default; flags | MCore `--moe-router-bias-update-rate` default 1e-3; `--moe-router-enable-expert-bias` default False; `--moe-router-score-function` default softmax | [RD] Router Arguments table |

## Mechanism facts (no numbers) used on slides

- Selection vs weighting: top-K is taken over s_{i,t} + b_i, but the gate g_{i,t} that multiplies the expert output uses the unbiased s_{i,t} ([LF] Eq. 3 and the note under it, p.4; [V3] Eq. 16 "the bias term is only used for routing"; [K3] Eq. 13 "b is omitted from p_{i,j}").
- Update rule: after each training batch, count tokens c_i per expert, e_i = mean(c) − c_i, b_i ← b_i + u·sign(e_i) ([LF] Algorithm 1, p.4). [V3] p.9: "keep monitoring the expert load on the whole batch of each training step... decrease the bias term by γ if overloaded, increase it by γ if underloaded".
- Causality: the bias uses the load of previous batches; using the current sequence's load would leak future tokens ([LF] §3 p.4).
- Expert Choice balances perfectly but conditions each token's experts on future tokens in the same chunk → leakage ([LF] Table 1, §5.2).
- Aux loss L_Bal = α Σ f_i P_i adds gradients that conflict with the LM objective ([LF] §1, §2.2 Eq. 2).
- Sigmoid vs softmax: sigmoid baseline better than softmax baseline; with softmax the normalization makes the score gap between two experts depend on the other experts' scores, which makes the bias harder to adjust → proportional update used ([LF] §4.1, App. C.1-C.2).
- DeepSeek-V3: s = Sigmoid(u·e); gate = selected s normalized to sum 1 ([V3] Eq. 13-15). MCore sigmoid branch does the same (`probs = scores / scores.sum()` over top-k) ([CODE]).
- EP compatibility: loss-free balance keeps improving with computation-batch size; aux-loss balance plateaus ([LF] §5.1, Fig. 5).
- Batch-wise vs sequence-wise scope: batch-wise balance lets experts specialize by domain (Fig. 9); two efficiency risks: imbalance within single sequences/small batches, and domain shift at inference (handled by large EP/DP micro-batches and redundant experts) ([V3] §4.5.3).
- MCore implementation ([CODE]): `expert_bias` is a registered fp32 buffer (not a parameter, so no gradient/optimizer state); `local_tokens_per_expert` int64 buffer incremented in the router forward only when `torch.is_grad_enabled()` (no counting in eval; no double counting when activations are recomputed, per the `_apply_expert_bias` docstring and the z-loss comment "Skip ... when using torch.no_grad() or checkpointing"); once per global batch, `finalize_model_grads` calls `_update_router_expert_bias`, which all-reduces the stacked counts over the TP×CP×DP group and applies `b += sign(mean − count)·rate`; counters are zeroed in `reset_model_temporary_tensors`. Fused router (`--moe-router-fusion`, MCore v0.14) passes `expert_bias` into TE's `fused_topk_with_score_function` ([CODE]; [RD]; [MC] §4.3.4).
- [MC] §7.1 Fig. 39 classifies: aux loss = differentiable soft balance; Sinkhorn = non-differentiable hard balance; expert bias = non-differentiable adaptive (feedback) balance.

## Discrepancies (Notion row vs primary sources; slides follow the sources)

1. **"A learnable per-expert bias."** The bias is not learned by back-propagation: [LF] Algorithm 1 updates it with a fixed ±u sign step from token counts, and MCore stores it as a non-gradient buffer ([CODE]). ([MC] §7.1 also uses the word "learnable"; the slides say "feedback-updated".)
2. **"Instead of an auxiliary loss."** Every production recipe found keeps a tiny complementary sequence-wise balance loss (weight 0.0001: [V3] p.23, [GLM] §2.4, [V4] p.25-26). The original paper uses no aux loss at all.
3. **"Scoring uses sigmoid."** Sigmoid is the scoring function in [LF]'s main runs, [V3] and GLM-4.5, but the method also works with softmax (with a proportional update, [LF] App. C) and DeepSeek-V4 replaced Sigmoid with Sqrt(Softplus) ([V4] §2.1).
4. **"Fixed update rate."** Correct for the rule, but both DeepSeek-V3 (last 500B tokens) and GLM-4.5 (after 15T tokens) set the rate to 0.0 late in training, i.e. the bias is frozen ([V3] p.23; [GLM] §2.4).
5. **"Now the default routing setup."** True for the open frontier models checked (DeepSeek-V3/V4, GLM-4.5; Kimi K3 keeps bias routing but replaces the update rule), but in Megatron-Core it is opt-in: `--moe-router-enable-expert-bias` defaults to False and `--moe-router-score-function` defaults to softmax ([RD]).
6. **Reference.** The Notion reference (GLM-4.5, arXiv 2508.06471) only adopts the technique (one sentence in §2.1 and its hyper-parameters in §2.4); the method and all controlled experiments come from DeepSeek's arXiv 2408.15664 and DeepSeek-V3 §4.5.2. The deck cites both.
7. **Source "DeepSeek V3.2".** DeepSeek-V3.2 was not read; the deck uses DeepSeek-V3 (origin of the production recipe) and DeepSeek-V4.
