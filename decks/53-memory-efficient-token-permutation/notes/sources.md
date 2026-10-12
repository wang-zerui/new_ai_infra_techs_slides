# Sources — deck 53: Memory-efficient token permutation

## Sources read

| Tag | Source | Where |
|---|---|---|
| [MC] | NVIDIA, "Scalable Training of Mixture-of-Experts Models with Megatron Core", arXiv 2603.07685v2. https://arxiv.org/abs/2603.07685 (cached text) | §2.1 + Fig. 1 (p.9), Table 3 + §4.1.1 (p.20), §4.1.2 Eqs. 1-2 (p.21), §4.1.3 (p.22), Table 4 (p.24), §4.1.6 (p.26-27), Table 6 (p.29), §4.3.3 (p.37), §9.2 (p.73), Table 18 (p.74) |
| [RD] | Megatron-LM `megatron/core/transformer/moe/README.md` (cached; Notion's reference) | What's New v0.13 ("Memory efficient token permutation by moving the probs multiplication from unpermutation to activation function of GroupedMLP"); feature list |
| [UT] | `megatron/core/transformer/moe/moe_utils.py` (main, fetched 2026-10-13 via `gh api`) | `permute(..., probs=...)` returns permuted_probs; `unpermute(..., probs=None)`; comment on FP32 promotion in the unfused weighted path |
| [TD] | `megatron/core/transformer/moe/token_dispatcher.py` (main) | `MoEAlltoAllTokenDispatcher.dispatch_preprocess` (permute with probs), `token_dispatch` (token all-to-all, then a separate probs all-to-all), `combine_postprocess` (unpermute without probs); AllGather and Flex dispatchers likewise return permuted probs |
| [EXP] | `megatron/core/transformer/moe/experts.py` (main) | `TEGroupedMLP.forward` `bias_act_func` multiplies `permuted_probs` inside the activation (`weighted_bias_swiglu_impl` when fused); `_apply_bias` scales the fc2 bias by probs; `moe_apply_probs_on_input` (top-1 only); `SequentialMLP` passes probs as `per_token_scale` |
| [SW] | `megatron/core/fusions/fused_bias_swiglu.py` (main) | `WeightedSwiGLUFunction` (saves input and weights), `weighted_swiglu_back` (input_grad = swiglu_back(g·w, y); weights_grad = Σ swiglu(y)·g) |
| [DSV3] | DeepSeek-V3 Technical Report, arXiv 2412.19437 (cached) | §4.2 (hidden 7168, 256 routed experts, expert intermediate 2048, top-8) p.22 |
| [K2] | Kimi K2 report, arXiv 2507.20534 (cached) | §2.4.3 "Selective recomputation" (MoE down-projections recomputed), p.8 |
| [K3] | Kimi K3 report, arXiv 2607.24653 (cached) | §5.2.2 "Unified activation manager" (prob-gradient rewrite after SonicMoE; dispatch recompute), p.20 |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 256 routed experts, top-8 | DeepSeek-V3 routing | [MC] §9.2 p.73; [DSV3] §4.2 |
| ~26.3 GB | activation memory per GPU saved by memory-efficient permutation, DeepSeek-V3, Table 3 config | [MC] §4.1.2 p.21 ("approximately 26.3 GB") |
| 131 GB / 199.5 GB | activations / total per GPU, DeepSeek-V3 BF16, PP4 × VPP4 × EP64, 256 GPUs | [MC] Table 3 p.20 |
| 20% (derived) | 26.3 / 131.0 = 0.201 | derived |
| 173.2 GB (derived) | 199.5 − 26.3, right bar of mem_compare | derived |
| 104.7 GB (derived, chart split) | 131.0 − 26.3 "other activations" | derived |
| 68.5 GB (chart) | 36.4 + 32.1 weights/grads + main weights/optimizer | [MC] Table 3 |
| 80 GB | H100 capacity | [MC] p.20 |
| 7,168 / 4,096 / 2,048 / 7,168 values; 35% (analytic) | per routed copy: x (hidden 7168), z (gated fc1 output 2×2048), a (2048), E (7168); E share 7168/20480 = 0.35 | [DSV3] §4.2 dims; saved-tensor roles from [EXP], [SW], [UT]; derived |
| toolbox chart: 42.4 / ~26.3 / 15–20 (0.1–0.2 s/iter) / ~16 / 10–12 GB | per-technique savings on DeepSeek-V3 vs Table 3 | [MC] Table 4 p.24; §4.1.2 p.21; §4.1.6 p.26-27; §4.1.3 p.22; §4.1.6 p.26 |
| "no trade-off" | Table 6 lists "-" as the trade-off for Memory-Efficient Permutation | [MC] Table 6 p.29 |
| v0.13 (2025/07) | release that added the technique | [RD] |
| both final DeepSeek-V3 configs | Table 18 lists memory-efficient permutation for GB200 and H100 | [MC] Table 18 p.74 |
| FP32 accumulation | unpermute sums in FP32 | [MC] §4.3.3 p.37 |

## Mechanism facts (no numbers) used on slides

- Equivalence: y = Σ p_i W2 φ(W1 x) (Eq. 1) = Σ W2 (p_i φ(W1 x)) (Eq. 2) when W2 has no bias; in Eq. 1 ∂L/∂p_i needs
  E_i(x) retained; in Eq. 2 it needs φ(z_i), recomputed from z_i by a fused backward kernel; z_i is saved for
  SwiGLU's own backward anyway ([MC] §4.1.2 p.21).
- Permute also permutes probs; "results directly entering the activation part of the expert"; unpermute "simply
  adds them up" ([MC] §4.3.3 p.37; [UT]; [TD]).
- All-to-all dispatcher sends tokens, then probs with a second `all_to_all` using the same splits ([TD]
  `token_dispatch`).
- Weighted SwiGLU forward saves (input, weights); backward computes input_grad = swiglu_back(g·w, y) and
  weights_grad = sum(swiglu(y)·g, dim=-1) ([SW]).
- Bias: `_apply_bias` adds the fc2 bias multiplied by the per-token probs ([EXP]); the paper states the
  equivalence for bias-free experts.
- Standard unfused path: `permuted_tokens * permuted_probs` may promote tokens to FP32 when probs are FP32
  (code comment in [UT] `unpermute`) — mentioned in notes only.
- Kimi K2 recomputes MoE down-projections ([K2] p.8); Kimi K3 rewrites the permuted-probs gradient to depend only
  on act_output and doutput at the cost of an extra lightweight elementwise op, and recomputes dispatch to recover
  the grouped-GEMM input ([K3] §5.2.2 p.20).

## Discrepancies

1. **"Removes one full activation copy."** The source removes the expert output kept for the router-weight
   gradient: one hidden-size row per routed token copy per MoE layer, about 26.3 GB per GPU on DeepSeek-V3
   (≈ 20% of activations, derived) — not a copy of all activations. Slides use the measured number.
2. **"Into the GroupedMLP activation function."** Current code applies the probs inside the activation of every
   expert backend (TEGroupedMLP, and SequentialMLP via `per_token_scale`), not only GroupedMLP ([EXP]). The
   README wording (v0.13) matches Notion.
3. **"Avoiding an extra large buffer allocation."** The paper frames the saving as tensors no longer *retained*
   for backward (peak activation memory), not an allocation; slides use the paper's framing.
4. **"Low-cost" vs "zero overhead".** The paper and Table 6 list no trade-off; the code shows one small cost
   the paper does not mention: probs need their own all-to-all (one scalar per routed row). Slides mention it
   as a limit, with the "negligible" judgment marked analytic.
5. **Reference.** Notion's reference (the MoE README) has no numbers; all numbers come from arXiv 2603.07685,
   so the summary QR points to the paper.
