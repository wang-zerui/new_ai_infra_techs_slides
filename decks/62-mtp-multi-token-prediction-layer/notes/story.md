# Story — deck 62: MTP (multi-token prediction) layer

- **Where it lives** (prosperity): every recent frontier MoE ships exactly one MTP layer: DeepSeek-V3
  (671B + 14B MTP), DeepSeek-V4 (same MTP), GLM-4.5 (an MoE MTP layer), GLM-5, Kimi K3. Megatron-Core:
  `--mtp-num-layers 1`, with MLA and CP.
- **Pain number** (betrayal): no published measurement isolates the cost, so the deck uses the structural count:
  autoregressive decoding emits **1 token per forward pass** through all 61 layers of DeepSeek-V3 (37B active),
  and next-token training gives each position one target.
- **Failed suitors**:
  1. A separate draft model (classic speculative decoding): a second model to train and serve, drafting from a
     different model's view of the text.
  2. Parallel independent heads (Gloeckle et al. 2024): each head predicts t+k from the same hidden state, so the
     head for t+3 never sees t+2 — no causal chain.
- **Villain**: the drafter and the trainer are different things: next-token training wastes the chance to
  supervise further ahead, and a bolt-on drafter never shares the main model's representation.
- **Turn / insight**: one extra block fed the model's own last hidden state and the next token's embedding learns
  the token after next well enough that **85–90%** of its drafts are accepted (DeepSeek-V3) — so the same block
  can densify training and serve as the drafter.
- **Mechanisms**:
  ❶ MTP module: [RMSNorm(h_i); RMSNorm(Emb(t_{i+1}))] → M (2d→d) → Transformer block → shared head → t_{i+2};
     shared embedding and output head; causal chain kept (answers: parallel heads lack the chain).
  ❷ Loss weighting: L_main + (λ/D)·Σ L_MTP, λ = 0.3 then 0.1 late in training (answers: the auxiliary objective
     must not dominate the main loss near the end).
  ❸ Speculative decoding: MTP drafts t+2, the main model verifies it in its next forward, emitting 2 tokens on
     accept and 1 on reject (answers: decode emits 1 token per pass).
  ❹ Systems: MTP is heavy (embedding + block + vocab head) → place it in its own pipeline slot, share embedding/head
     physically (answers: pipeline stage imbalance).
- **Release**: same inference cost, better quality: HumanEval **44.5 → 53.7** (+9.2) at 228.7B, 8 of 9 benchmarks
  up; decode **1.8×** TPS at 85–90% acceptance (analytic ceiling 1.85–1.90 tokens per pass); GLM-5's shared
  3-step MTP reaches accept length **2.76** vs 2.55 (DeepSeek-V3.2, 4 steps).
- **Limits**: +14B params (~2%) and an extra block + vocab-head pass per training step; pipeline balancing; trained
  1-deep but drafted deeper → acceptance falls (GLM-5 shares 3 layers, Kimi K3 fine-tunes it into an EAGLE-3 draft);
  MCore EP overlap allows ≤ 1 MTP layer; MMLU −0.9 in the ablation; acceptance depends on the text.

## Slide plan (15 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Frontier MoEs ship one extra MTP layer | prosperity | fig_stack (SVG) |
| 3 | Decoding still emits one token per pass | pain | `%% big: 1 token` |
| 4 | Bolt-on drafters and parallel heads fall short | suitors / villain | fig_suitors (SVG) |
| 5 | Insight: train the drafter inside the model | turn | boxed sentence (build) |
| 6 | One MTP depth: own block, shared embedding and head | overview | fig_overview (SVG) |
| 7 | ❶ Each depth chains h to the next token | mechanism | fig_module (SVG, worked example) |
| 8 | ❷ λ weights the MTP loss: 0.3, then 0.1 | mechanism | chart lambda |
| 9 | ❸ At decode, MTP drafts and the model verifies | mechanism | fig_specdec (SVG) |
| 10 | ❹ MTP is heavy: give it its own pipeline slot | mechanism (systems) | fig_placement (SVG) |
| 11 | +9.2 HumanEval at the same inference cost | release | chart mtp_ablation |
| 12 | 85–90% acceptance gives 1.8× decode speed | release | chart decode |
| 13 | Shared 3-step MTP drafts longer: 2.76 vs 2.55 | release | chart accept_len |
| 14 | The cost: 2% params, extra compute, draft depth | limits | two-col text |
| 15 | Summary | | QR |
