# Story — deck 55: Loss-free balance routing + sigmoid gating

- **Where it lives** (prosperity): fine-grained MoE pretraining at frontier scale. DeepSeek-V3 (671B, 256
  routed experts, top-8, 14.8T tokens), GLM-4.5 (355B, 160 experts, top-8, 23T tokens), DeepSeek-V4
  (284B / 1.6T); Megatron-Core ships it as `--moe-router-enable-expert-bias` + `--moe-router-score-function sigmoid`.
- **Pain number** (betrayal): with the tuned auxiliary loss (α = 0.001), the busiest expert still receives
  **+72%** over its fair share of tokens (MaxVio_global 0.72, 1B MoE); under expert parallelism the
  step waits for that expert.
- **Failed suitors**:
  1. Auxiliary balance loss: its gradient fights the LM loss; every α loses either balance (α = 0: MaxVio ≈ 5)
     or perplexity (α = 0.01: ≈ 9.69 vs 9.56).
  2. Expert Choice: perfect balance, but each token's experts depend on future tokens in the chunk
     (≈ 10% abnormal loss drop at 512-token chunks) — leakage.
- **Villain**: balance is enforced *through what the model learns from*: its gradient (aux loss) or its future
  (expert choice).
- **Turn / insight**: the router score has two jobs, choosing experts and weighting their outputs; balance only
  needs the choice, and top-K is a non-differentiable selection, so a bias can steer the choice without ever
  entering the loss or the output.
- **Mechanisms**:
  ❶ Biased selection, unbiased weights: top-K(s + b), gate from s (answers: steer load without touching the
     gradient or the layer output).
  ❷ Sign feedback: after each step, b_i ← b_i + u·sign(mean load − c_i), from past batches only (answers:
     no gradient to learn b from; must stay causal).
  ❸ Sigmoid scores: each expert's score is independent of the others, so a fixed bias shift has a predictable
     effect (answers: softmax couples the score gaps; needs a proportional rule).
  Plus the production recipe (u = 0.001, 1e-4 sequence loss, bias frozen at the end) and the MCore
  implementation (fp32 buffer, int64 counters, one all-reduce per global batch).
- **Release**: MaxVio_global 0.72 → **0.04** with *lower* perplexity (9.56 → 9.50; 3B: 7.97 → 7.92); at
  228.7B scale, better on 7 of 9 benchmarks (HumanEval +6.1, GSM8K +3.8).
- **Limits**: balance is per batch, not per sequence (production keeps a 1e-4 sequence loss); part of the
  quality gain is scope (a batch-wise aux loss matches it); fixed step u trades slow adaptation for
  oscillation and struggles at ~10³ experts (Kimi K3 → Quantile Balancing, deck 56); softmax needs a
  different update; MCore default is off.

## Slide plan (16 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Every MoE step waits for its busiest expert | prosperity | fig_hook (SVG) |
| 3 | A tuned aux loss still leaves 72% overload | pain | `%% big: +72%` |
| 4 | Every aux-loss weight loses balance or perplexity | suitor 1 | chart dilemma |
| 5 | Both fixes lean on what the model learns from | villain (+ suitor 2) | fig_suitors (SVG) |
| 6 | Insight: top-K selection needs no gradient | turn | boxed sentence (build) |
| 7 | Bias the pick, keep the weights, close the loop | overview | fig_overview (SVG) |
| 8 | ❶ The bias moves the pick, never the weight | mechanism | fig_pick (SVG, worked example) |
| 9 | ❷ A sign step nudges each bias by ±u | mechanism | fig_update (SVG) |
| 10 | ❸ Sigmoid keeps each expert's score independent | mechanism | fig_gates (SVG) |
| 11 | Production: u = 0.001 plus a 1e-4 sequence loss | recipe | fig_recipe (SVG table) |
| 12 | Megatron-Core updates the bias once per global batch | implementation | fig_mcore (SVG) |
| 13 | MaxVio 0.72 → 0.04 at lower perplexity | release | chart dilemma + star |
| 14 | At 228.7B: better on 7 of 9 benchmarks | release | chart v3_ablation |
| 15 | Limits: batch scope, one step size, ~1k experts | limits | two-col text |
| 16 | Summary | | QR |
