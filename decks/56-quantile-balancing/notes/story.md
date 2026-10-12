# Story — deck 56: Quantile Balancing (Kimi K3)

- **Where it lives** (prosperity): Kimi K3, a 2.8T-parameter MoE (104B active) whose Stable LatentMoE layers
  route every token to 16 of 896 experts (sparsity 56). Expert pools keep growing: DeepSeek-V3 256 routed
  experts, Kimi K2 384, Kimi K3 896. Routing is auxiliary-loss-free: a per-expert bias b is added to the
  sigmoid router score for Top-k selection only.
- **Pain number** (betrayal): the fixed-step sign update of loss-free balancing degrades as the expert pool
  grows: averaged over training batches, the busiest expert carries **+34%** above its fair share at 64 experts
  (AvgMaxVio 0.34), up from 0.13 at 16 experts (2.6× worse); the worst batch reaches MaxVio 2.71
  (third-party measurement, BIP paper Tables 2-3). Expert parallelism waits for the busiest expert.
  K3 needs 896 experts, 14× the 64 that loss-free balancing was tuned and validated on.
- **Failed suitors**:
  1. Auxiliary balance loss: the coefficient trades balance against interference gradients (LFB Fig. 2).
  2. Sign-update bias (DeepSeek loss-free): moves every bias by the same ±γ using only the sign of the load
     error; small γ adapts slowly, large γ oscillates (LFB Fig. 4; K3 §2.3.3); γ even needs a schedule
     (DeepSeek-V3 set it to 0 for the last 500B tokens).
  3. Assignment solver with inequality constraints (BIP): balanced and flat with expert count, but its
     clipping can only push down over-loaded experts, never pull up idle ones, so it equilibrates slowly
     (K3 App. C). The last corpse holds the ingredient: treat balancing as an assignment problem.
- **Villain**: the sign update throws away the size of each expert's load error and walks a dual variable with
  one global step size, so its speed and stability depend on γ, not on the routing statistics.
- **Turn / insight** (provable fact, K3 App. C): balanced Top-k routing is a bipartite assignment LP; the expert
  bias is its dual variable and the exact coordinate minimizer is a closed form: the (1−k/n)-quantile of the
  expert's routing margins. The sign update is one SignSGD step on the same dual; QB jumps to the minimizer.
- **Mechanisms**:
  ❶ Top-(k+1) routing yields each token's cutoff α_i (answers: what must an expert beat to win this token?).
  ❷ Per-expert quantile of margins s_ij − α_i at rank q+1 sets the next bias, mean-centred, applied next step
     (answers: how far to move each bias without a step size?).
  ❸ Per-expert histogram of required biases, scatter-added locally and summed by one integer all-reduce
     (answers: a quantile over millions of margins sharded over ranks and micro-batches).
- **Release**: no balancing hyperparameter; quantile error ≤ one bin (a few 10⁻³ at B = 1000) and no
  measurable residual load imbalance in K3; one all-reduce of n·B counts per layer per step, independent of
  batch size, below 1% of exchanging raw margins. Carried K3's 896-expert pretraining (with MoonEP balancing
  each micro-batch across EP ranks).
- **Limits**: K3 publishes no QB-vs-sign ablation numbers (no MaxVio curve, no throughput delta); balances
  the global step, not each micro-batch (MoonEP still needed); one-step lag for causality; needs Top-(k+1)
  and a bounded score range for the histogram; bias frozen at inference.

## Slide plan (14 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Expert pools grew 3.5× in two model generations | prosperity | chart experts.py |
| 3 | Sign-update balancing leaves hot experts overloaded | pain | `%% big: +34%` |
| 4 | Loss-free balancing nudges each bias by ±γ | villain / suitors | fig_signstep (SVG) |
| 5 | Imbalance grows 2.6× from 16 to 64 experts | escalation, last suitor | chart maxvio.py |
| 6 | Insight: the balancing bias is a quantile | turn | fig_dual (SVG) |
| 7 | QB rebuilds every bias once per step | overview | fig_overview (SVG) |
| 8 | ❶ Top-(k+1) gives each token a cutoff | mechanism | fig_cutoff (SVG) |
| 9 | ❷ Each bias jumps to a margin quantile | mechanism | fig_quantile_1/2 (SVG build) |
| 10 | ❸ A histogram turns the quantile into one all-reduce | mechanism | fig_histogram (SVG) |
| 11 | Histogram traffic: under 1% of raw margins | release | chart traffic.py (analytic) |
| 12 | K3 trains 896 experts with no balancing knob | release / fit | two-col text |
| 13 | The cost: one step of lag, two levels of balance | limits | two-col text |
| 14 | Summary | | QR |
