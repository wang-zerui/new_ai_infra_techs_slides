# Story — deck 63: anticipatory routing

- **Where it lives** (prosperity): DeepSeek-V4 pre-training (V4-Pro 1.6T/49B, 384 routed experts, top-6, 33T tokens,
  batches up to 94.4M tokens); every MoE layer's router picks experts from the current parameters.
- **Pain** (betrayal): loss spikes in trillion-parameter MoE training. Rollbacks restored the run only temporarily:
  spikes came back. **No measured pain number is published** (no spike magnitude, frequency or lost compute), so the
  pain slide is a schematic, not a `%% big:` number.
- **Failed suitors**: rollback-and-retry (the obvious fix) treats the symptom; spikes recur because the cause is
  still there.
- **Villain**: synchronous coupling. Spikes are tied to outliers in MoE layers, and routing appears to amplify them:
  the same update that perturbs the features also changes which experts the next step routes to (hypothesized loop,
  the paper says the mechanism is not understood).
- **Turn / insight**: empirically, decoupling the backbone's and the router's updates "significantly improves
  stability": route step t with θ_{t−Δt}, compute with θ_t.
- **Mechanisms**:
  ❶ Lookahead routing pass (answers: computing old-parameter routing would need a second copy of the parameters):
     at step t−Δt, prefetch step t's batch, run a routing-only forward pass with θ_{t−Δt} (already resident), cache
     the top-K indices; at step t, use cached indices; features and gradients with θ_t. Overlapped with the
     pipeline and EP communication.
  ❷ Spike-triggered mode (answers: ≈ 20% wall-clock overhead if always on): automatic detection → short rollback →
     anticipatory mode for a while → back to standard training.
- **Release**: spikes averted, "negligible" overall overhead, no loss in model quality (qualitative); ≈ 20% while
  active. Analytic: run-level overhead ≈ 20% × fraction of steps in anticipatory mode.
- **Companion**: SwiGLU clamping (linear part to [−10, 10], gate ≤ 10) suppresses the outliers directly.
- **Limits**: no published Δt, detector, ablation or spike statistics; ≈ 20% while active; needs rollback machinery
  and lookahead data loading; principles not understood (the paper says so).

## Slide plan (12 logical slides)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | The router picks experts from current parameters | prosperity | fig_route |
| 3 | Rollbacks restore the run, not the stability | pain + failed suitor | fig_spikes (schematic) |
| 4 | Routing and features update in lockstep | villain | fig_loop |
| 5 | Insight: let routing lag the backbone by Δt | turn | text + build |
| 6 | Step t computes with θ_t, routes with θ_{t−Δt} | overview | fig_versions |
| 7 | ❶ Routing indices are computed Δt steps early | mechanism | fig_lookahead |
| 8 | ❷ Anticipatory mode turns on only after a spike | mechanism | fig_modes |
| 9 | ≈ 20% slower while on, near zero overall | result | chart overhead (analytic) |
| 10 | V4 pairs it with SwiGLU clamping | companion | fig_clamp |
| 11 | The cost: an extra pass, rollbacks, no theory | limits | two-col |
| 12 | Summary | | QR |
