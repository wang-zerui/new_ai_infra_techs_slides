# Story — deck 65: MuonClip (Muon + QK-Clip)

- **Where it lives** (prosperity): Kimi K2, a 1.04T-parameter MoE (32.6B active, 61 MLA layers × 64 heads)
  pretrained on 15.5T tokens with Muon, because Muon matches AdamW with ~52% of the training FLOPs
  (Moonlight scaling law, ~2× compute efficiency).
- **Pain number** (betrayal): with vanilla Muon, the max attention logit of a 53B-total / 9B-active MoE
  quickly exceeds **1000**; logits at that level bring loss spikes and occasional divergence.
- **Failed suitors**:
  1. Logit soft-cap: caps the logit after q·k is formed, but the q·k dot products (and the weights behind
     them) can still grow before the cap.
  2. QK-Norm: needs per-head keys to normalize; MLA never materializes them at inference (only the latent
     c_KV and the shared rotary key are cached), so it does not apply to K2's MLA.
- **Villain**: |q_i·k_j| ≤ ‖x_i‖‖x_j‖‖W_q‖‖W_k‖. RMSNorm bounds the x terms, so logit growth is spectral-norm
  growth of W_q and W_k. Muon's updates have a flat (full-rank) singular spectrum, which more often aligns
  with the weights' singular directions and grows singular values additively; the bilinear W_q W_kᵀ
  compounds it (K2 App. E, a hypothesis).
- **Turn / insight**: the logit is bilinear in (W_q, W_k), so scaling both by √γ scales it by exactly γ,
  and the forward pass already computes each head's max logit. Fix the weights, not the logits: after the
  optimizer step, rescale only the heads whose max logit exceeded τ.
- **Mechanisms**:
  ❶ Per-head trigger: γ_h = min(1, τ / S_max^h) instead of one global γ (answers: only a few heads explode;
     clipping all heads over-regularizes).
  ❷ MLA-aware split: q_C and k_C scaled by √γ, q_R by γ, the shared k_R untouched (answers: MLA's key has a
     head-specific part and a part shared by all heads).
  ❸ Fused into the optimizer: Muon step (momentum, Newton-Schulz, update RMS matched to AdamW via
     0.2·√max(n,m), decoupled weight decay) then QK-Clip, after the update, without changing the current
     step's forward/backward (answers: keep Muon's efficiency and hyper-parameters).
- **Release**: K2 trained 15.5T tokens with **zero loss spikes**; max logit capped at τ = **100** (vs > 1000
  without), then decays on its own; QK-Clip touched **12.7%** of heads in the first **70K** steps and none
  after; τ = 30 on a 0.5B/3B MoE changes loss negligibly, no significant downstream drop.
- **Limits**: τ is a new hyper-parameter; clipping acts from the next step, so the measured step runs
  unclipped; S_max is a max over the global batch per head; the no-harm evidence is small-scale plus
  self-deactivation (no full-scale ablation); per-architecture surgery for MLA; QK-Norm remains the simpler
  fix for non-MLA attention (GLM-4.5 with Muon, deck 66).

## Slide plan (15 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Muon matches AdamW with half the FLOPs | prosperity | chart_muon |
| 3 | Vanilla Muon drives max logits past 1000 | pain | `%% big: >1000` |
| 4 | Muon inflates the weights that set logits | villain | fig_villain (SVG) |
| 5 | Soft-cap and QK-Norm both fall short | failed suitors | fig_suitors (SVG) |
| 6 | Insight: fix the weights, not the logits | turn | bullets + boxed sentence (build) |
| 7 | MuonClip adds one clip after every Muon step | overview | fig_overview (SVG) |
| 8 | ❶ Only heads above τ get clipped | mechanism | fig_heads (SVG) |
| 9 | ❷ MLA splits the scale: √γ, √γ, γ | mechanism | fig_mla (SVG) |
| 10 | ❸ The clip runs inside the optimizer step | mechanism | fig_algo (SVG pseudocode) |
| 11 | Max logit held at 100 instead of >1000 | release | chart_maxlogit |
| 12 | Zero loss spikes in 15.5T tokens | release | `%% big: 0` |
| 13 | QK-Clip switched itself off after 70K steps | result | chart_activity |
| 14 | The cost: a threshold and MLA surgery | limits | two-col |
| 15 | Summary | | QR |
