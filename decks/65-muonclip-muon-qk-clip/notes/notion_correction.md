# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Token-efficient Muon updates plus QK-Clip: after every step the Q/K projection weights are rescaled against threshold tau, killing attention-logit explosion at the source.
**After:** Token-efficient Muon updates plus QK-Clip: after each step, only heads whose max attention logit S_max exceeds tau (100 in K2) get their Q/K weights rescaled by tau / S_max, capping logits at tau from the next step.
**Why:**
- Per-head factor gamma_h = min(1, tau / S_max^h): heads under tau are untouched, so not every step rescales the weights ([K2] §2.1 p.3, Algorithm 1; App. D: per-head to avoid over-regularizing other heads).
- "Killing ... at the source" overstates: the clip is post-update, does not alter the current step, and caps logits at tau rather than preventing them from reaching it ([K2] §2.1 p.3, Fig. 2 right; tau = 100 p.4).

## Impact / Measured Results
**Before:** Zero loss spikes across K2's full 15.5T-token pretraining (1T MoE), with no downstream degradation from clipping.
**After:** Zero loss spikes across K2's full 15.5T-token pretraining (1T MoE); clipping acted on 12.7% of heads in the first 70K steps, then went inactive; no significant downstream degradation in a small ablation (3B-total MoE, tau 30).
**Why:**
- Zero loss spikes kept: matches [K2] abstract and Fig. 3.
- "No downstream degradation" rests on a small-scale ablation (0.5B active / 3B total, tau = 30) plus self-deactivation in K2; there is no full-scale ablation ([K2] App. D p.30, Fig. 12).
- 12.7% / 70K steps added to show how rarely the clip acts ([K2] App. D p.30).

## Other properties
none
