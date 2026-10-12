# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Adjusts the routing distribution directly from quantile statistics instead of aux-loss-free bias updates (unstable at ~1k experts); needs no auxiliary loss.
**After:** Each step sets every expert's selection bias to minus the (1 - k/n)-quantile of its routing margins against each token's Top-(k+1) cutoff, estimated by histogram plus one all-reduce, replacing sign-step bias updates K3 calls not well behaved near 1,000 experts; mixture weights exclude the bias.
**Why:**
- QB sets the per-expert selection bias; the mixture weights p_ij exclude the bias, so the routing weights are unchanged and only which experts are selected changes ([K3] Eq. 13 p.8).
- Mechanism: Top-(k+1) cutoff per token, bias = minus the (1 - k/n)-quantile of margins, mean-centred, applied next step ([K3] Eq. 14 p.9, App. C); quantile from per-expert histograms with one integer all-reduce per layer per step ([K3] App. D p.45).
- "Unstable at ~1k experts": K3 only says nearly 10^3 experts "exceeds the regime in which existing auxiliary-loss-free bias updates remain well behaved" and publishes no measurement ([K3] p.6-7); wording softened to K3's claim.
- "Needs no auxiliary loss" is not stated in the sources read; dropped.

## Impact / Measured Results
**Before:** Keeps load balanced at ~10^3 experts, avoids expert under-training and directly improves EP training throughput.
**After:** K3 balances 896 experts per layer with no step size, reporting no measurable residual imbalance and histogram traffic under 1% of raw-margin exchange; no QB ablation or throughput number is published (EP micro-batch balance comes from MoonEP).
**Why:**
- 896 routed experts, 16 per token ([K3] §2.3, Table 1); "no measurable residual load imbalance" and "below 1%" of the cost of exchanging raw margins ([K3] App. D p.45).
- K3 gives no throughput number for QB and attributes per-micro-batch EP balance and its throughput to MoonEP ([K3] §5.2.1); "directly improves EP training throughput" removed.
- "Avoids expert under-training" is not stated as a QB result in the sources read; dropped.

## Other properties
none
