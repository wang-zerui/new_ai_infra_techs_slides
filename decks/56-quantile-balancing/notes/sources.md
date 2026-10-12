# Sources — deck 56: Quantile Balancing

## Sources read

| Tag | Source | Where |
|---|---|---|
| [K3] | Kimi Team, "Kimi K3: Open Frontier Intelligence", technical report, arXiv 2607.24653. https://arxiv.org/abs/2607.24653 (cached text `arxiv-2607.24653.txt`) | Abstract (p.1); §1 (p.2); §2.3 intro (p.6-7); Eq. 11 (p.7); §2.3.3 Eq. 13-14 + Fig. 5 (p.8-9); Table 1 (p.11); §3.3 (p.11); §5.2.1 MoonEP (p.19-20); App. C Eq. 20-27 + Alg. 1 (p.43-44); App. D (p.44-45) |
| [K3-blog] | Kimi K3 tech blog, https://www.kimi.com/en/blog/kimi-k3 (downloaded 2026-10-13) | "Architecture and Infrastructure" paragraph |
| [K3-cfg] | Kimi-K3 Hugging Face `config.json` + `modeling_kimi_linear.py` (https://huggingface.co/moonshotai/Kimi-K3, downloaded 2026-10-13) | `num_experts` 896, `num_experts_per_token` 16, `moe_router_activation_func` sigmoid, `topk_method` noaux_tc; `KimiMoEGate.forward` (bias `e_score_correction_bias` added for selection only, weights gathered from unbiased scores, inference-only) |
| [LFB] | L. Wang et al., "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts", arXiv 2408.15664 (DeepSeek-AI) | Eq. 3 + Alg. 1 (p.4); Fig. 2 (p.4); §4.1 (p.5); Table 2 + Eq. 4 MaxVio (p.6); Fig. 4 + §4.3 (p.7); Table 5 (p.11) |
| [BIP] | Y. Sun, "Binary-Integer-Programming Based Algorithm for Expert Load Balancing in Mixture-of-Experts Models", arXiv 2502.15451v2 | Abstract + §1 (p.1); Table 1 (p.5); §4.1 metric definitions (p.5-6); Table 2 and Table 3 (p.6) |
| [DSV3] | DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv 2412.19437 (cached) | §4.2 Hyper-Parameters (p.22-23) |
| [ET] | H. Sun et al., "Expert Threshold Routing ...", arXiv 2603.11535 | Abstract (EMA thresholds, variable experts per token); cited only as related work |
| [HF-blog] | NormalUhr, "MoE load balancing" (Hugging Face blog) | DeepSeek-V3 section: bias update speed too large thrashes, too small adapts slowly (qualitative context only) |
| [MoonEP] | sibling deck 27 (MoonEP) and [K3] §5.2.1 | per-micro-batch rank balance with redundant experts |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 2.8T total / 104B activated | Kimi K3 size | [K3] abstract p.1; Table 1 p.11 (2.78T / 104.2B) |
| 896 routed experts, 16 per token, sparsity 56 | K3 Stable LatentMoE | [K3] §2.3 p.6; Table 1 p.11; [K3-cfg] |
| 2 shared experts; 93 layers | K3 | [K3] Table 1 p.11 |
| 384 routed, 8 per token | Kimi K2 | [K3] Table 1 p.11 |
| 256 routed, 8 per token | DeepSeek-V3 | [DSV3] §4.2 p.22 |
| 64 routed experts, 6 activated | models on which loss-free balancing was validated (1B and 3B) | [LFB] Table 5 p.11 |
| γ (u) = 0.001, tuned at the 1B scale only | loss-free update rate | [LFB] §4.1 p.5, §4.3 p.7 |
| u = 1e-4 slow early, u = 1e-2 fluctuates late | update-rate dilemma | [LFB] Fig. 4 + §4.3 p.7 |
| γ = 0.001 for the first 14.3T tokens, 0.0 for the last 500B | DeepSeek-V3 bias update speed schedule | [DSV3] §4.2 p.22-23 |
| AvgMaxVio aux loss 0.3852 (16 experts) / 0.7158 (64 experts) | average over training batches of MaxVio_batch | [BIP] Table 2, Table 3 p.6 |
| AvgMaxVio loss-free 0.1275 (16) / 0.3366 (64) | same | [BIP] Table 2, Table 3 p.6 |
| AvgMaxVio BIP (T = 14) 0.0547 (16) / 0.0529 (64) | same | [BIP] Table 2, Table 3 p.6 |
| SupMaxVio loss-free 1.7702 (16) / 2.7121 (64) | worst training batch | [BIP] Table 2, Table 3 p.6 |
| 16 experts k = 4 (0.3B); 64 experts k = 8 (1.1B); softmax router; u = 0.001; α = 0.1 | BIP experimental setup | [BIP] Table 1 p.5, §4.1 p.5, Table 2/3 captions |
| q = mk/n | target load per expert | [K3] §2.3.3 p.8 |
| (1 − k/n)-quantile, rank q + 1 | QB update | [K3] Eq. 14 p.9, App. C Eq. 26 p.44 |
| m = 8 tokens, n = 4 experts, k = 1, q = 2; loads (4, 3, 1, 0) → (2, 2, 2, 2) | worked example setting | [K3] Fig. 5 caption p.8 |
| B = 1000 bins → error at most a few 10⁻³ | histogram resolution | [K3] App. D p.45 |
| "a few hundred bins per expert" | histogram size in main text | [K3] §2.3.3 p.9 |
| w = (b_max − b_min + 2)/B; range [b_min − 1, b_max + 1] | bin width and range | [K3] App. D p.45 |
| one integer all-reduce of n·B values per layer per step, independent of m | histogram communication | [K3] App. D p.45 |
| below 1% of the cost of exchanging raw margins every micro-batch | measured in K3's configuration | [K3] App. D p.45 |
| "no measurable residual load imbalance" | K3 observation with the histogram estimator | [K3] App. D p.45 |
| "equilibrates within a few update steps even for nearly 10³ experts" | qualitative | [K3] App. C p.44 |
| ≈ 2.5× scaling efficiency over Kimi K2 | joint effect of KDA, AttnRes, Stable LatentMoE (incl. QB), data, recipes | [K3] abstract p.1, §3.2 p.10 |

### Derived numbers (written out)

- **+34%** (slide 3): MaxVio = max_j Load_j / mean Load − 1 ([LFB] Eq. 4; [BIP] §4.1). AvgMaxVio 0.3366 means the
  busiest expert carried 1.34× the mean load on an average training batch, i.e. +34% over its fair share.
  Likewise 0.1275 → +13% at 16 experts; SupMaxVio 2.7121 → worst batch 3.7× the mean.
- **2.6×** (slide 5): 0.3366 / 0.1275 = 2.64.
- **3.5×** (slide 2): 896 / 256 = 3.5 (K3 vs DeepSeek-V3 routed experts). **14×**: 896 / 64 = 14.
- **n·B = 896,000 counts per layer per step** (slide 11): n = 896 experts × B = 1000 bins (B = 1000 is the
  value K3 uses for its error statement; the main text says "a few hundred bins", so 1000 is an upper bound).
- **Analytic traffic chart** (`charts/traffic.py`): values per MoE layer per step; raw margins = m·n, histogram =
  n·B with n = 896, B = 1000, m = global tokens per step on a log axis. The shaded "millions" region is from
  [K3] §2.3.3 ("margins number in the millions"). No measured traffic numbers are plotted; the < 1% figure in
  the callout is K3's own measurement.
- **1000× fewer at m = 10⁶** (traffic chart callout): (m·n)/(n·B) = m/B = 10⁶/1000 = 1000 (analytic).
- **"under 0.9 million counts"** (speaker notes): n·B = 896 × 1000 = 896,000.

### Illustrative toy example (constructed for the figures, not from a source)

Same setting as K3 Fig. 5 (m = 8, n = 4, k = 1, q = 2, loads 4/3/1/0 → 2/2/2/2) but with our own score values,
verified by `scratchpad/tools-56-57-59/qb_toy_verify.py`:

| token | s(E1) | s(E2) | s(E3) | s(E4) | route | cutoff α | margins s − α (E1..E4) | after QB |
|---|---|---|---|---|---|---|---|---|
| t1 | .90 | .50 | .30 | .20 | E1 | .50 | +.40 0 −.20 −.30 | E1 |
| t2 | .85 | .55 | .20 | .10 | E1 | .55 | +.30 0 −.35 −.45 | E1 |
| t3 | .60 | .30 | .50 | .25 | E1 | .50 | +.10 −.20 0 −.25 | E3 |
| t4 | .45 | .20 | .15 | .40 | E1 | .40 | +.05 −.20 −.25 0 | E4 |
| t5 | .45 | .80 | .30 | .20 | E2 | .45 | 0 +.35 −.15 −.25 | E2 |
| t6 | .45 | .70 | .35 | .20 | E2 | .45 | 0 +.25 −.10 −.25 | E2 |
| t7 | .30 | .58 | .25 | .50 | E2 | .50 | −.20 +.08 −.25 0 | E4 |
| t8 | .45 | .30 | .75 | .30 | E3 | .45 | 0 −.15 +.30 −.15 | E3 |

Current bias b(t) = 0. Per-expert thresholds (3rd-largest margin): θ = (+.10, +.08, −.10, −.15); new bias
b̂ = −θ = (−.10, −.08, +.10, +.15); mean .0175 subtracted (does not change Top-k). Loads 4/3/1/0 → 2/2/2/2.

## Mechanism facts (no numbers) used on slides

- Router: s_i = Sigmoid(W_r x_i); T_i = argtopk(s_i + b); p_ij = s_ij / Σ_{r∈T_i} s_ir; b omitted from p, so it
  steers dispatch only ([K3] Eq. 13 p.8; [K3-cfg] `KimiMoEGate.forward`).
- Sign update: b_j ← b_j + γ·sign(ℓ̄ − ℓ_j), γ trades slow adaptation against load oscillation ([K3] p.8; [LFB] Alg. 1).
- QB: Top-(k+1) on s + b(t); first k = routes, (k+1)-th = cutoff α_i; b̂_j = −quantile_{1−k/n}(s_:,j − α);
  b ← b̂ − mean(b̂); takes effect next step (causality); frozen at inference ([K3] Eq. 14 p.9).
- Dual view: balanced assignment LP (each token k experts, each expert mk/n tokens) is exact (b-matching
  integrality); coordinate minimization over α (token) and β (expert) gives the same quantile on both axes;
  routing needs only β = −b; the sign update is SignSGD on ∂L/∂β_j = mk/n − load_j ([K3] App. C).
- BIP uses inequality constraints → max(0,·) clipping → can only suppress over-selected experts, slower
  equilibration in K3's experiments; Expert Threshold routing keeps EMA thresholds and a variable number of
  experts per token ([K3] App. C p.44).
- Histogram: required bias r_ij = α_i − s_ij; per-expert counts H ∈ N^{n×B}, scatter-added over all
  micro-batches with no communication; one all-reduce at the end of the step; first bin whose cumulative
  count reaches ⌈q⌉, linear interpolation; mean-centred; counts additive, so invariant to sharding; EMA across
  steps optional ([K3] App. D).
- MoonEP balances each micro-batch across EP ranks with redundant experts (≤ E/R per rank), because
  router-level balance holds over the step, not per micro-batch ([K3] §5.2.1; deck 27).

## Discrepancies (Notion row vs primary source)

1. **"unstable at ~1k experts"**: K3 says balancing nearly 10³ experts "exceeds the regime in which existing
   auxiliary-loss-free bias updates remain well behaved" (p.6-7) but publishes no measurement of the
   instability. The deck uses a third-party measurement ([BIP]) for the trend and says so.
2. **"directly improves EP training throughput"**: K3 gives no throughput number for QB. Imbalance is said to
   slow EP training (p.8, citing Step 3.5 Flash); K3 attributes per-micro-batch EP balance and its throughput
   to MoonEP (§5.2.1). The deck does not claim a throughput gain for QB.
3. **"Adjusts the routing distribution directly from quantile statistics"**: QB sets the per-expert selection
   bias; the mixture weights p_ij exclude the bias (Eq. 13), so the routing *weights* are unchanged; only
   which experts are selected changes.
4. **Bin count**: K3 §2.3.3 says "a few hundred bins per expert"; App. D quotes the error bound for B = 1000.
   The deck uses B = 1000 only as the error-bound case and as an upper bound in the analytic chart.
5. **Source-internal ([BIP] Table 1)**: the "Routed experts" and "Activated routed experts" rows are swapped
   (4/8 vs 16/64); the text and table captions give 16 experts with k = 4 and 64 experts with k = 8. The deck
   uses the captions.
