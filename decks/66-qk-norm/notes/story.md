# Story — deck 66: QK-Norm

- **Where it lives** (prosperity): GLM-4.5, a 355B-total / 32B-active MoE pre-trained on 23T tokens with Muon on every
  weight matrix. Its architecture adds QK-Norm "to stabilize the range of attention logits"; among the models in its
  comparison table (GLM-4.5-Air, DeepSeek-V3, Kimi K2) only GLM-4.5 has it. Every Muon-trained frontier MoE now carries
  some logit control (K2: QK-Clip; GLM-5: Muon Split; DeepSeek-V4: q/KV RMSNorm).
- **Pain number** (betrayal): with vanilla Muon, a 53B-total MoE's max attention logit passes **1,000** (K2 Fig. 2),
  which brings loss spikes and occasional divergence. Escalation: an 8B ViT with Adam reached logits > 50,000 and diverged.
- **Failed suitors**:
  1. Lower the learning rate (original ViT: 1e-3 → 4e-4 at ViT-H): stable but slower; gives up quality.
  2. Soft-cap the logits (Gemma 2): q·k still grows under the cap (K2); Gemma 3 replaced it with QK-norm.
  3. Clip W_q / W_k after the step (QK-Clip, K2): works, but needs per-head max-logit statistics and an optimizer hook
     (deck 65).
- **Villain**: the logit is bilinear in two weight matrices that nothing bounds: |q·k| ≤ ‖x_i‖‖x_j‖‖W_q‖‖W_k‖. Pre-norm
  bounds x, not W; Muon's full-rank updates grow ‖W‖ faster.
- **Turn / insight**: RMSNorm is scale-invariant (RMSNorm(c·q) = RMSNorm(q)). Normalize q and k *after* the projection and
  the weight norms drop out of the logit, by construction.
- **Mechanisms**:
  ❶ Per-head RMSNorm on q and k (after projection, before RoPE): every q, k lands on a sphere of radius √d (× gains).
     Answers: logits that grow with ‖W_q‖‖W_k‖.
  ❷ The bound: RoPE is a rotation, so |logit| ≤ √d · max|g_q| · max|g_k| = 11.3 at unit gains (d = 128); sharpness becomes a
     learned temperature in 2 × 128 gains per layer (shared across heads). Answers: bounded but still expressive.
  ❸ A forward-pass op: no optimizer hook, no statistics, cached keys are already normalized, attention kernel unchanged.
     Answers: QK-Clip's coupling to the optimizer and kernels.
- **Release**: max logit bounded at √128 × gains ≈ 11.3 at init (analytic) instead of > 1,000 measured; across labs the
  same fix turned divergence into training (8B ViT at LR 1e-3; a 1.2B LM at LR 0.3; Chameleon-7B); DeepSeek-V4 drops
  QK-Clip because RMSNorm on q and KV entries already prevents logit explosion.
- **Limits**: needs materialized per-head keys, so it does not fit MLA (K2 → QK-Clip; GLM-5 on MLA dropped it and relies
  on Muon Split); gains can still grow (soft bound); does not fix output-logit drift (z-loss); not sufficient alone in some
  runs (OLMo 2 and Chameleon-34B also reordered norms); GLM-4.5 publishes no ablation or logit curves.

## Slide plan (15 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Muon-era MoEs all carry a logit control | prosperity | fig_models (SVG matrix) |
| 3 | Muon lets attention logits run past 1,000 | pain | `%% big: > 1,000` |
| 4 | Nothing in the attention block bounds Wq, Wk | villain | fig_explode (SVG chain) |
| 5 | Lower LR, soft-caps and clipping act too late | suitors | bullets + build callout |
| 6 | Insight: RMSNorm cancels the weight norm | turn | boxed sentence |
| 7 | QK-Norm sits between projection and RoPE | overview | fig_block (SVG GLM-4.5 attention) |
| 8 | ❶ Per-head RMSNorm puts q and k on a sphere | mechanism | fig_sphere_1 |
| 9 | ❷ Logits bounded by √d times the gains | mechanism | fig_sphere_2 (build) |
| 10 | ❸ A forward-pass op, not an optimizer hook | mechanism | fig_where (SVG two lanes) |
| 11 | Bound: 11.3 × gains, versus > 1,000 unchecked | result | chart_logits (log scale) |
| 12 | Runs that diverged without it now train | result | bullets + callout |
| 13 | QK-Norm cannot reach MLA's absorbed keys | limit | fig_mla (SVG) |
| 14 | Cheap, but not a complete stability fix | cost / limits | two-col |
| 15 | Summary | release | bullets + QR |
