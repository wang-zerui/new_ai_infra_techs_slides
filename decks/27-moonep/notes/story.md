# Story: MoonEP (deck 27, container entry)

- **Where it lives** (prosperity): Kimi K3 pre-training, a 2.8T-parameter MoE (104B active, 896 routed experts, 16 per
  token) trained with expert parallelism; MoonEP is Moonshot's MIT-licensed EP library released with it (July 2026).
- **Pain number** (betrayal): DeepEP v2 end-to-end training **runs out of memory** at maxvio 20 (hottest expert at 21×
  the average load) on 8×H20, after its iteration time has crept up with imbalance; shifting activation shapes
  fragment memory.
- **Failed suitors**: (1) routing-side balancing (aux-loss-free bias, Quantile Balancing, decks 55/56) evens loads over
  the global batch, with the bias updated for the *next* step, so a single micro-batch can still be skewed;
  (2) offline redundant experts (EPLB / DeepSeek-V3) are re-planned from historical statistics, about every 10 minutes
  in V3 serving; (3) fixed-cap cloning (ECHO, UltraEP) presets slots or a token cap and stops when no plan fits.
- **Villain**: per-micro-batch routing decides how many tokens each rank receives, so the busiest rank sets the layer
  time *and* every received shape is dynamic: the host must sync each layer, or buffers must cover the S×K×R worst case.
- **Turn / insight**: a perfectly balanced plan always exists if each rank reserves E/R redundant-expert slots (proved,
  tight up to ⌈E(R−1)/R²⌉); and once every rank receives exactly S×K tokens, every shape is static.
- **Mechanisms** (library components, deep dive in deck 20):
  ❶ dynamic redundant experts — answers *skew*: replicate hot experts onto under-loaded ranks for this micro-batch,
     prefetch weights forward, reduce gradients home backward;
  ❷ online GPU planning — answers *staleness/overhead*: one GPU kernel plans every micro-batch and layer;
  ❸ zero-copy static shapes — answers *dynamic shapes*: tokens land in expert-grouped slots of a fixed S×K buffer, no
     host sync, no permute copies.
- **Release**: per-layer EP comm flat (8.7-9.1 ms) while DeepEP v2 grows 15% to 10.2 ms; iteration time up to 11.1%
  lower at maxvio 15; at maxvio 20 MoonEP trains while DeepEP OOMs.
- **Limits**: weight prefetch adds ≈0.5-0.75 ms per dispatch; slightly slower when routing is balanced (−1.9% e2e at
  maxvio 0.2); NVLink domain only; prefetch/reduce pool of E/R experts per projection (4.2 GB BF16 + 8.5 GB FP32 at the
  benchmark shape); per-expert skew inside a rank remains (needs a workload-aware GEMM scheduler); results are 8×H20
  with synthetic lognormal routing.

## Slide plan (as built: 15 logical slides, 8 minutes)

| # | Headline | Visual |
|---|---|---|
| 1 | Title | |
| 2 | Kimi K3 trains a 2.8T MoE with expert parallelism | fig_ep_layer |
| 3 | Skewed routing crashes DeepEP v2 training | big: OOM (pain) |
| 4 | The hottest GPU sets every layer's time | fig_skew |
| 5 | Balancing on average misses each micro-batch | fig_suitors + callout |
| 6 | Insight: E/R replica slots always suffice | fig_insight + callout (insight) |
| 7 | MoonEP stacks four calls on NVLink buffers | fig_arch (overview) |
| 8 | ❶ Hot experts get replicas for one micro-batch | fig_replicas |
| 9 | ❷ One GPU kernel plans every layer | fig_gpu_pipeline |
| 10 | ❸ Balance pins the token buffer at S×K | chart buffer_vs_ep (analytic) |
| 11 | EP comm stays flat; DeepEP v2 grows 15% | chart comm_total (headline result) |
| 12 | Iterations up to 11.1% faster; DeepEP OOMs at 20 | chart e2e_saving (headline result) |
| 13 | Kimi K3 runs on it; DeepEP added its primitives | fig_timeline (adoption) |
| 14 | Costs: prefetch time, a weight pool, NVLink only | bullets (limits) |
| 15 | Summary | QR |

Cross-references: deck 20 (deep dive of ❶❷❸), deck 24 (DeepEP v2 baseline), deck 55 (loss-free balance routing),
deck 56 (Quantile Balancing).
