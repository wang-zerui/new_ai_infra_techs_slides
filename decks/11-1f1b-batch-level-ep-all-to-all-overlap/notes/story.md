# Story — deck 11: 1F1B / batch-level EP all-to-all overlap

- **Where it lives** (prosperity): Megatron-Core MoE (v0.14+), fine-grained MoE pretraining; DeepSeek-V3
  685B (37B active, 256 experts, top-8, 58 MoE layers) on 1,024 H100 with EP64 spanning 8 nodes.
- **Pain number** (betrayal): even after DeepEP, EP all-to-all is still **30–40%** of DeepSeek-V3
  iteration time on H100 (40–60% before any optimization).
- **Failed suitors**:
  1. Faster dispatchers (DeepEP/HybridEP) shrink but cannot hide the all-to-all: it stays on the critical path.
  2. Overlapping inside one micro-batch: cross-node all-to-all is about half of layer time and the adjacent
     GEMMs are too short; DualPipe hides it but needs a bidirectional schedule and 2× parameters;
     merging FWD-FWD costs 2× activation memory and hides less (forward ≈ ½ backward).
- **Villain**: the dependency chain of a single micro-batch (attention → dispatch → experts → combine →
  next attention): every all-to-all sits between two computations that need its result.
- **Turn / insight**: the 1F1B steady state already executes two *independent* micro-batches back to back
  (forward of n+1, backward of n); pair them layer by layer and each pass's all-to-all hides under the
  other's compute.
- **Mechanisms**:
  ❶ Merged FWD-BWD on two CUDA streams (answers: no independent work in one micro-batch).
  ❷ W/D split, `--delay-wgrad-compute` (answers: backward dispatch must wait for the whole backward MLP).
  ❸ Pipeline integration: one extra warmup micro-batch under interleaved 1F1B (answers: in plain 1F1B the
     paired F and B can belong to the same micro-batch).
- **Release**: EP communication share 30–40% → **under 5%** of iteration time (93% overlap ratio), DeepSeek-V3
  on H100; part of the 368 TFLOPS/GPU configuration on 1,024 H100.
- **Limits**: extra buffers (feasible on H100 only after FP8 freed memory); SM carve-out for the dispatcher
  (20 SMs, ~20% GEMM efficiency); first FWD / last BWD stay exposed; prerequisites (EP>1, VPP if PP>1, no
  full/moe recompute, MTP ≤ 1). Turned OFF where EP fits in NVLink (DeepSeek-V3 on GB200 NVL72).

## Slide plan (14 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Fine-grained MoE adds two all-to-alls per layer | prosperity | fig_moe_layer (SVG) |
| 3 | EP all-to-all still eats a third of each step | pain | `%% big: 30–40%` |
| 4 | One micro-batch cannot hide its own all-to-all | villain | fig_baseline (SVG timeline) |
| 5 | Earlier fixes shrink or hide it at a price | failed suitors | two-col text |
| 6 | Insight: adjacent 1F1B passes are independent | turn | boxed sentence (build) |
| 7 | Merge each 1F1B step into one F+B pair | overview | fig_merge (SVG) |
| 8 | ❶ Two CUDA streams swap compute and all-to-all | mechanism | fig_layer_1 (SVG) |
| 9 | ❷ W/D split releases backward dispatch early | mechanism | fig_layer_2 (SVG, same canvas) |
| 10 | ❸ One extra warmup forward decouples every pair | mechanism | fig_warmup (SVG) |
| 11 | EP all-to-all drops from 30–40% to under 5% | release | chart a2a_share |
| 12 | On H100 always on; on NVL72 a tuning choice | result/limits | fig_onoff (SVG grid) |
| 13 | The cost: buffers, SMs, and exposed edges | limits | two-col text |
| 14 | Summary | | QR |
