# Story: MoonEP mechanisms (deck 20, deep dive; library overview is deck 27)

- **Where it lives** (prosperity): every MoE layer of Kimi K3's 2.8T pre-training, inside MoonEP's dispatch/combine;
  at the README benchmark shape each GPU sends S×K = 65,536 token copies per layer to E = 384 experts on R = 8 GPUs.
- **Pain number** (betrayal): the established fix for skew, redundant experts, is planned offline: DeepSeek-V3
  re-plans its replicas from load statistics about **every 10 minutes**, while routing changes every micro-batch.
- **Failed suitors**: (1) offline replicas (EPLB / DeepSeek-V3) are stale and DeepSeek itself deferred dynamic
  redundancy because it needs a fast global algorithm *and* fusion with the dispatch kernel; (2) dynamic-shape EP
  libraries (DeepEP v2 training mode) read received counts on the host every layer, or must reserve S×K×R buffers.
- **Villain**: received token counts are decided by routing at run time, per micro-batch and layer; anything that
  needs them on the host, or sized in advance without a guarantee, breaks.
- **Turn / insight** (provable fact): fill each under-loaded GPU to exactly S×K from a single overloaded GPU; at most
  R−1 fills, each GPU imports from one home GPU, so ≤ E/R replica slots always suffice (K3 App. E), and the bound is
  tight (⌈E(R−1)/R²⌉).
- **Mechanisms** (challenge → mechanism):
  ❶ *how many replicas, which tokens move?* → dynamic redundant experts: greedy fill + largest-expert-first split +
     contiguous token ranges; replicas' weights prefetched forward, FP32 grads reduced home backward.
  ❷ *how to plan every layer without the host?* → one cooperative GPU kernel: counts gathered to GPU 0 over NVLink,
     GPU 0 plans in one warp and multicasts, others sort tokens meanwhile.
  ❸ *how to avoid host sync and worst-case buffers?* → fixed [NvS, H] buffer in 2·E/R expert-grouped segments,
     tokens written in place, views handed to the GEMM.
- **Release**: plan per micro-batch and layer in ≈45 µs (≈2% of dispatch, estimated) instead of every ~10 minutes;
  combine latency flat at 1.93-1.98 ms while DeepEP v2's grows 16%; buffer fixed at S×K (+padding).
- **Limits**: weight prefetch ≈0.5-0.75 ms per dispatch (≈22-29%); pool of E/R experts per projection + FP32 grads;
  near-optimal (not optimal) replica count; per-expert skew inside a GPU remains; NVLink domain only.

## Slide plan (as built; logical slide numbers, PDF pages = +2 after slide 7 because of its 3-step build)

| # | Headline | Visual |
|---|---|---|
| 1 | Title | |
| 2 | Every GPU must receive exactly S×K token copies | fig_counts (running example) |
| 3 | Offline replicas are re-planned every ten minutes | big: 10 min (pain) |
| 4 | Dynamic counts force a host sync or S×K×R | fig_dilemma |
| 5 | Insight: one-source fills need ≤ E/R replicas | chart replica_bound (insight) |
| 6 | Three mechanisms run inside every MoE layer | fig_overview |
| 7 | ❶ Fill the emptiest GPU from the fullest | fig_plan_1..3 (build) |
| 8 | ❶ Expert copies split into contiguous ranges | fig_route |
| 9 | ❷ GPU 0 plans while the others sort | fig_kernel |
| 10 | ❸ Tokens land in a fixed, expert-grouped buffer | fig_layout |
| 11 | ❶ Replicas borrow weights and return FP32 grads | fig_replica_io |
| 12 | Plan takes ≈2% of dispatch, prefetch up to 29% | chart dispatch_breakdown (headline result) |
| 13 | Combine stays flat; DeepEP v2 grows 16% | chart combine_flat (headline result) |
| 14 | Limits: prefetch, pool memory, per-expert skew | bullets |
| 15 | Summary | QR |

Cross-references: deck 27 (MoonEP library overview, end-to-end results), deck 24 (DeepEP v2), deck 55 (loss-free
balance routing), deck 56 (Quantile Balancing).
