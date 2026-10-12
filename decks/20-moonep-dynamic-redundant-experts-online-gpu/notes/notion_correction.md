# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Three co-designed pieces in one library: (1) replicates hot experts online according to the current batch's routing distribution so every EP rank receives exactly S x K / EP tokens, with the backward pass reducing redundant replicas' gradients back to the home rank (mathematically equivalent to no redundancy); (2) a GPU kernel solves redundant-expert placement + token assignment online for every micro-batch, with no host involvement and negligible cost relative to the all-to-all; (3) permute/unpermute is fused into dispatch/combine with a fixed static S x K buffer instead of a dynamic S x K x R shape, removing per-layer host-device syncs and intermediate copies.
**After:** Three co-designed pieces in one library: (1) replicates hot experts online according to the current batch's routing so every EP rank receives exactly S x K tokens (S = input tokens per rank), with the backward pass reducing replicas' gradients back to the home rank (mathematically equivalent to no redundancy); (2) a GPU kernel solves redundant-expert placement + token assignment online for every micro-batch (one launch, rank 0 plans), with no per-layer host involvement and a near-optimal plan; (3) permute/unpermute is fused into dispatch/combine with a static, expert-grouped buffer of S x K rows plus alignment padding instead of a dynamic S x K x R shape, removing per-layer host-device syncs and intermediate copies.
**Why:**
- Each rank receives S x K tokens, not S x K / EP ([README]; [K3] §5.2.1).
- The buffer is NvS = align(S·K + (P−1)·2·E/R, P), i.e. S x K plus per-group padding ([API]).
- One cooperative launch, rank 0 plans and multicasts ([PLAN]); host still allocates buffers once at construction, so "no host involvement" holds per layer ([API]).
- K3 calls the plan near-optimal against offline ILP references ([K3] §5.2.1 "Online planning"); cost moved to Impact.

## Impact / Measured Results
**Before:** Removes the EP dispatch long tail and makes GroupedGEMM a fully static load, with no offline EPLB statistics; online planning overhead is about zero and routing drift is tracked batch by batch; static shapes give lower memory and copy overhead, are CUDA Graph friendly, and have no synchronization bubbles.
**After:** Every rank gets the same S x K tokens, removing the EP dispatch long tail without offline EPLB statistics; routing drift is tracked batch by batch. Per-rank GroupedGEMM totals are static, but per-expert sizes still vary. Planning is negligible (README), though it triggers replica-weight prefetch that adds to dispatch time; static shapes remove per-layer host syncs and the S x K x R worst-case buffer. README chart (H20, EP8): combine stays flat as imbalance grows, 1,954 vs 2,468 µs for DeepEP v2 at maxvio 20.
**Why:**
- "Fully static load" overstated: per-expert `cu_seqlens` change every micro-batch, and K3 adds a workload-aware expert-GEMM scheduler for the skew ([K3] §5.2.1).
- "About zero" overstated: the README says "negligible" for planning, but the weight prefetch it triggers is a visible segment of the README dispatch chart ([CHART-COMM]; estimated 0.5-0.75 ms, so no number given).
- "CUDA Graph friendly" is claimed by neither [README] nor [K3]; dropped. "No synchronization bubbles" narrowed to K3's "removes per-layer host sync".
- Added the one published comparison: combine fwd labels 1,954 vs 2,468 µs at maxvio 20 ([CHART-COMM]; H20, EP8 per [README]).

## Other properties
none
