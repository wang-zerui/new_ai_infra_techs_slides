# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Under decode plus CUDA Graph the CPU cannot know per-expert token counts, so a mask tensor tells the kernel to compute only the valid parts - and it consumes DeepEP low-latency kernel output directly.
**After:** Under decode plus CUDA Graph the CPU cannot know per-expert token counts, so a device-side mask (masked_m) tells the kernel to compute only the valid rows of each expert's fixed-capacity slot; it consumes DeepEP V1 low-latency dispatch output directly (DeepEP V2.5 decode instead uses the contiguous kernel's psum layout).
**Why:**
- Mask semantics from the API and scheduler: `masked_m[g]` on device, A shaped per expert at `max_m` capacity, M tiles computed from device memory ([OLD-API] l.114-136, [SCHED] l.188-204).
- The direct pairing is with DeepEP V1's `low_latency_dispatch` ([EP1-BUF]); V2.5 removed the V1 API and its decode path emits an expanded layout with GPU-side prefix sums, consumed by the contiguous kernel's psum layout ([EP2] l.19, l.163; [TEST] l.203-205).

## Impact / Measured Results
**Before:** Removes the per-MoE-layer host sync, a prerequisite for enabling CUDA Graphs in EP inference and rollout.
**After:** Together with DeepEP low-latency dispatch, removes the per-MoE-layer host sync, a prerequisite for enabling CUDA Graphs in EP decoding (inference and rollout).
**Why:**
- The sync is removed by the pair: normal dispatch still blocks the CPU and is not graph-compatible; the GEMM alone only removes the need for host-side shapes ([EP1] l.173, l.265, l.281; [LMSYS] l.163-188).
- Scoped to decoding, the only use the sources describe ([RD] l.82-86); rollout kept as decoding, with no number claimed (no source measures it).

## Other properties
none
