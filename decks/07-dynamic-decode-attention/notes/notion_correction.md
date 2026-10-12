# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Splits all requests into uniform KV tiles, builds a task graph before each decode step and greedily bin-packs it across CTAs; the attention kernel consumes the task map directly, with a combine kernel merging split-k results.
**After:** Splits all requests into uniform 64-token KV tiles and, once per decode step, builds a flat task map: every persistent CTA gets the same tile budget, filled in (KV head, request) order with overflow spilling into the next CTA; the attention kernel consumes the map directly and a combine kernel merges split-k results.
**Why:**
- The source builds a flat task map of per-CTA bins of task descriptors, not a graph (assign_task.cu [A]; sched_task_info.h [T]) (sources.md discrepancy 2).
- The packing is a deterministic sequential fill with one tile budget per CTA, in (KV head, request) order, spilling a request's remaining tiles into the next bin, not a sort-based greedy bin-packing ([A]: kTileN = 64, num_tile_per_cta formula, loop order) (discrepancy 2).
- Map built once per decode step and reused across layers (LMSYS [L] "Scheduling around live KV work"); persistent CTAs from [K].

## Impact / Measured Results
**Before:** Up to 2.88x over static split-k scheduling (long context and mixed-length batches), sharply cutting CTA tail latency.
**After:** Kernel latency up to 2.95x lower than static split-k on 1x128K + 31x4K (FP8 KV, H20; the README says up to 2.88x); end to end on H200 the attention-only gain is small (+2.0% / +3.0% output throughput).
**Why:**
- The newer LMSYS Table 1 reports 2.95x (0.186 to 0.063 ms) on 1x128K + 31x4K; the README's 2.88x is kept as a pointer (discrepancy 1).
- "Sharply cutting CTA tail latency" is qualitative; the measured effect is kernel latency (Table 1), and end to end on H200 dynamic vs static gives only +2.0% (Hy3-FP8) / +3.0% (Qwen3 BF16) output throughput (Table 6). The large H20 TPOT gains (Table 7) are Attention + MoE together, so they are not credited here (discrepancy 3).

## Other properties
none
