# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Experts are split into waves and compute starts the moment a wave's dispatch lands, so in steady state the current wave's compute, the next wave's transfer and the return of finished results all run concurrently, fused into one pipelined kernel.
**After:** unchanged
**Why:**
- Matches V4 §3.1 p.15 (waves, compute starts when a wave's communication completes, three-way steady-state concurrency, single pipelined kernel).

## Impact / Measured Results
**Before:** Fully hides MoE communication whenever per-layer comm time < compute time; derives the hardware balance point C/B <= 2d (about 6144 FLOPs/Byte), so 1 GBps of interconnect covers 6.1 TFLOP/s; up to 1.96x on RL rollout.
**After:** Hides MoE communication in steady state when per-layer comm time is below compute time (pipeline fill and drain stay exposed); balance point C/B at most 2d = 6144 FLOPs/Byte for V4-Pro (d = expert width 3072), so each GB/s of interconnect covers 6.1 TFLOP/s; MoE-layer speedup over non-fused kernels 1.50-1.73x, up to 1.96x (V4-Flash, 1 token per rank, EP8).
**Why:**
- "Fully" overstates: the first dispatch and last combine of the pipeline stay exposed, so the hiding holds in steady state (V4 §3.1 "Communication Latency Can Be Hidden"; Fig. 5) (sources.md discrepancy 1).
- 6144 is exact and V4-Pro only, with d = expert intermediate dimension 3072, not the hidden size (V4 §3.1 p.16; §4.2.1) (discrepancy 2).
- 1.96x is a MoE-layer kernel benchmark vs the non-fused DeepEP + DeepGEMM path (V4-Flash, 1 token per rank, EP8; PR316), quoted by V4 for latency-sensitive cases such as RL rollout, not an end-to-end RL speedup; general inference gets 1.50-1.73x (V4 §3.1 p.16) (discrepancies 3, 4).

## Other properties
none. Not changed (page body, out of scope): the body lists Flux (arXiv 2406.06858) as a paper, while V4 cites Comet and FlashDMoE as inspirations (sources.md discrepancy 5).
