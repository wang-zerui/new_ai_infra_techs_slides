# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Unlike CUTLASS generic grouped GEMM, only the M axis is grouped while N/K stay fixed (MoE experts share a shape); per-expert tokens are concatenated into one tensor with each segment aligned to the GEMM M block.
**After:** unchanged
**Why:**
- Matches the README ([RD] l.76-80). The CUTLASS comparison is the README's own framing; the measured baseline was DeepSeek's internal CUTLASS 3.6-based implementation ([RD-0325] l.10), but the row states no number against it, so it does not mislead.

## Impact / Measured Results
**Before:** Fits training forward and prefill, where per-expert token counts vary, without padding waste.
**After:** Fits training forward and prefill, where per-expert token counts vary; each expert segment is padded only up to the M alignment (128 rows by default), not to a fixed capacity.
**Why:**
- The layout does pad: every expert segment is rounded up to the M alignment (128 by default, 256 theoretical on SM100) with -1 / zero rows ([RD] l.78, [HEUR] l.12, l.57-64, [GEN] l.396-406). What it avoids is capacity-factor padding of every expert to a fixed size ([MB] §3 p.4).

## Other properties
none
