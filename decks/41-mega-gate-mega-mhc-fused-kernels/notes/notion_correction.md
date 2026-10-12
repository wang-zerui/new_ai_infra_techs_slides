# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Fuses router gating (Mega Gate) and mHC / HyperConnection residual mixing (Mega mHC) each into a single kernel, avoiding the swarm of small ops and intermediate tensors these new architectures introduce.
**After:** Fuses router gating (Mega Gate: BF16 gate GEMM, Sqrt(Softplus) scoring and top-k) and mHC residual mixing (Mega mHC: post-mix, 24-wide projection, Sinkhorn and RMSNorm, replacing a 4- to 7-kernel chain) each into one SM100 kernel, removing kernel boundaries and some stream re-reads; the new residual and split-K partials still pass through memory.
**Why:**
- "Avoiding ... intermediate tensors" overstates: Mega mHC still writes the new residual (an output), materialises the norm boundary and, in normal mode, re-reads the new residual (kept in L2 by EVICT_LAST); split-K partials go to a workspace, and Mega Gate passes scores through a workspace when experts or K are split ([MHC-K] lines 342-343, 570-655; [MHC-T] byte model; [GATE-K]).
- What the kernels fuse, from the code: the gate GEMM with sqrtsoftplus scoring, bias, top-k ([GATE-API], [GATE-K], [PR462]); post-mix, the 24-output TF32 projection, Sinkhorn and RMSNorm, replacing the 4-kernel (normal) or 7-kernel (shifted) baseline ([MHC-T] ref_mhc.py NORMAL_KERNELS / SHIFTED_KERNELS, [MHC-API]).
- Both kernels are SM100-only ([MHC-K], [GATE-K], tests filtered on arch 10).

## Impact / Measured Results
**Before:** Brings mHC's extra memory traffic into an acceptable range - a key enabler for shipping new residual structures to production.
**After:** Mega mHC is 45-85% faster than the multi-kernel TileLang mHC path and Mega Gate 30-75% faster than cuBLASLt GEMM + TileLang top-k (DeepGEMM, Sept 2026); mHC's 6.7% training overhead came earlier, from other fused kernels plus recomputation.
**Why:**
- The quantified "acceptable range" (6.7% overhead at n = 4) was reached by the mHC paper's TileLang kernels plus recomputation and DualPipe changes in training ([MHC] §1 p.4, §4.3; [V4] §3.4.2 p.20), before Mega mHC existed; crediting it to Mega mHC is a wrong attribution.
- DeepGEMM's only numbers for these kernels: Mega mHC 45-85% and Mega Gate 30-75% faster than the multi-kernel baselines ([PR432]; baselines from [MHC-T] and [GATE-T]).
- "Key enabler for shipping ... to production" has no source; dropped.

## Other properties
none changed. Source lists "DeepSeek V4": the V4 report describes fused mHC kernels but does not name Mega Gate or Mega mHC (those come from DeepGEMM PR #432); not plainly wrong, so left for the user to decide.
