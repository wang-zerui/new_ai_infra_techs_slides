# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Beyond EP, experimental primitives: PP and Engram (remote memory access) go over RDMA with 0 SMs, CP uses the Copy Engine with 0 SMs; DP/TP all-gather and reduce-scatter implementations are included.
**After:** Beyond EP, experimental primitives that keep SMs off the payload path (SMs at most stage data or post work requests): PP send/recv and Engram remote reads go over RDMA, the CP all-gather over NVLink copy engines (V2.5 adds RDMA and hybrid paths); V2.5 adds CP/DP reduce-scatter and all-reduce, which still use SMs.
**Why:**
- "0 SMs" holds for the payload, not for kernels: PP launches one-warp-per-CTA TMA staging kernels and Engram/RDMA all-gather launch warps that post puts/gets; only the pure-NVLink all-gather runs no kernel ([C-PP], [C-EN25], [C-AG25]).
- Copy Engine is the NVLink path only; V2.5 adds RDMA (NIC) and hybrid paths ([C-AG25]).
- DP/TP all-gather/reduce-scatter were "on-going" at V2 ([R2] l.37); V2.5 BucketBuffer covers CP and DP, not TP, and "reduce-scatter and all-reduce use GPU SMs" ([R25] l.14, l.35, l.293, l.328).

## Impact / Measured Results
**Before:** Communication no longer competes with compute for SMs - the key direction toward "free" communication (currently experimental).
**After:** PP, CP all-gather and Engram transfers no longer occupy SMs; EP dispatch/combine still do (4-6 SMs in V2 vs 24 in V1 for V3-like training). Still experimental; no bandwidth or end-to-end numbers are published for these primitives.
**Why:**
- "Communication" overall still uses SMs: EP dispatch/combine require them ([R25] l.39 "zero-SM RDMA EP is not supported"; [R2] l.22 for 24 to 4-6).
- Dropped the unsupported "key direction toward free communication" framing; no source measures these primitives' bandwidth or end-to-end effect (discrepancy item 4).

## Other properties
none
