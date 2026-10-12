# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Fuses the post-TP AllReduce, residual add and RMSNorm into an NVLink-native kernel: a CUDA multicast high-throughput mode for many tokens and a Lamport P2P dual-kernel mode with PDL overlap for few tokens, both on two-shot communication schedules.
**After:** Fuses the post-TP AllReduce, residual add and RMSNorm into one NVLink-native op: a single-kernel CUDA multicast high-throughput mode for many tokens and a two-kernel Lamport low-latency mode (P2P scatter, multicast broadcast, chained by PDL) for few tokens, both on two-shot schedules.
**Why:**
- "An NVLink-native kernel" is wrong for LL: HT is one kernel, LL is two kernels chained by PDL ([PR48]; [LL]) (sources.md discrepancy 3).
- LL is not purely P2P: only the scatter (reduce) phase uses P2P stores; the broadcast stores through the multicast address ([LL] twoshotAllreduceKernel) (discrepancy 2).

## Impact / Measured Results
**Before:** Up to 1.76x over NCCL / FlashInfer for single-node BF16 TP shapes at hidden 4096/5120/7168.
**After:** unchanged
**Why:**
- Matches the HPC-Ops README Performance table ([RD]; metric definition [BR]: best HPC-Ops mode vs best of NCCL / FlashInfer, 8xH20, BF16).

## Other properties
none. Not changed (page body, out of scope): the body cites the LMSYS HPC-Ops x SGLang blog as "vs NCCL / FlashInfer", but that blog has no AllReduce + RMSNorm result; the 1.76x is from the README (sources.md discrepancy 1).
