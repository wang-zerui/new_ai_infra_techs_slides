# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Fully-sharded data parallel and hybrid sharding (--num-distributed-optimizer-instances), composable with EP since v0.15; complements the distributed optimizer (ZeRO-1 equivalent).
**After:** Fully-sharded data parallel and hybrid sharding (--num-distributed-optimizer-instances sets the outer DP size; --outer-dp-sharding-strategy picks HSDP or HFSDP), composable with EP since v0.15; complements the distributed optimizer (ZeRO-1 equivalent).
**Why:**
- The flag alone only sets the DP-outer size; with Megatron-FSDP the hybrid mode is chosen by `--outer-dp-sharding-strategy` (`no_shard` = HSDP, `optim` = HFSDP) ([FSDP] "Data-Parallel Sharding Strategies", "Understanding Hybrid-FSDP"; [ARGS] help texts).
- Rest kept: FSDP with EP in v0.15, HSDP for MoE in v0.13 ([RD] "What's New"); distributed optimizer is the ZeRO-1 point and is enabled automatically by Megatron-FSDP ([RD] feature list; [FSDP] feature table).

## Impact / Measured Results
**Before:** Model-state sharding and DP become interchangeable, simplifying memory planning for very large MoEs.
**After:** Per-GPU memory and collective volume scale with the expert-DP size, and FSDP + EP needs no PP stage balancing; no first-party benchmark (user report, Qwen3-30B-A3B on 32 H800: 23.5 vs 45.6 GB, 108 vs 138 TFLOPS against EP8 alone).
**Why:**
- "Interchangeable" is not a source claim; replaced by the source statements "Per-GPU memory and collective volume both scale with EDP size" and "PP-free flexibility" ([MC] §4.1.7 p.27-29).
- No first-party throughput benchmark exists ([BR] card marks performance claims `doc_only`); the only end-to-end measurement is a public user report, labeled as such, showing the memory gain costs throughput ([I2772] comment of 2026-01-14: 23,548 vs 45,552 MB max reserved, 108.0 vs 137.8 TFLOPS).

## Other properties
none
