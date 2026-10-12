# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** --overlap-moe-expert-parallel-comm with --delay-wgrad-compute hides EP all-to-all inside 1F1B pipeline slots and batch-level splits; DP/TP/PP communication overlap is supported as well.
**After:** --overlap-moe-expert-parallel-comm with --delay-wgrad-compute merges the forward of one micro-batch with the backward of another in the 1F1B steady state, so each one's EP all-to-all hides behind the other's compute (two CUDA streams, weight gradients deferred); DP/TP/PP communication overlap is supported as well.
**Why:**
- The all-to-all is hidden behind the compute of an adjacent micro-batch in merged FWD-BWD, not in pipeline slots or bubbles; the first forward and last backward stay exposed (MC p.33; CODE-PLAN run docstrings; CODE-1F1B) (sources.md discrepancy 1).
- --delay-wgrad-compute splits the backward MLP into data and weight gradients and defers the latter (MC p.33; CODE-CFG docstring).

## Impact / Measured Results
**Before:** Buries EP communication in pipeline bubbles - the default optimization for large-EP training.
**After:** DeepSeek-V3 on H100 (EP64): EP all-to-all drops from 30-40% of iteration time after DeepEP to under 5%; off by default, and left off when EP fits in NVLink (DeepSeek-V3 on GB200).
**Why:**
- Not "pipeline bubbles": see Core Mechanism (discrepancy 1).
- Measured result: 30-40% remaining after DeepEP, under 5% with 1F1B FWD-BWD overlap + W/D split, DeepSeek-V3 EP64 on H100 (MC §4.2.3 p.32-34).
- Not "the default": both flags default to False (RD lines 563-564), and the best DeepSeek-V3 GB200 config (EP64 inside NVL72) does not use it (MC Table 17; App. B.2) (discrepancy 2).

## Other properties
none
