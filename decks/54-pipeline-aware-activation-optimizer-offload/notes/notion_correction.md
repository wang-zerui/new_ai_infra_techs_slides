# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** --fine-grained-activation-offloading offloads activations to CPU per pipeline stage and --offload-optimizer-states offloads optimizer state; K2/K3 likewise combine CPU activation offload with selective recomputation to hold utilization.
**After:** --fine-grained-activation-offloading copies chosen modules' saved inputs to CPU on a side stream after forward and reloads them one layer early in backward, ordered by the 1F1B/VPP schedule (later PP ranks offload less); --offload-optimizer-states (dev branch, now --chunked-optimizer-state-offload) keeps optimizer states and master weights on CPU between steps; K2/K3 combine offload with recomputation and FP8 storage.
**Why:**
- Offload granularity is per module (each listed submodule's saved input), with one handler per micro-batch x VPP chunk; PP awareness is the ordering and the per-rank amount, not "per pipeline stage" ([MC] §4.1.5 p.24-25; [DOC]; [FGO]).
- `--offload-optimizer-states` is in the paper (Table 13) and on the dev branch as a deprecated alias of `--chunked-optimizer-state-offload`; it keeps exp_avg, exp_avg_sq and master weights on CPU between steps while the step stays on GPU ([MC] p.26; [ARG] dev; [OCFG] dev). Main's `--optimizer-cpu-offload` is a different mechanism (CPU step).
- K2 and K3 also compress activations (K2 FP8 storage; K3 block-wise FP8 and remote offload to other PP ranks) ([K2] §2.4.3; [K3] §5.2.2). "To hold utilization" is not stated as such; dropped.

## Impact / Measured Results
**Before:** Sustains stable trillion-scale MoE training on H800 clusters with 2TB of host memory per node.
**After:** Megatron-Core: 10-18% less memory for 1.6-2% throughput, or 15% more Qwen3-235B throughput via TP1/EP64; optimizer offload frees 15-20 GB per GPU; Kimi K2 offloads all remaining activations (no measurement).
**Why:**
- The H800 / 2 TB host RAM facts are Kimi K2's cluster ([K2] §2.4.1), but K2 does not attribute training stability to offload and reports no offload measurement ([K2] §2.4.3).
- Measured results come from Megatron-Core: 10-18% memory for 1.6-2% throughput ([MC] §4.1.5 p.25); Qwen3-235B 800 to 920 TF/s, +15% ([MC] Table 5); optimizer-state offload 15-20 GB per GPU on DeepSeek-V3 ([MC] §4.1.6 p.26-27).

## Other properties
none
