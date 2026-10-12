# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Muon needs full gradient matrices, which conflicts with ZeRO element-wise sharding: V4 uses knapsack-based hybrid ZeRO bucketing (\<10% padding memory), optimizes MoE params separately and syncs their gradients in BF16 with stochastic rounding; GLM-5 uses zero-redundancy communication so every DP rank need not all-gather full params.
**After:** Muon needs full gradient matrices, which conflicts with ZeRO element-wise sharding: V4 packs whole dense matrices into knapsack-balanced hybrid ZeRO buckets (under 10% padding memory), handles MoE params separately and syncs their gradients in BF16 with stochastic rounding; GLM-5 uses zero-redundancy communication so every DP rank need not all-gather full params.
**Why:**
- The under-10% padding figure holds for the dense buckets only ("each rank manages no more than five parameter matrices"); MoE padding is called "negligible" ([V4] §3.4.1 p.19-20). Wording now scopes it to dense matrices.
- Rest unchanged: MoE flattening and BF16 stochastic-rounding sync ([V4] §3.4.1 p.20); GLM-5's zero-redundant all-gather of owned shards ([GLM5] §2.4.1 p.9).

## Impact / Measured Results
**Before:** Halves MoE gradient-sync traffic; removes naive Muon's transient memory spikes and redundant communication.
**After:** V4: BF16 sync halves MoE gradient communication volume; GLM-5 reports, with no numbers, that its zero-redundant all-gather removes naive Muon's transient memory spikes and redundant communication.
**Why:**
- Halving is V4's result and applies to MoE gradients only ([V4] §3.4.1 p.20).
- The memory-spike / redundant-communication claim is GLM-5's statement about its own implementation, with no published number, not a V4 result ([GLM5] §2.4.1 p.9).

## Other properties
none
