# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Buffers samples and packs them to a target sequence length instead of uniform padding; the MCore counterpart is the THD packed format (e.g. GLM-5.2's 128K THD packed CP training).
**After:** Buffers samples and packs each micro-batch to a token budget instead of uniform padding, passing cu_seqlens so attention stays per sample; the MCore counterpart is the THD packed format; GLM-5 adds sequence reordering and variable-size CP groups for long sequences.
**Why:**
- The packing target is a token budget per micro-batch ([VO] §3.2.5; [V-BATCH] `get_samples`; [V-PACK]); boundaries are passed to varlen attention as `cu_seq_lens` ([V-PACK], [V-COLL]).
- THD is Megatron-Core's packed format ([MC] §6.4.1); kept.
- The GLM-5 report mentions no THD format, no "GLM-5.2" and no 128K THD packed CP run; it describes workload-aware sequence reordering and flexible CP groups of varying sizes for long sequences ([GLM5] §2.4.2). Replaced the example with what the report says.

## Impact / Measured Results
**Before:** Large gains in effective FLOP utilization for multimodal and variable-length data.
**After:** Megatron-Core reports 1.5-2x throughput and 40-60% less memory vs padding (RL thinking models); no VeOmni measurement isolates packing.
**Why:**
- The only quantified packing gains in the sources are Megatron-Core's, for RL with thinking models ([MC] §6.4.1 p.60); VeOmni publishes no packing-only measurement ([VO]).

## Other properties
none (Source keeps "GLM-5 series": the report does cover variable-length long-sequence CP scheduling, though not THD packing; left for the user to decide)
