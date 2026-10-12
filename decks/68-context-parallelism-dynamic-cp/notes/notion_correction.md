# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Shards inputs and all activations along the sequence dimension (unlike SP, which only splits LayerNorm/Dropout), exchanging KV via ring/p2p; supports MHA/MQA/GQA, MLA and MTP, composes with TP/PP/DP, and now supports dynamic CP.
**After:** Shards inputs and all activations along the sequence dimension (unlike SP, which only splits LayerNorm/Dropout), exchanging KV via ring/p2p; supports MHA/MQA/GQA, MLA and MTP, composes with TP/PP/DP; Dynamic CP picks a power-of-2 CP size per sequence, not yet with MLA.
**Why:**
- Dynamic (hybrid) CP is not yet supported with MLA ([MLA] assert in `multi_latent_attention.py`), so "supports ... MLA ... and now supports dynamic CP" read together overstates; MLA/MTP support itself is for static CP since v0.14 ([RD]).
- Dynamic CP mechanism added in a few words: CP size = next power of 2 of seq_len / max_seqlen_per_rank, from pre-built power-of-2 groups ([HCP]; [MC] §6.4.2 p.62; [BLOG]).
- Rest kept: [CPDOC] overview (sequence sharding, KV ring exchange, SP comparison, TP x CP x PP x DP).

## Impact / Measured Results
**Before:** Beats full recomputation beyond 8K sequences by removing most recompute cost while balancing compute and communication.
**After:** Beats full recomputation at 16K and 32K on a 175B GPT (CP configs about 1.15x to 1.3x, read from the CP guide's chart) by avoiding the about 30% recompute overhead while keeping TP small enough to hide its communication.
**Why:**
- The only measurement covers 16K and 32K ([CPDOC] Fig. 2, 175B GPT); 8K is an illustrative example in the guide and the README's recommended threshold for enabling CP (Guideline 5), not a measured crossover.
- Speedups are approximate values read from the bar chart (TP8CP2 / TP16CP2 / TP8CP4: about 1.15 to 1.30), since the guide prints no numbers.
- "Removing most recompute cost" made concrete with the guide's own figure (about 30% overhead of full recompute) and its TP argument ([CPDOC] "Context Parallelism Benefits").

## Other properties
none
