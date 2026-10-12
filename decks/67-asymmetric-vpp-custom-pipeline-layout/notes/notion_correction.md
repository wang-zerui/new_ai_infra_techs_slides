# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** --pipeline-model-parallel-layout allows non-uniform assignment of layers, embeddings and MTP heads across PP stages (flexible asymmetric virtual pipelining).
**After:** unchanged
**Why:** matches the sources: the layout string places E, t, m and L per stage, stages split over PP x VPP ([DOC]; [CODE-L] `parse_str_to_list`, `__init__`; [RD] v0.13; [MC] §7.5).

## Impact / Measured Results
**Before:** Fixes the PP load imbalance caused by MTP/embedding layers and shrinks pipeline bubbles.
**After:** Lets users hand-balance stages around embedding, dense, MTP and loss layers (used for DeepSeek-V3 on H100 and GB200); no measured bubble or throughput gain from the layout itself is published.
**Why:**
- "Fixes" implies automatic balancing; the source describes manual, cost-aware placement written by the user (e.g. "Embed + 3 Dense about 2 MoE" on the first stage) ([MC] §7.5 p.66, Fig. 41; [DOC]).
- "Shrinks pipeline bubbles" has no measurement: the DeepSeek-V3 layouts appear only inside full-stack configurations ([MC] Table 17, §9.2.2 p.74-75); the deck's bubble numbers are analytic PTD-P estimates ([PTDP] §2.2.2).

## Other properties
none
