# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Every four sparse-attention layers reuse the top-k selection of a single indexer, avoiding repeated indexing computation.
**After:** In GLM-5.2 the first of every four DSA layers runs the indexer and the next three reuse its top-k (the first three layers keep their own: 21 indexers in 78 layers); the IndexCache paper also uses searched, non-uniform layouts.
**Why:**
- "Every four" is the GLM-5.2 production layout (G52B; HF52 indexer_types FFF + FSSS x 18 = 21 full of 78), not a rule of the technique (sources.md discrepancy 2).
- The paper's headline training-free results use a greedy-searched, non-uniform pattern (IC §3.1, Table 2, App. B).

## Impact / Measured Results
**Before:** 2.9x lower per-token FLOPs at 1M context; the improved MTP adds up to 20% more accepted length in speculative decoding.
**After:** GLM-5.2 vs GLM-5.1: 2.9x lower per-token FLOPs at 1M (a FLOPs count, not a measured speedup); MTP accepted length up to +20% only together with KVShare, rejection sampling and a TV loss (IndexShare + KVShare alone: +12%, derived).
**Why:**
- 2.9x is a model-level FLOPs comparison, GLM-5.2 vs GLM-5.1, from the Z.ai blog figure (G52R; G52B), not a measured speedup (discrepancy 4).
- The +20% (4.56 to 5.47) stacks IndexShare + KVShare, rejection sampling and an end-to-end TV loss (G52B table, GLM-5.1 backbone, 7 MTP steps); IndexShare + KVShare alone is 4.56 to 5.10 = +12%, derived (discrepancy 3).

## Other properties
none. Title left as is: GLM-5.2's README and blog call it IndexShare; the cited paper (arXiv 2603.12201) calls it IndexCache (sources.md discrepancy 1).
