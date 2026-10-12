# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Decouples the parallel mappings of attention and MoE layers so EP can be "folded" inside TP x CP groups, resolving the dense-sparse parallelism mismatch.
**After:** Decouples the parallel mappings of attention (TP x CP x DP) and MoE layers (ETP x EP x EDP), sharing only PP, so EP can be "folded" across any of TP, CP and DP, resolving the dense-sparse parallelism mismatch.
**Why:**
- EP folds across sub-groups that include DP, not only TP x CP: the 256-GPU example folds EP64 across TP4 x CP2 x DP8 ([MC] §3.3.3 p.17-18), and [PF] App. 6.1 validates EP folded "with all of TP, CP, DP".
- The two mappings and the single PP constraint added from [MC] §3.3.1 p.16 and [PF] §3.2 p.5; code asserts identical PP groups ([CODE-PS]).

## Impact / Measured Results
**Before:** Breaks the EP \<= DP limit: in a 256-GPU example EP rises from at most 8 to 64 (ETP=1, EDP=1).
**After:** unchanged
**Why:** matches [MC] §3.3.3 item 1 (p.17-18) and [RD] lines 340-342 exactly.

## Other properties
none. Left for the user: the Reference (arXiv 2603.07685) does describe folding, but the mechanism, dispatcher and all measured MFU results come from the companion paper arXiv 2504.14960 ([MC] p.16 defers to it).
