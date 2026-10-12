# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Main pretraining on short sequences (K2: 4k->32k->128k with YaRN; K3: 8k->64k), stretched to 256k->1M only during cooldown / mid-training, with constant LR plus cosine decay and final annealing.
**After:** Main pretraining on short sequences, long windows only in a low-LR tail: K2 trains 4K then 32K and reaches 128K with YaRN (WSD schedule, then anneal); GLM-4.5 goes 4K to 32K to 128K in mid-training (cosine); K3 trains 8K to 64K, then 256K to 1M in cooldown (cosine).
**Why:**
- K2 trains only 4K and 32K; 128K is a YaRN extension on top of the 32K stage (K2 §2.5) (sources.md discrepancy 3).
- "Constant LR plus cosine decay and final annealing" is K2's WSD schedule only (K2 §2.5); K3 uses cosine with 1% warm-up (K3 §3.3) and GLM-4.5 uses cosine, reporting WSD underfits (GLM §2.4) (discrepancy 1).
- "Cooldown" is K3's term; GLM-4.5's long stages are "mid-training" within one cosine schedule (GLM Fig. 3, §2.3-2.4; K3 §3.3-3.4) (discrepancy 5).

## Impact / Measured Results
**Before:** Concentrates expensive long-context compute on a small tail of tokens; this is how K3 gets its 1M context.
**After:** Puts long-context compute on a small tail of tokens (K2: 60B at 32K; GLM-4.5: 100B at 128K; K3 publishes no budgets); no report ablates staging, and K3's 1M also relies on NoPE + KDA.
**Why:**
- The quantified tails come from K2 (60B at 32K, §2.5) and GLM-4.5 (100B at 128K, Fig. 3); K3 says only "small fraction of the overall training budget" (§3.4) (discrepancy 2).
- None of the three reports ablates staged vs. unstaged training, so there is no measured result (discrepancy 4).
- K3's 1M context also rests on NoPE + KDA, with no positional rescaling (K3 §2.1.2, §3.4), not on staging alone (discrepancy 2).

## Other properties
none
