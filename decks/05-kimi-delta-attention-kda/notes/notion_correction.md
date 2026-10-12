# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** A Gated DeltaNet refinement: the delta rule plus channel-wise forget gates maintain a fixed-size recurrent state; three KDA layers alternate with one gated-MLA layer, fully NoPE (position information comes implicitly from gated decay).
**After:** A Gated DeltaNet refinement: the delta rule plus channel-wise forget gates maintain a fixed-size recurrent state; in K3, three KDA layers alternate with one Gated MLA layer, plus one final MLA layer (69 KDA + 24 MLA), fully NoPE (position comes implicitly from gated decay).
**Why:**
- K3 places one extra Gated MLA layer at the end of the backbone (§2.1), so the stack is 69 KDA + 24 MLA = 93 layers (Table 1), not an exact 3:1 (sources.md discrepancy 2).

## Impact / Measured Results
**Before:** State does not grow with length, so 1M context needs no RoPE interpolation; with AttnRes and Stable LatentMoE it gives roughly 2.5x better overall scaling efficiency than K2.
**After:** 1M context needs no RoPE interpolation because K3 is NoPE, not because the state is fixed-size; the roughly 2.5x scaling-efficiency gain over K2 is joint (KDA, AttnRes, Stable LatentMoE, refined data and training recipes), with no KDA-only ablation.
**Why:**
- K3 §3.4 attributes the no-interpolation property to NoPE (no RoPE to rescale); the fixed state is a memory property, not the reason (discrepancy 3).
- K3 credits the ~2.5x to KDA, AttnRes, Stable LatentMoE and refined data and training recipes together; there is no KDA-only ablation at K3 scale (Abstract; §1; §3.2, Fig. 7) (discrepancy 1).

## Other properties
none
