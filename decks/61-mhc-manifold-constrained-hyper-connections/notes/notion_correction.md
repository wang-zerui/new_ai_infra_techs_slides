# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Projects Hyper-Connections' residual mixing matrices onto the doubly-stochastic manifold (Birkhoff polytope) with Sinkhorn-Knopp, keeping spectral norm \<= 1 and restoring identity-mapping behavior; companion kernel optimizations cut the extra memory traffic.
**After:** unchanged
**Why:** all statements match the sources: H_res projected by Sinkhorn-Knopp onto the Birkhoff polytope, spectral norm at most 1, identity-mapping property restored ([mHC] Abstract, §4.1-4.2; [V4] §2.2); fused kernels cut residual I/O ([mHC] §4.3.1).

## Impact / Measured Results
**Before:** Eliminates HC's gradient explosion at 27B (compound gain at depth 64 goes from 1e16 to about 1); a single Sinkhorn iteration suffices; also adopted by GLM-5.3.
**After:** Eliminates HC's gradient explosion at 27B: peak composite residual gain drops from about 3000 (HC) to a maximum of about 1.6; Sinkhorn-Knopp uses 20 iterations (t_max, mHC and V4); about 6.7% extra training time at n = 4; adopted in DeepSeek-V4.
**Why:**
- No 1e16 value or depth-64 measurement exists; the 27B model (60 sublayers) shows HC peaking at about 3000 and mHC at a maximum of about 1.6 ([mHC] §3.1 Fig. 3(b), §5.4 Fig. 7(b)).
- "A single Sinkhorn iteration suffices" is contradicted: both papers use t_max = 20 ([mHC] §4.2 p.9; [V4] §2.2, §4.2.1); mHC-lite argues for replacing Sinkhorn, not one iteration ([lite] abstract).
- "Also adopted by GLM-5.3" dropped: the GLM-5 report never mentions mHC and no GLM-5.3 source was found; V4 adoption is documented ([V4] §2.2, §4.2.1).
- 6.7% overhead added as the published cost ([mHC] §1 p.4; [V4] §3.4.2).
- "Eliminates" kept: consistent with the HC loss surge near step 12k and stable mHC curves ([mHC] Fig. 2, Fig. 5).

## Other properties
none. Left for the user: Source still lists "GLM-5 series", but no source read shows GLM adopting mHC (GLM-5 report arXiv 2602.15763 has no mention; no GLM-5.3 source available). Not changed because absence in the sources read is not proof the value is wrong.
