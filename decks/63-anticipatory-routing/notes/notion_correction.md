# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Deliberately desynchronizes backbone and router updates: features use the newest parameters while routing decisions use cached older ones, breaking the "anomaly -\> routing drift -\> larger anomaly" feedback loop.
**After:** Features at step t use the current parameters, but routing indices come from the network's parameters Δt steps earlier: step t's data is fetched in advance and its indices computed and cached in one extra forward pass; meant to break the vicious cycle in which routing appears to amplify MoE outliers.
**Why:**
- What is old is the whole network θ_t−Δt (backbone layers feeding the router as well as the router), and what is cached is the routing indices, not router parameters ([V4] §4.2.3 p.26).
- The "anomaly to routing drift to larger anomaly" loop is Notion's elaboration; [V4] only says spikes are tied to MoE outliers, routing "appears to exacerbate" them, and the technique breaks "the vicious cycle induced by routing", with the principles not well understood ([V4] §4.2.3, §6).

## Impact / Measured Results
**Before:** With mHC and Muon, suppresses loss spikes and divergence in trillion-scale MoEs.
**After:** Turned on only after a detected loss spike (with a short rollback), at about 20% extra wall-clock while active; with SwiGLU clamping, averted V4's loss spikes without hurting performance.
**Why:**
- Companion technique in [V4] §4.2.3 is SwiGLU clamping; mHC and Muon are stability features elsewhere in the paper, not part of the spike mitigation ([V4] §2.2, abstract, §4.2.3).
- No divergence results are reported; "divergence" dropped.
- Spike-triggered operation and the 20% cost added: without them the row implies an always-on, free mechanism ([V4] §4.2.3 p.26: about 20% while active, "negligible" overall).

## Other properties
none
