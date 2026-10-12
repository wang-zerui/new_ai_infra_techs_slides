# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Unifies the high-throughput and low-latency dispatch/combine APIs into ElasticBuffer (with a new GEMM layout) and derives the optimal SM and QP counts analytically from the MoE configuration, removing auto-tuning; handle caching reuses routing metadata.
**After:** Unifies the high-throughput and low-latency dispatch/combine APIs into one buffer (ElasticBuffer in V2.0, EPBuffer in V2.5) with an expanded layout for grouped expert GEMMs, and computes SM and QP counts from an analytic model of the MoE configuration and link bandwidths instead of tuned per-EP-size tables; a cached handle replays the same routing (e.g., backward), not new routing.
**Why:**
- ElasticBuffer was split into EPBuffer / EngramBuffer / PPBuffer / BucketBuffer in V2.5 ([PR763]).
- The README's name for the layout is the "expanded layout for grouped expert GEMMs" (`do_expand`, `expert_alignment`) ([R25]).
- "Optimal" overstated: the model is a balanced-gate bandwidth estimate with +30% headroom and rounding ([EP25] l.431-560); V1 used a per-EP-size config table plus cluster tuning ([V1code], [R1]).
- V2.5 narrows handle reuse: "A new routing decision needs a fresh dispatch; decoding alone does not make an old routing handle reusable" ([R25]).

## Impact / Measured Results
**Before:** Communication SMs drop from 24 to 4-6 at equal or better performance in V3-style training; 726/740 GB/s on SM100 NVLink EP8.
**After:** Communication SMs drop from 24 to 4-6 at equal or better performance in V3-style training; SM100 NVLink EP8 reaches 726/740 GB/s (dispatch/combine) at 64 SMs, 643/675 GB/s at 24 SMs.
**Why:**
- 726/740 GB/s is the 64-SM "max perf" row; the minimum-SM row is 643/675 GB/s at 24 SMs ([R2] Performance table rows 4-5).

## Other properties
none
