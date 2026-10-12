# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** --recompute-granularity selective with --recompute-modules selects mla_up_proj / moe / moe_act / layernorm / norm per module, and works under FP8.
**After:** --recompute-granularity selective with --recompute-modules picks per module (layernorm, mla_up_proj, moe_act, mlp, moe, core_attn, shared_experts); the first three discard their outputs and rebuild them before backward, while moe re-runs experts and all-to-alls; FP8 works with TE 2.6+ and no delayed scaling.
**Why:**
- There is no `norm` module; the supported list is `core_attn, moe_act, layernorm, mla_up_proj, mlp, moe, shared_experts` (plus newer ones) ([CFG] `recompute_modules` docstring; [RD]).
- layernorm, mla_up_proj and moe_act use the output-discarding checkpoint (output storage freed, recomputed from a grad hook) ([RNG], [TL], [MLA], [EXP]; [MC] §4.1.4 p.23); `moe` uses standard checkpointing and replays the experts and their all-to-alls ([CFG]).
- FP8 support (MCore v0.14) has constraints: TE 2.6.0dev0 or newer for moe_act/layernorm, delayed scaling rejected ([CFG] `__post_init__`; [RD]).

## Impact / Measured Results
**Before:** Cuts activation memory for negligible extra compute, avoiding the 30%+ overhead of full-layer recomputation.
**After:** Saves 42.4 GB per GPU on DeepSeek-V3 (layernorm, mla_up_proj, moe_act; 256 GPUs) for under 5% extra compute, vs about 33% for full-layer recomputation.
**Why:**
- "Negligible" is quantified in the paper as "less than 5%" ([MC] §4.1.4 p.23); full recomputation is "~33%" (p.22).
- 42.4 GB = 30.4 (MLA up-proj) + 8.2 (LayerNorm) + 3.8 (SwiGLU), DeepSeek-V3, PP4 x VPP4 x EP64 ([MC] Table 4 p.24).

## Other properties
none
