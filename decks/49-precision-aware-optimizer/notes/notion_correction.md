# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** --use-precision-aware-optimizer lets TE optimizers keep first/second moments and master weights in BF16/FP8, mixing precision as needed.
**After:** --use-precision-aware-optimizer lets TE FusedAdam store first/second moments in BF16 or FP8 (master weights FP32 or FP16) and upcast them to FP32 inside the kernel for the update; with BF16 params the FP32 master can be kept as a 16-bit remainder, still exact FP32.
**Why:**
- Master weights cannot be BF16 or FP8: `--main-params-dtype` accepts only fp32 or fp16 and FusedAdam supports FP32/FP16 masters; only the moments accept bf16/fp8 ([ARGS], [TEA]).
- Storage precision is decoupled from compute: states are cast to FP32 inside FusedAdam and the update runs in FP32 ([MC] §4.1.6; [ARGS] help text).
- The remainder option stores the 16 low bits of the FP32 master next to the BF16 weight, which reconstructs exact FP32 ([OC] `store_param_remainders`, [TEK]).

## Impact / Measured Results
**Before:** Large reduction in optimizer-state memory; standard for trillion-parameter MoEs under limited HBM.
**After:** BF16 moments cut optimizer-state memory about 50%: 10-12 GB of DeepSeek-V3's 32.1 GB of master weights and states per GPU, no performance overhead; used in DeepSeek-V3 training and MCore's recipes.
**Why:**
- "Large reduction" quantified: about 50% of optimizer-state memory, 10-12 GB of 32.1 GB for DeepSeek-V3 on 256 GPUs, with "no performance overhead" ([MC] §4.1.6 p.26-27, Table 3).
- No trillion-parameter run is documented in the sources read; the documented uses are MCore's DeepSeek-V3 (685B) final configs ([MC] Table 18) and DeepSeek-V3's own training with BF16 moments ([DSV3] §3.3.3).

## Other properties
none
