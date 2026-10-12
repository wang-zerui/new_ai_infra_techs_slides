# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** E2M1 with two-dimensional block scaling; stochastic rounding removes gradient-quantization bias, Random Hadamard transforms spread outliers, and sensitive layers stay high precision (bf16_with_nvfp4_mixed).
**After:** E2M1 values with an E4M3 scale per 16-value block plus an FP32 tensor scale; 16x16 2D scaling for weights only; stochastic rounding on gradients removes quantization bias; 16x16 random Hadamard transforms on Wgrad inputs only spread outliers; about 15% of linear layers, mostly the last blocks, stay BF16 (12B: first 2 and last 8 of 62 blocks).
**Why:**
- The NVFP4 format is 1x16 E4M3 block scales plus an FP32 tensor scale; 2D (16x16) scaling is a recipe choice for weights only ([NV4] Sec. 2, Sec. 4.3; [TE-nv4]).
- RHT is applied only to Wgrad inputs; Fprop/Dgrad transforms gave no benefit or hurt ([NV4] Sec. 4.2, App. E.4.1).
- SR is for gradients; SR on forward tensors is detrimental ([NV4] Sec. 4.4, App. E.3).
- `bf16_with_nvfp4_mixed` only switches GEMMs to NVFP4; which layers stay BF16 is set separately ([MB]). Replaced with the paper's rule and the 12B configuration ([NV4] Sec. 4, App. A.1).

## Impact / Measured Results
**Before:** <1% loss gap vs FP8 over 10T tokens on a 12B model (~1.5% late in training); a further 1.31-1.73x over FP8 on GB200/GB300.
**After:** Under 1% loss gap vs blockwise FP8 over 10T tokens on a 12B model (slightly above 1.5% after LR decay); NVFP4 GEMMs run 2.03x over BF16 on B300 (5B model), 1.41x over MXFP8 (derived).
**Why:**
- Loss gap kept ([NV4] Sec. 3, Fig. 2); the baseline is the blockwise 1x128 / 128x128 FP8 recipe ([NV4] App. A.1).
- "1.31-1.73x over FP8 on GB200/GB300" appears in none of the sources read; the paper reports no runtime. Replaced with the measured TE GEMM speedup on B300 and the derived 2.03 / 1.44 = 1.41x over MXFP8 ([TE-speed]).

## Other properties
none
