# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** FP8 grouped GEMM for EP and grouped-expert matmuls, accepting cu_num_tokens input with unequal per-group token counts and both quantization granularities.
**After:** unchanged
**Why:**
- No real error: the API arguments are `seqlens` / `cu_seqlens` ([API]), but the README Basic Usage example names the same tensor `cu_num_tokens_per_group` ([RD]); per-tensor and block-wise FP8 paths both exist ([API], [PT], [BW]).

## Impact / Measured Results
**Before:** vs DeepGEMM: 1.1x on prefill and 1.88x on decode (H20).
**After:** vs DeepGEMM on H20: up to 1.1x on prefill and 1.88x on decode (maximum observed speedups as published in the HPC-Ops README).
**Why:**
- The README states both figures as maxima ("Up to" in the current Performance table, "Maximum observed speedup per operator" in the January 2026 version); no per-shape data is published ([RD], [RD-JAN]).

## Other properties
none (the page body's LMSYS link note "prefill/decode comparison vs DeepGEMM" is inaccurate, but the body is out of scope)
