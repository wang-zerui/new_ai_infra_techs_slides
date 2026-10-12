# Notion row correction (2026-10-13)

No change: the row has no real error, only omissions.

## Core Mechanism
**Before:** All kernels are JIT-compiled at runtime (no CUDA compilation at install time) through a low-CPU-overhead C++ JIT module with on-disk cache; runtime knobs include set_num_sms / set_tc_util / set_pdl and use_deterministic_algorithms.
**After:** unchanged
**Why:**
- Every statement is supported: runtime-only compilation (S4.4, S1.1), the low-CPU-overhead C++ JIT module (S3.12, PR #112) with an on-disk cache (S1.4), and the four knobs (S3.11). Since 2026-09 the module is the separate DeepJIT project (S3.12, S1.1); the title already names DeepJIT, so not saying so here is harmless.

## Impact / Measured Results
**Before:** Per-shape configuration and SM budget caps let GEMMs coexist with communication kernels; determinism switches serve reproducible training.
**After:** unchanged
**Why:**
- Supported: per-shape compile-time constants and deterministic config selection (S3.2-S3.5, S4.1); `set_num_sms` caps the SM count, a compile-time constant (S3.2, S3.6); `use_deterministic_algorithms` exists for reproducibility (S3.9, S5.1, S6.1).
- Omitted but not wrong: determinism has been opt-in (default False) since the 2026-09 release (S6.1). The row says "switches", which does not claim a default.

## Other properties
none
