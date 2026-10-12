# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Takes a precomputed block mask, skips masked KV tiles wholesale, and applies per-tile FP8 scaling to preserve numerical quality under sparsity (paged KV supported).
**After:** Takes a precomputed 128x128 block mask, loads and computes only the active KV tiles, and runs FP8 with Q scaled per token and head, K per tensor or per token group and head, V per tensor or per head, P by a fixed 256 (no per-tile scales); paged KV supported.
**Why:**
- There is no scale computed per KV tile: Q per-token per-head, K per-tensor or per-token-group per-head, V per-tensor or per-head ([P] docstring), and P x 256 before its FP8 cast ([K] kFp8PrefillPScale) (sources.md discrepancy 1).
- 128 x 128 tiles ([E] kTileM = kTileN = 128); the producer warp builds an active-tile list and TMA-loads only those tiles ([K]).

## Impact / Measured Results
**Before:** Up to 3.16x over MIT-BSA / FlashPrefill-BSA (BF16) and a dense FP8 baseline; the companion kernel that makes DSA-style sparsity deployable.
**After:** Up to 3.16x vs MIT-BSA / FlashPrefill-BSA (BF16) and HPC / FA3 dense FP8 (README, no per-case table); at 10% block density it takes 11% of dense FP8 latency (H20, 64K). Paired with Stem block masks, not DSA (token-level).
**Why:**
- README lists two dense FP8 baselines (HPC-Dense FP8, FA3-Dense FP8) and publishes no per-case table ([R]) (discrepancy 2).
- The only per-density data is the PR #45 plot: 11% of dense FP8 latency at 10% density (batch 4, seq 65536, Hq 4, Hkv 1, D 128, H20) ([PR45]).
- "Companion kernel that makes DSA-style sparsity deployable" is unsupported: the kernel is block-sparse, DSA picks individual tokens (top-2048); the repo pairs it with Stem as the mask generator ([PR50]; [S]) (discrepancy 3).

## Other properties
none
