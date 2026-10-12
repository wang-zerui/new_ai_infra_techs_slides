# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Container entry: DeepSeek's expert-parallel communication library, rewritten in V2. Sub-techniques are the rows whose "Parent Library / Framework" points here: ElasticBuffer unified API, NCCL Gin backend, elastic GPU/CPU buffers, 0-SM PP/CP/Engram, FP8 dispatch + BF16 combine, and network-side traffic isolation.
**After:** Container entry: DeepSeek's expert-parallel communication library, rewritten in V2 (Apr 2026) and restructured in V2.5 (Sep 2026). Sub-techniques are the rows whose "Parent Library / Framework" points here: ElasticBuffer unified API (EPBuffer since V2.5), NCCL Gin backend, elastic GPU/CPU buffers (shipped for Engram storage only), 0-SM PP/CP/Engram (experimental), FP8 dispatch + BF16 combine, and network-side traffic isolation.
**Why:**
- V2.5 split ElasticBuffer into EPBuffer / EngramBuffer / PPBuffer / BucketBuffer ([R25] News; [PR763]).
- Elastic GPU/CPU buffers were "Still on-going" in V2; V2.5 ships Engram storage on GPU or CPU, not a hybrid buffer for EP ([R2]; [R25] News).
- PP / Engram / bucket collectives are marked experimental, and V2.5 drops the "0 SM" label ([R25] features; [PR763]).

## Impact / Measured Results
**Before:** Several times lower SM usage than V1 and scale-up/out domains up to EP2048; FP8 dispatch with IB virtual-lane isolation and adaptive routing.
**After:** Up to 4x fewer SMs than V1 (24 to 4-6 for V3-like training) at up to 1.3x peak performance; V2 README claims scale-up/out domains up to EP2048, but published benchmarks stop at EP32 (8x4); FP8 dispatch with IB virtual-lane isolation and adaptive routing.
**Why:**
- Replaced "several times" with the release's own numbers ([R2] New features and Performance: "24 to 4 - 6", "up to 1.3x peak performance, while saving up to 4x SM count").
- EP2048 is a V2 README statement; V2.5 no longer gives a number and the largest published configuration is EP 8x4 ([R2], [R25]; [R2] "We omit results for larger EP configurations").
- Adaptive-routing advice matches V2 ("under all network load conditions"); kept.

## Other properties
none
