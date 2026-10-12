# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Container entry: Moonshot's MIT-licensed EP communication library released with Kimi K3, aiming for identical compute per rank under any routing skew. Sub-techniques: dynamic redundant-expert balancing, online GPU planning kernel, zero-copy static shapes.
**After:** Container entry: Moonshot's MIT-licensed EP communication library released with Kimi K3, giving every rank exactly S x K tokens under any routing skew. Sub-techniques: dynamic redundant-expert balancing, online GPU planning kernel, zero-copy static shapes.
**Why:**
- What is equalized is the per-rank token count (S x K, S = input tokens per rank), not compute: per-expert counts inside a rank stay skewed, so K3 adds a workload-aware expert-GEMM scheduler ([README] property 1; [K3] §5.2.1).

## Impact / Measured Results
**Before:** Static shapes and no per-layer host sync; communication buffer drops from a worst-case S x K x R to a fixed S x K; increasingly outperforms DeepEP v2 as routing imbalance grows on H20/EP=8.
**After:** Static shapes and no per-layer host sync; communication buffer drops from a worst-case S x K x R to S x K plus alignment padding; on H20/EP=8 the advantage over DeepEP v2 grows with routing imbalance (end-to-end iteration time 1.9% slower at maxvio 0.2, 11.1% faster at maxvio 15; DeepEP v2 OOMs at maxvio 20).
**Why:**
- Actual allocation is NvS = align(S·K + 127·2·E/R, 128): 77,824 vs 65,536 slots at the benchmark config ([API]).
- "Outperforms" is not true at low imbalance: MoonEP is 1.9% slower end-to-end at maxvio 0.2 and its dispatch fwd is 2,598 vs 2,296 µs ([CHART-E2E], [CHART-COMM]); added the published endpoints.

## Other properties
none
