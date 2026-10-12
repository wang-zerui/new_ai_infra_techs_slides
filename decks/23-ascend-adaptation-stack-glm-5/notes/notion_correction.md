# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Container entry: GLM-5's day-one full-stack engineering for domestic GPU/NPU ecosystems, built on three pillars - aggressive quantization, high-performance fused operators, and inference-engine scheduling. Sub-techniques are the rows pointing here.
**After:** Container entry: GLM-5's full-stack deployment (inference) adaptation for seven domestic chip platforms, with Ascend Atlas as the case study, built on three pillars - aggressive quantization (W4A8 experts, W8A8 elsewhere), high-performance fused operators, and inference-engine scheduling. Sub-techniques are the rows pointing here.
**Why:**
- §5 is a deployment/inference adaptation; the report never says GLM-5 was trained on domestic chips, so the scope is stated ([GLM5] §5 p.21-22, §2.4 p.9; discrepancy item 5).
- "Day-one" is not in any source read; dropped.
- Seven platforms and Ascend Atlas "as a case study" ([GLM5] §1 p.4, §5 p.21); precision split from [GLM5] §5 p.21.

## Impact / Measured Results
**Before:** A single domestic GLM-5 node matches a dual-GPU cluster, with 50% lower long-sequence serving cost.
**After:** Report claims, without naming GPU, benchmark or metric: a single Chinese node performs comparably to dual-GPU international clusters, and long-sequence deployment cost falls 50%.
**Why:**
- The report's sentence is "comparable to dual-GPU international clusters" and gives no GPU model, node count, benchmark, metric or serving configuration ([GLM5] §5 p.22); restated as the report's claim, not a measurement.

## Other properties
none

Left for the user: the row's Source is "GLM-5 series, VeOmni". None of the sources read for this deck mentions VeOmni (ByteDance's training framework, deck 28); the discrepancy notes do not flag it, so it was not changed.
