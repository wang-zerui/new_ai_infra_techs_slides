# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Narrower than DeepSeek-V3/K2 (hidden dim, number of routed experts) but deeper (89 MoE layers); GQA with partial RoPE and 2.5x more heads (96 heads at 5120 hidden).
**After:** unchanged
**Why:** all statements match the sources: hidden 5120 vs 7168, 160 vs 256/384 routed experts, 3 dense + 89 MoE layers, GQA 96 Q / 8 KV heads with partial RoPE ([GLM45] §2.1, Table 1; [CFG-GLM45]). "2.5x more heads" is the paper's own published figure ([GLM45] §2.1 p.3); the derived 2.4x against a width-matched 40-head layout is a refinement, not a correction, so the published number is kept.

## Impact / Measured Results
**Before:** Deeper models reason better; more heads do not lower training loss yet keep improving MMLU/BBH.
**After:** GLM-4.5 authors report, without published ablation numbers, that deeper models reason better and that more heads do not lower training loss yet consistently improve MMLU/BBH.
**Why:**
- Both claims are the authors' qualitative findings; [GLM45] §2.1 p.2-3 publishes no numbers for either ablation, so they should not read as measured results.
- "Keep improving" changed to the paper's wording "consistently improves" ([GLM45] §2.1 p.3).
- Not added (length): at base-model level GLM-4.5-Base trails DeepSeek-V3-Base and Kimi-K2-Base on MMLU and BBH, confounded by 23T vs 14.8T / 15.5T tokens ([GLM45] Table 2, Abstract).

## Other properties
none
