# Sources — deck 67: Asymmetric VPP / custom pipeline layout

## Sources read

| Tag | Source | Where |
|---|---|---|
| [RD] | Megatron-LM `megatron/core/transformer/moe/README.md` (Notion reference; cached copy) | "What's New" MCore v0.13 (line 27); Guideline 3 / VPP size tuning (lines 195-205) |
| [DOC] | Megatron-LM `docs/user-guide/features/pipeline_parallel_layout.md` (main, fetched 2026-10-13): "Custom Pipeline Model Parallel Layout", marked "experimental feature and may be changed" | whole page (37 lines) |
| [CODE-L] | `megatron/core/transformer/pipeline_parallel_layer_layout.py` (main): `parse_str_to_list`, `__init__` (stage i → vpp i // PP, pp i % PP), `validate_layer_layout`, `pretty_repr` | |
| [CODE-CFG] | `megatron/core/transformer/transformer_config.py` docstrings of `pipeline_model_parallel_layout`, `account_for_embedding_in_pipeline_split`, `account_for_loss_in_pipeline_split` (lines ~118-160) and conflict checks (~2760-2800); `megatron/training/arguments.py` (~805-860): uniform VPP divisibility asserts, mutual exclusion of `--num-layers-per-virtual-pipeline-stage`, `--num-virtual-stages-per-pipeline-rank`, `--pipeline-model-parallel-layout`; VPP derived as stages / PP | |
| [MC] | NVIDIA, "Scalable Training of Mixture-of-Experts Models with Megatron Core", arXiv 2603.07685v2 | §4.2.3 (p.33), §4.1.7 (p.27-28), §7.5 + Table 10 + Fig. 41 (p.65-66), §9.1.2 Guideline 3 (p.71), Table 17 + §9.2.2 (p.74-75), App. B Table 20 (p.87) |
| [PTDP] | D. Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (PTD-P), arXiv 2104.04473 (listed in the Notion row) | §2.2.1-2.2.2 bubble formulas (p.3-4), abstract and §5.3.2 (interleaving gain) |
| [DSV3] | DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv 2412.19437 | §4.2 hyper-parameters: all FFNs except the first three layers are MoE |
| [CODE-CFG11] | `megatron/core/transformer/transformer_config.py` assert: EP A2A overlap needs `virtual_pipeline_model_parallel_size` when PP > 1 (see deck 11 sources) | |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 61 decoder layers + 1 MTP layer | DeepSeek-V3 stack | [MC] §7.5 (p.66); [DOC] |
| 3 dense + 58 MoE | DeepSeek-V3 decoder composition | [DSV3] §4.2 ("all FFNs except the first three layers"); [MC] §4.2 p.29 (58 MoE layers) |
| "3 dense decoders ≈ cost of 2 MoE layers" | placement heuristic for the first stage | [MC] §7.5 p.66 and Fig. 41 ("Embed + 3*Dense ≈ 2*MoE") |
| 61 is prime; 61 / 32 = 1.9 | why no uniform split | arithmetic (derived) |
| `Et*3|(tt|)*29,m|L`, PP16 → 32 stages, VPP2; rank table (PP0: E+3t | 2t; PP1–13: 2t | 2t; PP14: 2t | m; PP15: 2t | L) | documented example | [DOC]; [MC] Table 10 |
| `Et*3|(tt|)*29m|L` on H100 (PP8, VPP4) and `Et*4|(tttt|)*14tmL` on GB200 (PP4, VPP4) | production layouts | [MC] §9.2.2 (p.74-75) |
| 1,024 H100, TP2/PP8/EP64, VPP4, GBS 8192, MBS 1, 368 TFLOPS/GPU; 256 GB200, TP1/PP4/EP64, VPP4, GBS 8192, MBS 1, 1,048 TFLOPS/GPU | configurations that use the layouts | [MC] Table 17 (p.74) |
| bubble fraction (p−1)/m; interleaved (1/v)·(p−1)/m; P2P communication × v | PTD-P model | [PTDP] §2.2.1-2.2.2 |
| "up to 10%" | throughput gain of the interleaved schedule (GPT-3 175B, 96 A100) | [PTDP] abstract and contributions ("as much as 10%"), Fig. 12 |
| v = 24 layers, PP4, VPP2 → [6,6,6,6] | uniform VPP example | [MC] §7.5 (p.65) |
| MCore v0.13 (2025/07) | release that added the layout flag | [RD] line 27 |

Derived / analytic (labelled on slides):
- **m = 128** micro-batches per pipeline for both DeepSeek-V3 configs: m = GBS / (MBS · DP), DP = world / (TP · PP)
  (attention DP; CP = 1): H100 1024 / (2 · 8) = 64 → 8192 / 64 = 128; GB200 256 / (1 · 4) = 64 → 128.
- **Analytic bubble** (PTD-P formula, balanced stages assumed): H100 p = 8: v = 1 → 7/128 = **5.5%**, v = 4 → 7/512 = **1.4%**;
  GB200 p = 4: v = 1 → 3/128 = **2.3%**, v = 4 → 3/512 = **0.6%**. Chart `charts/bubble.py`.
- Per-rank contents of the H100 layout (stage s → PP s mod 8, chunk s div 8): PP0 = E + 3 dense + 6 MoE; PP1–5 = 8 MoE;
  PP6 = 6 MoE + MTP (chunk 3); PP7 = 6 MoE + loss (chunk 3). Computed from the layout string with the [CODE-L] mapping.
- Per-rank contents of the GB200 layout (PP4): PP0 = E + 16 t; PP1–2 = 16 t; PP3 = 13 t + MTP + loss (61 t in total).

## Mechanism facts used on slides

- Grammar: `E` embedding, `t` transformer decoder layer, `m` MTP layer, `L` loss; `|` separates stages; `x*n` and
  `(…)*n` repeat; commas are cosmetic; empty stages allowed ([DOC]; [CODE-L] `parse_str_to_list`).
- Number of stages must be divisible by PP; VPP = stages / PP ([CODE-L] `__init__`; [CODE-CFG] arguments.py).
- Stage i → (vpp = i // PP, pp = i % PP) ([CODE-L] `__init__`; [CODE-CFG] docstring "layout[i] ... a-th vpp stage and b-th pp stage").
- Validation ([CODE-L] `validate_layer_layout`): first layer E, last layer L, exactly once each; count of `t` = num_layers;
  count of `m` = mtp_num_layers; no decoder after an MTP layer; MTP only in the last virtual stage of its rank, all MTP layers
  in one virtual stage, not on PP rank 0; MTP on a rank other than the last → "standalone MTP"; encoder layers unsupported.
- Old knobs ([CODE-CFG]): `--num-layers-per-virtual-pipeline-stage` requires num_layers (+1 if embedding is
  accounted, +1 if loss is accounted) divisible by PP and by layers-per-virtual-stage; `--account-for-embedding/loss-in-pipeline-split`
  treat embedding/loss "as a standard transformer layer"; `--decoder-first/last-pipeline-num-layers` set only the first/last stage;
  all of these are mutually exclusive with the layout flag.
- Guideline: "Larger VPP usually reduces pipeline bubbles but increases P2P communications; a middle value often provides
  the best balance. The workloads of each VPP rank should be balanced" ([MC] §9.1.2 Guideline 3).
- 1F1B EP overlap is faster with large VPP; flexible asymmetric VPP is used to balance hybrid dense/MoE stages for overlap
  ([MC] §4.2.3 p.33). EP A2A overlap requires VPP when PP > 1 ([CODE-CFG11]).
- Alternative: FSDP + EP avoids PP stage design entirely ("uneven PP/VPP stage balancing, placement of output and MTP layers")
  ([MC] §4.1.7 p.27-28; deck 70).

## Discrepancies

1. **Notion: "shrinks pipeline bubbles".** No primary source reports a measured bubble or throughput change from the layout
   itself; the Megatron report gives the DeepSeek-V3 layouts only inside full-stack configurations. The slides use
   the PTD-P bubble formula as an *analytic* estimate and say so.
2. **Notion: "Fixes the PP load imbalance caused by MTP/embedding layers".** The source describes cost-aware manual
   placement ("practitioners can design layouts"); there is no automatic balancer. Slides say the user writes the balance.
3. **README vs docs vs paper on the example string**: [DOC] uses `Et*3|(tt|)*29,m|L` (PP16, VPP2); the paper's H100 layout
   is `Et*3|(tt|)*29m|L` (PP8, VPP4). Both parse to the same 32 stages; the PP size decides the VPP size. Not a conflict, but
   the same string means different rank tables under different PP.
4. **[MC] Table 17 vs Table 20 (H100 PP)**: Table 17 says PP8 (consistent with the 32-stage layout and VPP4); App. B Table 20
   says "TP2 PP4". Slides use PP8.
5. Notion lists the PTD-P and Megatron-LM papers; neither covers asymmetric layouts (they predate it); PTD-P is used only for
   the interleaved-schedule bubble model.
