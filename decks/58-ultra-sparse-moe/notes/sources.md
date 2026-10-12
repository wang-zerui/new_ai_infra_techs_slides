# Sources — deck 58: Ultra-sparse MoE (fine-grained routed experts + shared expert)

## Sources read

| Tag | Source | Where |
|---|---|---|
| [V4] | DeepSeek-AI, "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence", arXiv 2606.19348v1 (26 Apr 2026). https://arxiv.org/abs/2606.19348 (cached text `arxiv-2606.19348.txt`) | Abstract (p.1); §1 (p.4-5); §2 + §2.1 "Designs Inherited from DeepSeek-V3 / Mixture-of-Experts" (p.6-7); §3.1 (p.15-16); §4.2.1 Model Setups (p.24-25); §4.2.2 (p.25-26); §4.2.3 (p.26); Table 1 (p.28) |
| [K2] | Kimi Team, "Kimi K2: Open Agentic Intelligence", arXiv 2507.20534 (cached) | §2.3 Model Architecture + Table 2 (p.6); "Sparsity Scaling Law" + Fig. 5 (p.7); §2.4.2 Parallelism (p.8) |
| [GLM5] | GLM-5 Team, "GLM-5", arXiv 2602.15763 (cached) | §2.1 "Model size scaling" (p.4); App. A Table 10 (p.36) |
| [V3] | DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv 2412.19437 (cached) | §2.1.2 (p.8-9, MoE formula); §3.2.2 (p.13-14, node-limited dispatch); §4.2 Hyper-Parameters (p.22-23) |
| [DSMoE] | Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models", arXiv 2401.06066 (downloaded to `$SCRATCH/research/58/`) | Abstract; §1 (p.2-3); §3.1 Fine-Grained Expert Segmentation (p.5-6); §3.2 Shared Expert Isolation (p.6) |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 1.6T total / 49B activated | DeepSeek-V4-Pro | [V4] Abstract p.1; §4.2.1 p.25 |
| 284B total / 13B activated | DeepSeek-V4-Flash | [V4] Abstract p.1; §4.2.1 p.24 |
| V4-Pro: 61 layers, hidden 7168, 1 shared + 384 routed experts, expert intermediate dim 3072, 6 routed active | V4-Pro MoE config | [V4] §4.2.1 p.25 |
| V4-Flash: 43 layers, hidden 4096, 1 shared + 256 routed, expert dim 2048, 6 routed active | V4-Flash MoE config | [V4] §4.2.1 p.24 |
| MoE in all blocks; Hash routing for first 3 MoE layers | V4 layer layout | [V4] §2.1 p.7; §4.2.1 p.24-25 |
| Sqrt(Softplus) affinity; aux-loss-free balancing + sequence-wise balance loss (weight 0.0001), bias update speed 0.001; node-limit removed | V4 routing changes | [V4] §2.1 p.7; §4.2.2 p.25 |
| 1.04T total / 32.6B activated (abstract: 32B); 384 experts, 8 active, 1 shared; 61 layers; 1 dense layer; hidden 7168; expert dim 2048 | Kimi K2 | [K2] §2.3 Table 2 p.6 (32.6B); abstract (32B) |
| sparsity = total experts ÷ activated experts; K2 uses 48 (8 of 384) | definition and choice | [K2] p.7 "Sparsity Scaling Law" |
| Fixed 8 activated + 1 shared expert, vary total; loss falls consistently with sparsity | scaling-law setup | [K2] p.7 + Fig. 5 caption |
| 1.69×, 1.39×, 1.15× | FLOPs reduction of sparsity 48 vs. 8, 16, 32 at equal validation loss 1.5 | [K2] p.7 |
| "increased infrastructure complexity" | why K2 stops at 48 | [K2] p.7 |
| ≈ 6 TB GPU memory for BF16 params + FP32 grad buffers over a 256-GPU model-parallel group; EP = 16 (smallest feasible) | cost of holding all experts / small EP | [K2] §2.4.2 p.8 |
| 744B total / 40B activated; 256 experts, 8 routed, 1 shared; hidden 6144; MoE intermediate 2048; 75 MoE layers (+3 dense, +1 MTP) | GLM-5 | [GLM5] Table 10 p.36; §2.1 p.4 |
| GLM-5 "reduces its layer count to 80 to minimize expert parallelism communication overhead" | EP cost shaping the model | [GLM5] §2.1 p.4 |
| 671B / 37B; 256 routed + 1 shared, 8 active, expert dim 2048; 58 MoE layers (first 3 dense of 61); ≤ 4 nodes per token | DeepSeek-V3 reference | [V3] §4.2 p.22-23 |
| 120 vs. 4,426,165,368 | combinations: top-2 of 16 experts vs. top-8 of 64 quarter-size experts (same FLOPs) | [DSMoE] §3.1 p.6 |
| 1.5× | GShard 2.9B (GShard×1.5: 1.5× expert size, hence 1.5× expert parameters and expert computation, top-2 routing) only matches DeepSeekMoE 2B | [DSMoE] Abstract; §1 p.3; §4.3 p.10; Fig. 2(a) conventional top-2 |
| ~40% | DeepSeekMoE 16B matches LLaMA2 7B with about 40% of the computation | [DSMoE] Abstract; §1 p.3 (speaker notes only) |
| expert segmentation: split FFN intermediate dim to 1/m, activate m× more; shared experts: K_s always on, routed top-k reduced by K_s | mechanism | [DSMoE] §3.1-3.2 p.5-6 |
| V4-Flash-Base (13B act.) vs V3.2-Base (37B act.): MMLU-Pro 68.3 vs 65.5; LongBench-V2 44.7 vs 40.2 | base-model comparison (speaker notes only; confounded) | [V4] Table 1 p.28 |
| 6hd FLOPs and 3h bytes per token-expert pair (h = hidden, d = expert dim) | per-expert compute (confirms 3 matrices h×d per SwiGLU expert) | [V4] §3.1 p.16 |

## Derived numbers (labelled "derived" on slides)

All from the configs above; SwiGLU expert = 3 matrices of hidden × expert-dim ([V4] §3.1: 6hd FLOPs per token-expert pair = 2 FLOPs per weight).

| Derived | Formula | Value |
|---|---|---|
| params per V4-Pro expert | 3 × 7168 × 3072 | 66.06M ≈ **66M** |
| V4-Pro experts per layer | 384 routed + 1 shared | **385** |
| experts touched per token per layer (V4-Pro) | (6 + 1) / 385 | **1.8%** |
| V4-Pro expert weights | 66.06M × 385 × 61 layers | 1,551B ≈ **1.55T** (≈ 97% of 1.6T; MTP module excluded; 1.6T is rounded) |
| V4-Pro expert weights used per token | 66.06M × 7 × 61 | **28.2B** (≈ 58% of the 49B active) |
| same accounting, V4-Flash | 3×4096×2048 = 25.2M/expert; × 257 × 43 | 278B of 284B (98%); 7.6B of 13B active |
| same accounting, K2 | 3×7168×2048 = 44.0M; × 385 × 60 MoE layers | 1.02T of 1.04T (98%) |
| same accounting, GLM-5 | 3×6144×2048 = 37.7M; × 257 × 75 | 728B of 744B (98%) |
| sparsity (total ÷ active routed experts) | V3 256/8, GLM-5 256/8, V4-Flash 256/6, K2 384/8, V4-Pro 384/6 | **32, 32, 42.7, 48, 64** |
| active ÷ total parameters | V3 37/671, GLM-5 40/744, V4-Flash 13/284, K2 32.6/1040, V4-Pro 49/1600 | **5.5%, 5.4%, 4.6%, 3.1%, 3.1%** |

Charts: `charts/sparsity_flops.py` plots the three published K2 ratios (1.69, 1.39, 1.15) with sparsity 48 = 1.0
as the reference. `charts/active_share.py` plots the derived active-share percentages above.

## Mechanism facts used on slides (no numbers)

- Knowledge hybridity (few experts → each expert covers diverse knowledge) and knowledge redundancy (common knowledge
  duplicated across experts) are the two failure modes of conventional MoE ([DSMoE] §1 p.2).
- Fine-grained segmentation keeps total expert parameters and per-token compute constant ([DSMoE] §3.1).
- Shared experts are deterministically assigned to every token regardless of the router ([DSMoE] §3.2).
- DeepSeek-V4 "still adopt[s] the DeepSeekMoE architecture, with only minor adjustments from DeepSeek-V3" ([V4] §2 p.7).
- V4 training instability: loss spikes consistently tied to outliers in the MoE layers; routing exacerbates them;
  mitigations = Anticipatory Routing and SwiGLU clamping ([V4] §4.2.3 p.26) → deck 63.
- V4 fine-grained EP: experts split into waves, communication hidden under computation ([V4] §3.1) → deck 14 / deck 35.

## Discrepancies

1. **Kimi K2 activated parameters.** Notion says "K2 1.04T/32B"; the K2 abstract says 32B, Table 2 says 32.6B.
   Slides use 32.6B where a ratio is computed and "32B" nowhere else.
2. **"Total and activated parameters fully decoupled".** Notion's wording; the sources show the two are decoupled
   only up to infrastructure limits: K2 explicitly stops at sparsity 48 because of infrastructure complexity, and
   GLM-5 reduces its layer count to cut EP communication. Slides present sparsity as a dial bounded by infra cost.
3. **"Sparsity chosen by scaling laws".** Only Kimi K2 publishes a sparsity scaling law. DeepSeek-V4 and GLM-5 state
   their configurations without a published sparsity sweep. Slides attribute the law to K2 only.
4. **Not every model uses "one shared expert + fine-grained routed" identically**: V4 also hash-routes the first 3 MoE
   layers and K2 keeps 1 dense layer (GLM-5 keeps 3 dense layers). Mentioned in speaker notes.
