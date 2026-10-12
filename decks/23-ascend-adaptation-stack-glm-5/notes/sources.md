# Sources — deck 23: Ascend adaptation stack (GLM-5)

## Sources read

| Tag | Source | Where |
|---|---|---|
| [GLM5] | GLM-5 Team (Zhipu AI & Tsinghua), "GLM-5: from Vibe Coding to Agentic Engineering", arXiv 2602.15763. https://arxiv.org/abs/2602.15763 (cached text in `$CACHE/arxiv-2602.15763.txt`) | §1 Introduction p.4 (seven chip platforms); §2.1 p.4-5 (744B/40B, 256 experts, MLA head reshaping); §2.4 p.9 (training infrastructure); §5 "Adapting GLM-5 to Chinese Chip Infrastructure" p.21-22 (three pillars, result claim); App. A Table 10 p.36 (architecture) |
| [VA-GLM5] | vLLM Ascend project, deployment tutorial `docs/source/tutorials/models/GLM5.md` (repo `vllm-project/vllm-ascend`, commit 8194ece, fetched 2026-10-13 with `gh api`) | §3 weights table (Atlas 800 A3 / A2 nodes), §5 deployment flags, §5.3 Prefill-Decode disaggregation, §9.1 Tables 1-3 (reference configurations) |
| [VA-SP] | vLLM Ascend, `docs/source/user_guide/feature_guide/sequence_parallelism.md` (same commit) | FlashComm switch, SP MoE dataflow, `fuse_gemm_comms` |
| [GT] | Global Times, "Zhipu AI releases GLM-5.3-Flash model, powered by 100,000 domestically made chips", 27 Aug 2026. https://www.globaltimes.cn/page/202608/1369157.shtml | whole piece (company claim relayed by press) |
| [FLUX] | Chang et al., "FLUX: Fast Software-based Communication Overlap on GPUs via Kernel Fusion", arXiv 2406.06858 | §1 Fig. 1, §6 discussion p.15 (TP communication share by fabric) — used only on the sibling deck 12 |

`$CACHE` = the shared research cache for this series. Downloaded copies for this deck live in
`$SCRATCH/research/23/`.

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| **7** chip platforms (Huawei Ascend, Moore Threads, Hygon, Cambricon, Kunlunxin, MetaX, Enflame) | platforms GLM-5 reports full-stack adaptation for | [GLM5] §1 p.4 and §5 p.21 — the two lists are identical |
| 744B total / 40B activated parameters | GLM-5 size | [GLM5] §2.1 p.4; App. A Table 10 p.36 |
| 256 experts, 8 routed + 1 shared, 75 MoE + 3 dense + 1 MTP layer, hidden 6144, MoE intermediate 2048 | GLM-5 architecture | [GLM5] Table 10 p.36 |
| 28.5T training tokens | token budget | [GLM5] §1 p.4, §2 p.4 |
| **1.4 TiB** BF16 weights (1,386 GiB) | derived — see "Derived numbers" below | derived from [GLM5] Table 10 |
| **1.0 TiB** per node (8 × 128 GB) | memory of one Atlas 800 A3 node | [VA-GLM5] §3 weights table ("1 Atlas 800 A3 (128GB × 8) node") |
| ~693 GiB (W8A8) and ~354 GiB (W4A8 mixed) weight memory | derived — see below | derived from [GLM5] Table 10 + the precision assignment in [GLM5] §5 p.21 |
| **13** pre-processing operators fused into one MLAPO "super operator" | kernel fusion pillar | [GLM5] §5 p.22 |
| W4A8 for MoE experts (INT4), W8A8 (INT8) for attention and MLP blocks | quantization pillar | [GLM5] §5 p.21 |
| 750B (report's wording in §5) vs 744B (Table 10) | model size as quoted in the deployment section | [GLM5] §5 p.21 vs Table 10 p.36 — see Discrepancies |
| "a single Chinese node achieves performance comparable to dual-GPU international clusters" | headline result claim, quoted as written | [GLM5] §5 p.22 |
| **50%** lower deployment cost in long-sequence scenarios | headline result claim | [GLM5] §5 p.22 |
| head dim 192 → 256, attention heads cut by 1/3 | model-shape change because the DeepSeek-V3 head count follows the H800 roofline and "is inappropriate for other hardware" | [GLM5] §2.1 p.5 |
| 100,000 domestic chips (GLM-5.3-Flash) | company claim relayed by press, about a *later* model in the series | [GT] (27 Aug 2026) |
| MTP speculation 3 tokens; w4a8 / w8a8 / w8a8c8 weights on 1 Atlas 800 A3 or 2 Atlas 800 A2 nodes; 198K / 202,752-token serving configs | upstream reference deployment recipes | [VA-GLM5] §3, §5, §9.1 Table 1 |

## Derived numbers (analytic; used only in `charts/weight_memory.py` and on slides 3 and 10)

All derived values are marked "analytic" on the slide.

Expert weight count, from [GLM5] Table 10 (hidden 6144, MoE intermediate 2048, 256 experts,
1 shared expert, 75 MoE layers; SwiGLU ⇒ 3 matrices per expert):

```
per expert         = 3 × 6144 × 2048            = 37.75 M params
routed experts     = 37.75 M × 256 × 75         = 724.8 B
shared experts     = 37.75 M × 1  × 75          =   2.8 B
expert weights     = 727.6 B  (97.8% of the 744B total)
everything else    = 744 B − 727.6 B            =  16.4 B
```

Cross-check: the activated parameter count implied by this arithmetic is
9 × 37.75 M × 75 = 25.5 B of expert weights per token, consistent with the published 40B activated
(the remainder being attention, dense layers, the indexer and the embedding/output path).

Weight memory (1 GiB = 2^30 bytes; excludes quantization scales/zero-points, KV cache and activations):

```
BF16        744 B × 2 B                         = 1,386 GiB  ("1.4 TiB" on the slide)
W8A8 INT8   744 B × 1 B                         =   693 GiB
W4A8 mixed  727.6 B × 0.5 B + 16.4 B × 1 B      =   354 GiB
one node    8 × 128 GB                          = 1,024 GiB  ("1.0 TiB")
```

The report does not publish a memory breakdown; these three bars are an analytic illustration of why
the quantization pillar exists, not measurements. The KV-cache headroom on slide 10 is stated
qualitatively ("leaves room for the KV cache"), not as a number.

## Facts without numbers used on slides

- §5 is explicitly a *deployment/inference* adaptation: the three pillars are quantization, fusion
  kernels and inference-engine scheduling, and the result is phrased as deployment cost ([GLM5] §5).
  The report never claims GLM-5 was pre-trained on domestic chips; its training-infrastructure section
  (§2.4) talks about GPU memory, pipeline parallelism and the Muon distributed optimizer ([GLM5] §2.4 p.9).
- Ascend Atlas is used "as a case study"; the other six platforms are named but not detailed ([GLM5] §5 p.21).
- Quantization tooling: msModelSlim; QuaRot for outlier suppression; Flex_AWQ_SSZ for scaling
  calibration ([GLM5] §5 p.21). → deck 44.
- Fusion kernels: Lightning Indexer (score + ReLU + TopK in one kernel, overlapping compute with memory
  access), Sparse Flash Attention (TopK selection from the KV cache in parallel with the sparse attention
  computation), MLAPO (13 pre-processing operators fused, using Vector and Cube units in parallel)
  ([GLM5] §5 p.21-22). → deck 37.
- Engine scheduling: asynchronous scheduling (overlap D2H sampling copies with the next decode step's
  preparation), RadixCache prefix sharing + Prefix Cache extending KV storage to system RAM,
  DP attention + MoE EP + FlashComm (splits AllReduce to hide communication behind computation),
  MTP ([GLM5] §5 p.22). → deck 12.
- Upstream corroboration ([VA-GLM5], [VA-SP]): the same mechanisms exist as flags in vLLM Ascend —
  `enable_flashcomm1` (FlashComm_v1 = AllReduce split into Reduce-Scatter + All-Gather),
  `enable_mlapo` (MLA pre-process fusion), `enable_fused_mc2`, `enable_dsa_cp`, prefix caching,
  `--speculative-config` with `deepseek_mtp`, and Prefill-Decode disaggregation with Mooncake KV transfer.

## Discrepancies

1. **Model size in the deployment section.** [GLM5] §5 p.21 says "the 750B parameter GLM-5 model" while
   Table 10 p.36 gives 744B total parameters. The slides use 744B and never quote 750B as a fact.
2. **Notion: "A single domestic GLM-5 node matches a dual-GPU cluster".** The report's sentence is
   "GLM-5 on a single Chinese node achieves performance comparable to dual-GPU international clusters"
   ([GLM5] §5 p.22). The report names neither the GPU model, the node count behind "dual-GPU", the
   benchmark, the metric, nor the serving configuration, so the slide quotes the claim and labels it as
   the report's unqualified wording rather than restating it as a measured 2× result.
3. **Node naming.** [GLM5] §5 p.21 writes "a single Atlas 800T A3 machine"; the upstream deployment
   guide writes "Atlas 800 A3 (128GB × 8)" ([VA-GLM5] §3). The slides use the upstream spelling, since the
   node-memory figure comes from there.
4. **Notion property calls the row "domestic GPU/NPU ecosystems".** The report's own §1 heading phrase is
   "Chinese GPU ecosystems"; §5 is "Chinese Chip Infrastructure". Ascend parts are NPUs, Moore Threads and
   MetaX parts are GPUs. The slides say "domestic accelerators" and name Ascend Atlas where the report does.
5. **Training vs inference.** Neither the Notion row nor the report claims pre-training on domestic
   accelerators; the Notion wording "day-one full-stack engineering" could be read that way. The deck
   states explicitly that §5 covers deployment, and that the training hardware is not stated in the report.
6. **Press claim [GT] is about GLM-5.3-Flash, not GLM-5**, is a company statement relayed by a newspaper,
   and names no chip vendor. The deck cites it on the adoption slide with that attribution only.
