# Sources — deck 28: VeOmni (container entry)

## Sources read

| Tag | Source | Where |
|---|---|---|
| [VO] | Q. Ma et al., "VeOmni: Scaling Any Modality Model Training with Model-Centric Distributed Recipe Zoo", arXiv 2508.02317v3 (7 Aug 2025). https://arxiv.org/abs/2508.02317 (cached `arxiv-2508.02317.txt` + PDF) | Abstract (p.1); §1 + Fig. 1 (p.1-2); §3.1 + Fig. 2 (p.3-4); §3.2.1-3.2.5 + Fig. 3 (p.5-7); §4.1-4.3 + Fig. 4-6 (p.8-9); §5 (p.9); App. A Tables 1-4 (p.15-16); App. B.1-B.7 Listings 1-8 (p.17-23) |
| [OPT] | W. Feng et al., "Optimus: Accelerating Large-Scale Multi-Modal LLM Training by Bubble Exploitation", arXiv 2408.03505v2 (USENIX ATC'25); cited by [VO] as ref. [17]. Downloaded to `$SCRATCH/research/28/dl/` | Abstract, §1 (p.1), §2.2 + Table 1 (p.2-3) |
| [README] | VeOmni GitHub README (cached `gh-veomni-README.md`; repo main @ 57e5f6d, 2026-10-12) | Overview, Latest News, Key Features, Supported Models, "Awesome work using VeOmni" |
| [DOC-EP] | `docs/key_features/ep_fsdp2.md` (repo main @ 57e5f6d) | "Sharding Dimension", "Parallelism Setup", "Expert Parallelism Details", "Consequences of Mixed Sharding Strategy", "DCP Support" |
| [DOC-XP] | `docs/key_features/extra_parallel.md` | Motivation (checkpoint save and resharding load for different parallelism patterns) |
| [DOC-ULY] | `docs/key_features/ulysses.md` | Overview, DeepSpeed-Ulysses, Communication Analysis, "Support Ulysses for a New Model" |
| [DOC-NEW] | `docs/usage/support_new_models/guide_and_checklist.md` | "Integration Complexity by Model Type" table; Step 5 parallel plan |
| [DOC-PS] | `docs/design/local_parallel_state.md` | named ParallelState per module (thinker/talker example) |
| [DOC-CKPT] | `docs/usage/checkpoint.md` | DCP layout (weights + optimizer as DCP directories) |
| [DOC-PACK] | `docs/usage/data_packing_and_dyn_bsz.md` | packing with cu_seqlens; dyn_bsz |
| [DOC-HW] | `docs/hardware_support/rocm/README.md`, `docs/hardware_support/mlu/README.md`, `docs/hardware_support/get_started_npu.md` | supported accelerators |
| [CODE-PLAN] | `veomni/models/transformers/qwen3_moe/parallel_plan.py`; `veomni/distributed/parallel_plan.py` | the 2-entry Qwen3-MoE EP plan; `ParallelPlan.apply` |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 48% ("over 48%") | GPU cycles idle in ByteDance's internal >100B MLLM training (ViT encoder + GPT backbone) with Megatron-LM on >3,000 NVIDIA GPUs, even with MegaScale, Zero Bubble PP and fine-grained comm overlap | [OPT] §2.2 p.2-3 ("observed over 48% GPU cycle idleness"); §1 p.1 ("around 48% of GPU cycles are wasted") |
| >100B, >3,000 GPUs | size/scale of that job | [OPT] §2.2 p.2-3 |
| 5.12 s | average step time in that profile (notes only) | [OPT] Table 1 |
| Qwen3-MoE#Omni = Qwen2.5-Omni ViT (image/video), Qwen2.5-Omni Whisper (audio), Qwen3-30B-A3B, MoVQGAN (image decoder) | example omni model | [VO] §4.3 p.9 |
| 2,800+ tokens/s/GPU | throughput of the 30B MoE omni-modal LLM, 3D parallelism, 128 GPUs | [VO] Abstract ("over 2,800 tokens/sec/GPU") |
| 160K | max context of the same 30B MoE with FSDP+SP+EP on 128 GPUs | [VO] Abstract; §4.2 p.9 |
| 192K, 61.5% MFU | Qwen2-VL 7B, FSDP+SP, 8 GPUs | [VO] §4.2 p.8 and Fig. 4 |
| 96K, 54.82% MFU | Qwen2-VL 72B, FSDP+SP, 128 GPUs | [VO] §4.2 p.8-9 and Fig. 5 |
| 7B-72B, 8-128 GPUs | evaluation range | [VO] §1 contribution 3; §4.1 |
| SP 1-4 (7B), 1-8 (72B); SP and EP 1-4 (30B MoE) | parallel sizes explored | [VO] Fig. 4-6 captions |
| Memory per GPU, Qwen2-7B, FSDP, 128 GPUs, mbs 1, gbs 128, full activation checkpointing: VeOmni 21.91 / 24.45 / 31.41 / 43.00 / 70.26 GB at 8K/16K/32K/64K/128K; TorchTitan 23.26 / 39.58 / 71.81 / OOM / OOM | chart `mem_vs_titan` | [VO] App. A Table 1 p.15 |
| 2.3× | 71.81 / 31.41 = 2.29 (derived), TorchTitan vs VeOmni memory at 32K | derived from [VO] Table 1 |
| 5,315 vs 4,878 tokens/s/GPU | Qwen2-7B FSDP 32K throughput, VeOmni vs TorchTitan (notes only) | [VO] Table 1 |
| 7.45%-17.92% MFU (slide rounds to 7–18%) | Qwen3-30B-A3B text-only MoE runs (FSDP+SP{1,4}+EP8, 8K-128K, 128 GPUs) | [VO] App. A Table 4 p.16 |
| 2,853 / 1,075 tokens/s/GPU | Qwen3-30B-A3B text MoE: FSDP+SP1+EP8 at 16K / FSDP+SP4+EP8 at 128K (notes, discrepancy 1) | [VO] Table 4 |
| 128 experts, hidden 2048, intermediate 768; weights [128, 2048, 768] (down_proj) | Qwen3-30B-A3B expert shapes | [DOC-EP] "Parallelism Setup" |
| 16 GPUs (2 × 8), EP 8 → 16 experts/rank; ranks 0 and 8 form an EP-FSDP group of degree 2; dim-1 2048 → 1024 | worked EP+FSDP2 example | [DOC-EP] "Parallelism Setup" |
| EP size × expert-FSDP size = world size | mesh rule | [DOC-EP] summary |
| 2 entries | Qwen3-MoE EP plan has two FQN patterns (`gate_up_proj`, `down_proj`), each `Shard(0)` | [CODE-PLAN] qwen3_moe/parallel_plan.py |
| two all-to-alls; [S/P, H] → [S, H/P]; P = 4 in the figure | Ulysses re-sharding around attention | [DOC-ULY] "DeepSpeed-Ulysses"; [VO] §3.2.2 and App. B.6 |
| heads divisible by SP size | Ulysses constraint | [DOC-ULY] |
| `__init__.py` + 4 more files | integration effort for an omni-modal MoE (dense text: `__init__.py` only) | [DOC-NEW] "Integration Complexity by Model Type" |
| 8 × MI308X | AMD ROCm validation | [DOC-HW] rocm README |
| Ascend A2 / A3 / 950 | supported Ascend products | [DOC-HW] get_started_npu.md |
| 2025/04 open source; 2025/08 tech report; 2025/09 v0.1.0; AAAI 2026 | timeline | [README] Latest News |

Derived numbers: 2.3× = 71.81 GB / 31.41 GB (TorchTitan vs VeOmni, Qwen2-7B FSDP at 32K, [VO] Table 1).
The chart plots the Table 1 memory values directly; OOM entries are drawn as labelled empty slots.

## Mechanism facts (no numbers) used on slides

- Encoder/decoder modules extend HF `PreTrainedModel` plus a mixin; encoders implement `lm_encode`
  (raw modality → token embeddings inserted into the foundation's input embeddings); decoders implement
  `lm_encode` (training-time target tokens) and `lm_head` (loss), plus `lm_embed`/`lm_generate` for
  inference ([VO] §3.1, Fig. 2, Listing 3: `masked_scatter` with per-modality input/output masks).
- Parallel plan: FQN wildcard → DTensor placement (`Shard(0)`) applied by `build_parallelize_model`
  ([VO] App. B.5, B.7; [CODE-PLAN]). A global DeviceMesh replaces manual process groups ([VO] §3.2.4,
  App. B.4). Named ParallelStates let modules of one job use different SP sizes ([DOC-PS]).
- FSDP2 is applied bottom-up; experts get EP on dim-0 and FSDP2 on dim-1; non-experts FSDP2 over the
  world; manual prefetch of next layer's attention and experts; split optimizers / LR schedulers /
  grad-norm clipping for expert vs non-expert params ([DOC-EP]).
- HSDP = FSDP inside shard groups + DDP across replicate groups, switched by one config value ([VO] §3.2.1).
- Ulysses: all-to-all before attention (gather sequence, scatter heads) and after (gather heads,
  scatter sequence); communication volume stays constant when S and P scale together; Async-Ulysses
  overlaps the all-to-alls with the linear projections ([VO] §3.2.2; [DOC-ULY]).
- MoE all-to-all overlap is operator-level (Flux/Comet-style), not a pipeline schedule like DualPipe ([VO] §3.2.3).
- Other optimizations kept outside model code: dynamic batching, liger kernels + FlashAttention + MoE ops,
  recompute + activation/optimizer offload, ByteCheckpoint (paper) / torch DCP (repo), meta-device
  init with DTensor parallel loading ([VO] §3.2.5; [README]).
- Eval setup: encoders/decoders frozen, foundation + projectors fully fine-tuned ([VO] §4.1).
- Future work: non-intrusive pipeline parallelism; modality-aware SP data balancing ([VO] §5).
- Hardware: NVIDIA GPU, AMD ROCm, Ascend NPU ([README]); Cambricon MLU ([DOC-HW] mlu README:
  "most of the device-agnostic training path ... applies as-is").
- Users listed in the README: UI-TARS / UI-TARS-2, LMMs-Engine, dFactory, Open-dLLM, OpenHA, LingBot-VLA.
- Supported model families listed in the README: DeepSeek-V2.5/V3/R1, DeepSeek-V4, Llama 3, Qwen2-3,
  Qwen VL/MoE/Omni, GPT-OSS, Wan 2.1, MiniMax H3, LTX-2.3.

## Discrepancies

1. **Notion: "More than 2800 tokens/s/GPU for a 30B omni-modal MoE on 128 GPUs with 160K context."**
   The paper states two separate facts: throughput "over 2,800 tokens/sec/GPU" and the ability to
   "scale to 160K context lengths". Fig. 6(a) shows the highest throughput at the shortest context
   (16K, SP1/EP1) and the 160K point only with larger SP/EP at much lower throughput; the text-only
   Qwen3-30B-A3B table shows the same pattern (2,853 tokens/s/GPU at 16K vs 1,075 at 128K, Table 4).
   Slides keep the two claims separate.
2. **Notion: "FSDP1/2".** The paper integrates FSDP1 and FSDP2 (§3.2.1), but the current repo says
   FSDP1 "has since been deprecated upstream and removed from VeOmni"; FSDP2 only ([DOC-EP]).
   Slides say FSDP2 (HSDP mode).
3. **Notion: "DCP checkpointing".** The paper (Aug 2025) uses ByteCheckpoint (§3.2.5); the open-source
   repo uses PyTorch Distributed Checkpoint ([README], [DOC-CKPT]). Slides follow the repo and deck 50.
4. **Notion: "adding a new modality takes very little code".** The paper says "minimal code change"
   without a number; the repo guide lists `__init__.py` + 4 more files for an omni-modal MoE and
   per-model patches for SP and fused MoE ([DOC-NEW]). Slides state the file count as the cost.
5. **README news calls the AAAI 2026 paper "OmniScale"** while the arXiv title is VeOmni; slides use VeOmni.
