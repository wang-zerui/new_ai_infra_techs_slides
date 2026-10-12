# Story — deck 28: VeOmni (container entry)

- **Where it lives** (prosperity): ByteDance Seed's VeOmni (open-sourced 2025/04, tech report arXiv 2508.02317,
  v0.1.0 2025/09, AAAI 2026). Trains omni-modal LLMs built as modality encoders + an LLM "foundation" +
  modality decoders, e.g. Qwen3-MoE#Omni = Qwen2.5-Omni ViT + Whisper encoders, Qwen3-30B-A3B backbone,
  MoVQGAN image decoder. Evaluated on 7B-72B dense and a 30B omni MoE, 8-128 GPUs.
- **Pain number** (betrayal): **over 48% of GPU cycles idle** when ByteDance trained a >100B ViT+GPT
  multimodal LLM with Megatron-LM on >3,000 GPUs (Optimus, arXiv 2408.03505 §1/§2.2). VeOmni cites
  Optimus [17] as its evidence that applying LLM-style parallelism to omni models causes imbalance.
- **Failed suitors**:
  1. Text-LLM frameworks (Megatron-LM): fast TP/PP blocks, but parallelism is written into the model
     code and one uniform plan is applied to encoder and LLM alike → load imbalance, and every new
     modality means touching parallel code.
  2. Multimodal systems (DistMM, DistTrain, Optimus): attack any-to-text efficiency, but still couple
     communication with computation; none trains any-to-any end to end.
- **Villain**: model definition entangled with parallel logic (communication ops live inside model
  blocks; the plan is global rather than per module).
- **Turn / insight**: FSDP, Ulysses SP and EP each act at one narrow boundary of an unmodified model —
  the parameters, the attention input/output, and the expert weight tensors — so all three can be
  attached from outside, per module, through one DeviceMesh.
- **Mechanisms** (container: one slide per component, sibling decks cited):
  ❶ Encoder–foundation–decoder protocol (`lm_encode`, `lm_head`, masked scatter) — answers "any
     modality, anywhere in the sequence".
  ❷ Declarative `ParallelPlan` + global DeviceMesh: FSDP2/HSDP everywhere, EP on expert dim-0 with
     FSDP2 on dim-1 — answers "different modules need different sharding" (FSDP deep dive: deck 70).
  ❸ Ulysses SP: two all-to-alls around attention, async variant overlaps them — answers "long
     multimodal sequences" (deck 68).
  Services outside the model: dynamic batching / packing (deck 51), DCP checkpoints with resharding
  (deck 50). Operator-level MoE overlap instead of pipeline-level schemes (contrast: deck 11).
- **Release**: 30B omni-modal MoE at **over 2,800 tokens/s/GPU** on 128 GPUs, context up to **160K**;
  7B at 192K context with **61.5% MFU**, 72B at 96K with **54.82% MFU**. Versus TorchTitan on the same
  text model and GPUs: **2.3× less memory at 32K** (derived), and runs 64K/128K where TorchTitan OOMs.
  (Not the pain's currency: the 48% idle is a different system and workload; the deck says so.)
- **Limits**: no pipeline parallelism (paper: future work); models still need registration/patch files
  (omni MoE: `__init__.py` + 4 more files); MoE MFU 7-18% in the text-only MoE table; only
  text-only baselines exist (no other any-to-any framework); encoders/decoders frozen in the eval;
  Ulysses needs heads divisible by SP size; mixed sharding forces split optimizers.

## Slide plan (15 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Omni models = encoders + LLM + decoders | prosperity | fig_omni (SVG) |
| 3 | 3D parallelism wastes half the GPU cycles | pain | `%% big: 48%` |
| 4 | In-model parallel code blocks new modalities | villain + suitors | fig_coupled (SVG, 2 panels) |
| 5 | Insight: each strategy touches one boundary | turn | fig_hooks (SVG) |
| 6 | VeOmni puts recipes under unmodified models | overview | fig_stack (SVG) |
| 7 | ❶ Modalities plug in via lm_encode and lm_head | mechanism | fig_protocol (SVG) |
| 8 | ❷ A two-entry plan shards experts on a 2D mesh | mechanism | fig_mesh (SVG) |
| 9 | ❸ Ulysses re-shards attention with two all-to-alls | mechanism | fig_ulysses (SVG) |
| 10 | Batching and checkpoints stay outside too | services | two-col text, decks 50/51 |
| 11 | A 30B omni MoE trains at 2,800+ tokens/s/GPU | release | `%% big:` (tone ours) |
| 12 | Same recipe, 2.3× less memory than TorchTitan | result | chart mem_vs_titan |
| 13 | VeOmni spans model families and accelerators | adoption | two-col text |
| 14 | The cost: no PP, per-model patches, low MoE MFU | limits | two-col text |
| 15 | Summary | | QR arxiv.org/abs/2508.02317 |
