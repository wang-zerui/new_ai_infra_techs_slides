# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/431ab65669934bd2af65da396bb96f7f
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)
Page last edited: 2026-10-12T19:13:41Z

## Properties
- Technique: Pipeline-aware activation / optimizer offload
- Category: Memory & Offload Scheduling
- Core Mechanism: --fine-grained-activation-offloading offloads activations to CPU per pipeline stage and --offload-optimizer-states offloads optimizer state; K2/K3 likewise combine CPU activation offload with selective recomputation to hold utilization.
- Impact / Measured Results: Sustains stable trillion-scale MoE training on H800 clusters with 2TB of host memory per node.
- Reference: https://arxiv.org/abs/2603.07685
- Source: Megatron-LM / MCore, Kimi K2, Kimi K3

## Body
### Papers & slides (PDF)
- Reducing activation recomputation (PDF) — https://arxiv.org/pdf/2205.05198
- ZeRO-Offload (PDF) — https://arxiv.org/pdf/2101.06840
### Public resources
- NVIDIA: reducing HBM bottlenecks with host offloading — https://developer.nvidia.com/blog/reducing-high-bandwidth-memory-bottlenecks-in-jax-based-llm-training-with-host-offloading (offload vs rematerialization)
- NeMo: activation recomputation — https://docs.nvidia.com/nemo-framework/user-guide/24.07/nemotoolkit/features/optimizations/activation_recomputation.html
- Kimi K2 blog — https://moonshotai.github.io/Kimi-K2/ (CPU offload in a trillion-scale run)
