# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/7d97c06866b54afdb590597a7dc39756
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: CUDA Graphs (full-iteration / per-layer / TE graphed callables)
- Category: Kernel Fusion & Runtime
- Core Mechanism: --cuda-graph-impl offers three capture-and-replay modes: full_iteration (FullCudaGraphWrapper), local (per-layer CudaGraphManager) and transformer_engine (make_graphed_callables, FP8- and PP-schedule-aware).
- Impact / Measured Results: Removes kernel-launch overhead in small-kernel-heavy regimes; standard in tuned DeepSeek-V3 configurations on GB200.
- Reference: https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/cuda_graph.html
- Source: Megatron-LM / MCore, TransformerEngine

## Body
### Papers & slides (PDF)
- FlashInfer (CUDA Graph compatible kernels) (PDF) — https://arxiv.org/pdf/2501.01005
- Scalable MoE training with Megatron-Core (PDF) — https://arxiv.org/pdf/2603.07685
### Public resources
- Accelerating PyTorch with CUDA Graphs — https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/ (canonical explainer)
- NVIDIA: Getting started with CUDA Graphs — https://developer.nvidia.com/blog/cuda-graphs
- Megatron-Core MoE guide — https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html (capture modes in training)
