# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/c0f6478d3ece4c3c8cef0965065c4c0a
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: Network-side optimization: VL traffic isolation / adaptive routing / SM-free RDMA
- Category: Expert-Parallel Communication
- Core Mechanism: InfiniBand virtual lanes (VL) isolate EP traffic from everything else and adaptive routing is enabled; community work (AntGroup) adds Normal-SMFree (decoupling comm-kernel execution from NIC transfer), LL-SBO (overlapping Down GEMM with combine sends) and LL-Layered (rail-optimized forwarding and merging).
- Impact / Measured Results: Stops EP all-to-all from colliding with DP/checkpoint traffic and returns RDMA-path SMs to compute.
- Parent Library / Framework: https://app.notion.com/p/d471baa2bcdf452592855568ee99f726 (DeepEP v2 row)
- Reference: https://github.com/deepseek-ai/DeepEP
- Source: DeepEP

## Body
### Papers & slides (PDF)
- Fire-Flyer AI-HPC: network design (PDF) — https://arxiv.org/pdf/2408.14158
- DeepSeek-V3 technical report (PDF) — https://arxiv.org/pdf/2412.19437
### Public resources
- Azure: optimal DeepEP performance (IB tuning) — https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/achieving-optimal-performance-for-deepseek-expert-parallelism-deepep-on-azure/4414699 (adaptive routing, traffic isolation)
- LMSYS: large-scale EP deployment — https://www.lmsys.org/blog/2025-05-05-large-scale-ep
- DeepEP repo — https://github.com/deepseek-ai/DeepEP
- Sub-page: "讨论笔记：EP 网络侧问题、两种跨节点路径、Normal-SMFree 与 LL-SBO" — https://app.notion.com/p/3ee95647fb2a814b9042d892ea48350e

### Sub-page summary (discussion notes, fetched 2026-10-13; read for orientation, numbers re-verified against primary sources)
- Two bottleneck classes: EP packets squeezed/colliding in the network (fix: VL isolation + adaptive routing) and comm
  kernels occupying SMs / serializing with compute (fix: Normal-SMFree, LL-SBO).
- Single VL → head-of-line blocking behind DP gradient sync and checkpoint traffic; static ECMP → hash collisions,
  worst for low-latency mode that crosses the spine.
- Rail-optimized topology: same-index GPUs share a leaf (1 switch hop); cross-rail goes leaf → spine → leaf.
  Normal mode: same-rail RDMA, then NVLink fan-out, IB traffic deduplicated per node; LL mode: direct RDMA per expert,
  mostly cross-rail.
- Normal-SMFree (PR #347): large RDMA buffer, send phase + recv hook, one SM per channel; H20 numbers
  (EP16 FP8 dispatch 1358 → 765 µs, 44 → 79 GB/s; EP32 3535 → 884 µs); ~270 MB RDMA buffer per rank;
  loses NVLink/RDMA pipelining; maintainer prefers keeping a few SMs; V2 reduced SMs 24 → 4–6.
- LL-SBO (PR #483/#390): per-block_m signals from Down GEMM to combine send; +8–10% at BS 20–56 on H20.
- LL-Layered (PR #500): to-do in the notes.
