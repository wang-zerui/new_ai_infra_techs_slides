# Notion row (fetched 2026-10-12T19:13:49Z, read-only)

URL: https://app.notion.com/p/288a723bba1048b2964f7c67a24d4a3a
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: ElasticBuffer unified API + analytic SM/QP budgeting
- Category: Expert-Parallel Communication
- Core Mechanism: Unifies the high-throughput and low-latency dispatch/combine APIs into ElasticBuffer (with a new GEMM layout) and derives the optimal SM and QP counts analytically from the MoE configuration, removing auto-tuning; handle caching reuses routing metadata.
- Impact / Measured Results: Communication SMs drop from 24 to 4-6 at equal or better performance in V3-style training; 726/740 GB/s on SM100 NVLink EP8.
- Parent Library / Framework: DeepEP v2 (https://app.notion.com/p/d471baa2bcdf452592855568ee99f726)
- Reference: https://github.com/deepseek-ai/DeepEP
- Source: DeepEP

## Body
### Papers & slides (PDF)
- DeepSeek-V3 technical report (PDF): https://arxiv.org/pdf/2412.19437
- Fire-Flyer AI-HPC (PDF): https://arxiv.org/pdf/2408.14158
### Public resources
- DeepEP repo: https://github.com/deepseek-ai/DeepEP — ElasticBuffer API
- Azure: tuning DeepEP performance: https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/achieving-optimal-performance-for-deepseek-expert-parallelism-deepep-on-azure/4414699 — SM/QP tuning in practice
- NVIDIA: Hybrid-EP: https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/
