# Notion row (fetched 2026-10-12T19:13:49Z, read-only)

URL: https://app.notion.com/p/d471baa2bcdf452592855568ee99f726
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: DeepEP v2 (EP communication library)
- Category: Frameworks & Platform Stacks
- Core Mechanism: Container entry: DeepSeek's expert-parallel communication library, rewritten in V2. Sub-techniques are the rows whose "Parent Library / Framework" points here: ElasticBuffer unified API, NCCL Gin backend, elastic GPU/CPU buffers, 0-SM PP/CP/Engram, FP8 dispatch + BF16 combine, and network-side traffic isolation.
- Impact / Measured Results: Several times lower SM usage than V1 and scale-up/out domains up to EP2048; FP8 dispatch with IB virtual-lane isolation and adaptive routing.
- Reference: https://github.com/deepseek-ai/DeepEP
- Source: DeepEP

## Body
### Papers & slides (PDF)
- DeepSeek-V3 technical report (PDF): https://arxiv.org/pdf/2412.19437
- Fire-Flyer AI-HPC (PDF): https://arxiv.org/pdf/2408.14158
### Public resources
- DeepEP repo: https://github.com/deepseek-ai/DeepEP — official source and docs
- Azure HPC blog: optimal DeepEP performance: https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/achieving-optimal-performance-for-deepseek-expert-parallelism-deepep-on-azure/4414699
- SGLang expert parallelism guide: https://docs.sglang.ai/advanced_features/expert_parallelism.html — integration in a serving stack
