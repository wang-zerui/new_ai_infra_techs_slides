# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/3e7e3ebd211b43769e3a8acb1883c7f6
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: HybridEP + FlexDispatcher
- Category: Expert-Parallel Communication
- Core Mechanism: MCore's Flex dispatcher abstracts the DeepEP and HybridEP backends uniformly; HybridEP uses TMA/IBGDA and fewer SMs, with native MNNVL support.
- Impact / Measured Results: Higher MoE all-to-all bandwidth utilization on GB200 NVL72 / B200 / H100 (available by default in MCore v0.15).
- Reference: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md
- Source: Megatron-LM / MCore, DeepEP

## Body
### Papers & slides (PDF)
- Scalable MoE training with Megatron-Core (PDF) — https://arxiv.org/pdf/2603.07685
- DeepSeek-V3 technical report (PDF) — https://arxiv.org/pdf/2412.19437
### Public resources
- NVIDIA: Optimizing MoE communication with Hybrid-EP — https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/ (the primary blog)
- Megatron-Core MoE guide — https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html (Flex dispatcher config)
- GTC session: Scale MoE training with Megatron Core — https://www.nvidia.com/gtc/session-catalog/sessions/gtc26-cwes81540/
