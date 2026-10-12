# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/8f8e53cef45c40b58e4079637b461706
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- Technique: FP8 dispatch + BF16 combine low-precision communication
- Category: Expert-Parallel Communication
- Core Mechanism: Dispatch sends FP8 (data plus scale) and combine returns BF16; the backward dispatch is exactly the forward combine; the Hybrid-EP backend further supports NVFP4.
- Impact / Measured Results: About 3h bytes per token-expert - the premise behind DeepSeek V4's C/B balance point (about 6144 FLOPs/Byte).
- Parent Library / Framework: https://app.notion.com/p/d471baa2bcdf452592855568ee99f726 (DeepEP v2 row)
- Reference: https://github.com/deepseek-ai/DeepEP
- Source: DeepEP, DeepSeek V4

## Body
### Papers & slides (PDF)
- DeepSeek-V3 technical report (PDF) — https://arxiv.org/pdf/2412.19437
- DeepSeek-V4 technical report (PDF) — https://arxiv.org/pdf/2606.19348
### Public resources
- DeepEP repo — https://github.com/deepseek-ai/DeepEP (FP8 dispatch API)
- LMSYS: DeepSeek with PD disaggregation and large-scale EP — https://www.lmsys.org/blog/2025-05-05-large-scale-ep (end-to-end EP deployment)
- vLLM: DeepSeek-V3.2 blog — https://vllm.ai/blog/2025-09-29-deepseek-v3-2
