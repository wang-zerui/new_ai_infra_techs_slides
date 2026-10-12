# Notion row (fetched 2026-10-13, page last edited 2026-10-12T19:13:56Z)

URL: https://app.notion.com/p/02c9479032474c7e8e387fb2e6b7a964
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- **Technique:** MXFP8 micro-scaling training
- **Category:** Low-Precision & Quantization
- **Core Mechanism:** Every 32 contiguous values share one E8M0 (pure power-of-two) scale, with E4M3 for weights/activations/gradients; natively supported by Blackwell Tensor Cores, requires keeping both transposed and non-transposed copies.
- **Impact / Measured Results:** 1.28-1.37x over BF16 on B200 (up to 2x on GEMMs), loss curves indistinguishable from BF16, and covers more layers than per-tensor FP8.
- **Reference:** https://arxiv.org/abs/2506.08027
- **Source:** TransformerEngine

## Body
### Papers & slides (PDF)
- Recipes for pre-training LLMs with MXFP8 (PDF): https://arxiv.org/pdf/2506.08027
- FP8 formats for deep learning (PDF): https://arxiv.org/pdf/2209.05433

### Public resources
- NVIDIA: per-tensor and per-block scaling strategies for FP8: https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training — how MXFP8 works
- NVIDIA: FP8 introduction: https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training — microscaling formats on Blackwell
- Recipes for pre-training LLMs with MXFP8: https://arxiv.org/html/2506.08027v1
