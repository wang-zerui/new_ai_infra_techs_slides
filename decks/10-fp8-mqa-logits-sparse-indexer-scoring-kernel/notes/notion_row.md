# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/ac84a3f18a034bac9f7c33b63a8eb6e3
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties

- **Technique**: FP8 MQA logits / sparse indexer scoring kernel
- **Category**: Attention Kernels
- **Core Mechanism**: Weighted-ReLU MQA logits for the lightning indexer: each query takes E4M3 dot products over KV in [start, end), then ReLU, then a head-weighted sum; separate non-paged (prefill) and paged (decode) versions, now extended to an FP4 indexer.
- **Impact / Measured Results**: The prerequisite for DSA sparsity to actually save compute - the indexer itself must not become the new bottleneck.
- **Parent Library / Framework**: https://app.notion.com/p/c0133413563f4ee390f9e4fb55027780 (DeepGEMM row)
- **Reference**: https://github.com/deepseek-ai/DeepGEMM
- **Source**: DeepGEMM, DeepSeek V3.2

## Body

### Papers & slides (PDF)
- DeepSeek-V3.2 technical report (PDF): https://arxiv.org/pdf/2512.02556
- FlashInfer (PDF): https://arxiv.org/pdf/2501.01005

### Public resources
- DeepGEMM repo — https://github.com/deepseek-ai/DeepGEMM (MQA logits kernels)
- vLLM: DeepSeek-V3.2 sparse attention — https://vllm.ai/blog/2025-09-29-deepseek-v3-2 (how indexer scores are served)
- NVIDIA cuDNN DSA ops — https://docs.nvidia.com/deeplearning/cudnn/latest/fe-oss-apis/dsa.html
