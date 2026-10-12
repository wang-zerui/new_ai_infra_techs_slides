# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/cf0641f4eac74fd9bd6c903a15b59ca3
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties

- **Technique**: DeepSeek Sparse Attention (DSA) + Lightning Indexer
- **Category**: Attention & Long Context
- **Core Mechanism**: A lightweight indexer scores history tokens (multi-head ReLU dot-product weighting), top-k (about 2048) picks a KV subset, and sparse attention then runs on top of MLA; migrated through dense-to-sparse continued pretraining.
- **Impact / Measured Results**: Large drop in long-context training and serving cost with base quality on par with V3.1; GLM-5 adopts it wholesale and completed large-scale DSA-based RL.
- **Reference**: https://arxiv.org/abs/2512.02556
- **Source**: DeepSeek V3.2, GLM-5 series

## Body

### Papers & slides (PDF)
- DeepSeek-V3.2 technical report (PDF): https://arxiv.org/pdf/2512.02556
- Native Sparse Attention (PDF): https://arxiv.org/pdf/2502.11089

### Public resources
- DeepSeek-V3.2-Exp in vLLM: fine-grained sparse attention — https://vllm.ai/blog/2025-09-29-deepseek-v3-2 (serving-side walkthrough of DSA)
- Sebastian Raschka — DeepSeek Sparse Attention (architecture gallery) — https://sebastianraschka.com/llm-architecture-gallery/deepseek-sparse-attention/ (illustrated explainer)
- NVIDIA cuDNN: DeepSeek Sparse Attention ops — https://docs.nvidia.com/deeplearning/cudnn/latest/fe-oss-apis/dsa.html (indexer fwd/bwd kernel spec)
