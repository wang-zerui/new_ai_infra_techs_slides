# Notion row (fetched 2026-10-13, page last edited 2026-10-12T19:13:56Z)

URL: https://app.notion.com/p/8406549e9fd645e49d2f2e90a838f1bd
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- **Technique:** FP8 training (per-tensor / blockwise) and FP8 activation compression
- **Category:** Low-Precision & Quantization
- **Core Mechanism:** TE provides DelayedScaling / blockwise FP8 recipes with automatic autocast; MCore supports FP8 parameter all-gather and fine-grained recompute under FP8; K2 compresses activations in FP8-E4M3.
- **Impact / Measured Results:** FP8 parameter all-gather cuts roughly 50% of that communication; activation memory halved - the default recipe for DeepSeek-V3-class models on H100.
- **Reference:** https://github.com/NVIDIA/TransformerEngine
- **Source:** TransformerEngine, Megatron-LM / MCore, Kimi K2

## Body
### Papers & slides (PDF)
- FP8 formats for deep learning (PDF): https://arxiv.org/pdf/2209.05433
- DeepSeek-V3 (blockwise FP8) (PDF): https://arxiv.org/pdf/2412.19437

### Public resources
- Transformer Engine FP8/FP4 primer: https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html — recipes and autocast
- TE FP8 blockwise scaling docs: https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html — DeepSeek-V3-style quantization
- NVIDIA: FP8 introduction to lower-precision training: https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training
