# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/ad31c31051744ea2b53ea97c1417ac71
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties

| Property | Value |
|---|---|
| Technique | FP4 QAT (MXFP4 expert weights + indexer QK path) |
| Category | Low-Precision & Quantization |
| Core Mechanism | Post-training MXFP4 QAT on MoE expert weights and the CSA indexer's QK path; stored as FP4 (E2M1) and dequantized to FP8 (E4M3) for compute (FP8's extra exponent bits make dequantization mathematically lossless), backward uses STE to update FP32 master weights, index scores drop from FP32 to BF16. |
| Impact / Measured Results | 2x faster top-k selector with KV recall held at 99.7%; deployment runs the FP4 weights directly, so training and serving behavior match. |
| Reference | https://arxiv.org/abs/2606.19348 |
| Source | DeepSeek V4 |

## Page body

### Papers & slides (PDF)
- DeepSeek-V4 technical report (PDF): https://arxiv.org/pdf/2606.19348
- Pretraining LLMs with NVFP4 (PDF): https://arxiv.org/pdf/2509.25149

### Public resources
- NVIDIA Research: Pushing intelligence to 4-bit: https://research.nvidia.com/labs/eai/blogs/pushing-intelligence-to-4-bit/ — FP4 weights/activations/KV background
- DeepSeek-V4 paper review: https://andlukyane.com/blog/paper-review-deepseekv4 — where FP4 QAT is applied
- Transformer Engine FP8/FP4 primer: https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html
