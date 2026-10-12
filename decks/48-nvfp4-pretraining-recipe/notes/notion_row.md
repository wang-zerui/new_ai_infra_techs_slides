# Notion row (fetched 2026-10-13, page last edited 2026-10-12T19:13:56Z)

URL: https://app.notion.com/p/eae9504538664595866784566f5d14cd
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties
- **Technique:** NVFP4 pretraining recipe
- **Category:** Low-Precision & Quantization
- **Core Mechanism:** E2M1 with two-dimensional block scaling; stochastic rounding removes gradient-quantization bias, Random Hadamard transforms spread outliers, and sensitive layers stay high precision (bf16_with_nvfp4_mixed).
- **Impact / Measured Results:** <1% loss gap vs FP8 over 10T tokens on a 12B model (~1.5% late in training); a further 1.31-1.73x over FP8 on GB200/GB300.
- **Reference:** https://arxiv.org/abs/2509.25149
- **Source:** TransformerEngine, Megatron-LM / MCore

## Body
### Papers & slides (PDF)
- Pretraining LLMs with NVFP4 (PDF): https://arxiv.org/pdf/2509.25149
- MXFP8 pretraining recipes (PDF): https://arxiv.org/pdf/2506.08027

### Public resources
- NVIDIA: NVFP4 trains with 16-bit precision at 4-bit speed: https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit — the pretraining recipe
- NVIDIA Research: Pushing intelligence to 4-bit: https://research.nvidia.com/labs/eai/blogs/pushing-intelligence-to-4-bit/
- Pretraining LLMs with NVFP4 (paper): https://arxiv.org/html/2509.25149v1
