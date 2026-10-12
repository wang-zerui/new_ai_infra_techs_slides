# Notion row (fetched 2026-10-12T19:13:52Z, read-only)

URL: https://app.notion.com/p/85f1abe2180f4b86bf1e9ea6d23705ce
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties

- **Technique**: HPC-Ops (Tencent Hunyuan inference operator library)
- **Category**: Frameworks & Platform Stacks
- **Core Mechanism**: Container entry: a production-grade CUDA operator library for LLM inference hot paths
  (attention, MoE, GEMM, sampling, normalization, comm/compute fusion), deeply tuned for SM90 (H20) and
  upstreamed into vLLM / SGLang. Sub-techniques are the rows pointing here.
- **Impact / Measured Results**: In Tencent's large-scale serving: about 30% QPM gain for Hunyuan models and
  about 17% for DeepSeek-class models; 15.1-48.8% lower end-to-end TPOT for Hy3-FP8 on 8x H20.
- **Reference**: https://github.com/Tencent/hpc-ops
- **Source**: HPC-Ops

## Body

### Papers & slides (PDF)
- FlashInfer: customizable attention engine (PDF) — https://arxiv.org/pdf/2501.01005
- FlashAttention-3 (PDF) — https://arxiv.org/pdf/2407.08608

### Public resources
- HPC-Ops repo — https://github.com/Tencent/hpc-ops — official source
- LMSYS: HPC-Ops × SGLang — https://www.lmsys.org/blog/2026-08-07-hpc-ops-sglang/ — integration + benchmark blog
- MarkTechPost: Tencent Hunyuan releases HPC-Ops —
  https://www.marktechpost.com/2026/01/27/tencent-hunyuan-releases-hpc-ops-a-high-performance-llm-inference-operator-library/

## Child rows (from data/manifest.json, Source = HPC-Ops)
07 Dynamic Decode Attention · 09 FP8 block-sparse prefill attention · 13 Fused AllReduce + residual + RMSNorm ·
29 FP8 grouped GEMM · 31 Fused FP8 MoE low-latency pipeline · 36 Route GEMM (BF16 x FP32 via dual-BF16 split) ·
40 Fused Sampler
