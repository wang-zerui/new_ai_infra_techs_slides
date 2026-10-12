# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Container entry: a production-grade CUDA operator library for LLM inference hot paths (attention, MoE, GEMM, sampling, normalization, comm/compute fusion), deeply tuned for SM90 (H20) and upstreamed into vLLM / SGLang. Sub-techniques are the rows pointing here.
**After:** Container entry: a production-grade CUDA operator library for LLM inference hot paths (attention, MoE, GEMM, sampling, normalization, comm/compute fusion), deeply tuned for SM90 (H20); attention, router GEMM and MoE kernels are upstreamed into SGLang main, and the API is designed for vLLM / SGLang integration. Sub-techniques are the rows pointing here.
**Why:**
- Upstreaming is documented only for SGLang (Attention #30540/#32304, Router GEMM #30247/#31943, MoE #30541; [L] integration table); for vLLM the README only says the API is "designed for seamless integration" and vLLM appears as a benchmark baseline ([R]).

## Impact / Measured Results
**Before:** In Tencent's large-scale serving: about 30% QPM gain for Hunyuan models and about 17% for DeepSeek-class models; 15.1-48.8% lower end-to-end TPOT for Hy3-FP8 on 8x H20.
**After:** Tencent-reported via MarkTechPost (not in the README): about 30% QPM gain for Hunyuan models and about 17% for DeepSeek models in production serving; LMSYS: 15.1-48.8% lower end-to-end TPOT for Hy3-FP8 on 8x H20 in SGLang at batch 4-64 (3.3% at batch 1).
**Why:**
- The QPM figures appear in no README version; their only source is MarkTechPost's release article quoting Tencent ([MTP]).
- The TPOT range covers batch 4-64 only; batch 1 is -3.3% ([L] Table 7, TP8, FP8 KV, 8K in / 4K out).

## Other properties
none
