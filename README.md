# Industrial Pretraining Architecture & Infra Tech Landscape (2026): one talk deck per technique

71 conference-style explainer decks, one for each technique in the Notion database *Industrial Pretraining Architecture & Infra Tech Landscape (2026)*: sparse and linear attention, MoE routing, expert-parallel communication, low-precision training, kernels, parallelism and memory scheduling, as used in DeepSeek-V3.2/V4, Kimi K2/K3, GLM-4.5/5, Megatron Core, TransformerEngine, DeepEP, DeepGEMM, HPC-Ops, MoonEP and VeOmni.

Each deck is an ~8-minute talk in the USENIX OSDI/NSDI house style: a measured pain number, one insight, the mechanism walked through with redrawn diagrams, results charts whose numbers trace to the primary source, and a summary with the reference link. Every deck folder holds:

| File | What it is |
|---|---|
| `<slug>.pdf` | the slides (Chrome render) |
| `<slug>.pptx` | editable PowerPoint/Keynote/Google Slides deck with speaker notes |
| `<slug>.md` | slide-markdown source |
| `figures/` | hand-authored SVG diagrams + PDF/PNG renders |
| `charts/` | matplotlib scripts (data at the top) + PDF/PNG renders |
| `notes/` | story plan, sources with the location of every number, the Notion row |

How they are made: [`docs/deck-recipe.md`](docs/deck-recipe.md). Rebuild everything with `bash scripts/build_all.sh` (needs the `usenix-talk-slides` and `ipads-arch-figures` skills, Google Chrome and `uv`).

## Attention & Long Context

| # | Technique | Source | Slides | PPTX | Notion |
|---|---|---|---|---|---|
| 01 | [CSA compressed sparse attention](decks/01-csa-compressed-sparse-attention/) | DeepSeek V4 | [PDF](decks/01-csa-compressed-sparse-attention/01-csa-compressed-sparse-attention.pdf) (14 slides) | [PPTX](decks/01-csa-compressed-sparse-attention/01-csa-compressed-sparse-attention.pptx) | [row](https://app.notion.com/p/e36ba1f7295647b897468dc2b4d30590) |
| 02 | [DeepSeek Sparse Attention (DSA) + Lightning Indexer](decks/02-deepseek-sparse-attention-lightning-indexer/) | DeepSeek V3.2, GLM-5 series | [PDF](decks/02-deepseek-sparse-attention-lightning-indexer/02-deepseek-sparse-attention-lightning-indexer.pdf) (16 slides) | [PPTX](decks/02-deepseek-sparse-attention-lightning-indexer/02-deepseek-sparse-attention-lightning-indexer.pptx) | [row](https://app.notion.com/p/cf0641f4eac74fd9bd6c903a15b59ca3) |
| 03 | [HCA highly-compressed attention + hybrid attention stacking](decks/03-hca-highly-compressed-attention-hybrid-attention/) | DeepSeek V4 | [PDF](decks/03-hca-highly-compressed-attention-hybrid-attention/03-hca-highly-compressed-attention-hybrid-attention.pdf) (14 slides) | [PPTX](decks/03-hca-highly-compressed-attention-hybrid-attention/03-hca-highly-compressed-attention-hybrid-attention.pptx) | [row](https://app.notion.com/p/ceae2901ccd44561a14abacc7f6574bf) |
| 04 | [IndexShare (cross-layer shared indexer)](decks/04-indexshare-cross-layer-shared-indexer/) | GLM-5 series | [PDF](decks/04-indexshare-cross-layer-shared-indexer/04-indexshare-cross-layer-shared-indexer.pdf) (15 slides) | [PPTX](decks/04-indexshare-cross-layer-shared-indexer/04-indexshare-cross-layer-shared-indexer.pptx) | [row](https://app.notion.com/p/e18d1e0afe6f4d3d9c37bfdf815b0738) |
| 05 | [Kimi Delta Attention (KDA)](decks/05-kimi-delta-attention-kda/) | Kimi K3 | [PDF](decks/05-kimi-delta-attention-kda/05-kimi-delta-attention-kda.pdf) (16 slides) | [PPTX](decks/05-kimi-delta-attention-kda/05-kimi-delta-attention-kda.pptx) | [row](https://app.notion.com/p/0a25e512d50542b08af4e090523e14b2) |
| 06 | [Staged context extension + annealing / cooldown](decks/06-staged-context-extension-annealing-cooldown/) | Kimi K2, Kimi K3, GLM-4.5 | [PDF](decks/06-staged-context-extension-annealing-cooldown/06-staged-context-extension-annealing-cooldown.pdf) (13 slides) | [PPTX](decks/06-staged-context-extension-annealing-cooldown/06-staged-context-extension-annealing-cooldown.pptx) | [row](https://app.notion.com/p/ff95a3d4ac0745728425571a7344ee70) |

## Attention Kernels

| # | Technique | Source | Slides | PPTX | Notion |
|---|---|---|---|---|---|
| 07 | [Dynamic Decode Attention (dynamic task scheduling)](decks/07-dynamic-decode-attention/) | HPC-Ops | [PDF](decks/07-dynamic-decode-attention/07-dynamic-decode-attention.pdf) (14 slides) | [PPTX](decks/07-dynamic-decode-attention/07-dynamic-decode-attention.pptx) | [row](https://app.notion.com/p/395fb76ae58944d0ae141c2a047478b6) |
| 08 | [FlashKDA (chunkwise linear-attention kernel)](decks/08-flashkda-chunkwise-linear-attention-kernel/) | Kimi K3 | [PDF](decks/08-flashkda-chunkwise-linear-attention-kernel/08-flashkda-chunkwise-linear-attention-kernel.pdf) (14 slides) | [PPTX](decks/08-flashkda-chunkwise-linear-attention-kernel/08-flashkda-chunkwise-linear-attention-kernel.pptx) | [row](https://app.notion.com/p/addf5d5fcfad445081c78ebac0e50d60) |
| 09 | [FP8 block-sparse prefill attention](decks/09-fp8-block-sparse-prefill-attention/) | HPC-Ops | [PDF](decks/09-fp8-block-sparse-prefill-attention/09-fp8-block-sparse-prefill-attention.pdf) (14 slides) | [PPTX](decks/09-fp8-block-sparse-prefill-attention/09-fp8-block-sparse-prefill-attention.pptx) | [row](https://app.notion.com/p/6ee53460854a4c988e083e2ba08d1f88) |
| 10 | [FP8 MQA logits / sparse indexer scoring kernel](decks/10-fp8-mqa-logits-sparse-indexer-scoring-kernel/) | DeepGEMM, DeepSeek V3.2 | [PDF](decks/10-fp8-mqa-logits-sparse-indexer-scoring-kernel/10-fp8-mqa-logits-sparse-indexer-scoring-kernel.pdf) (13 slides) | [PPTX](decks/10-fp8-mqa-logits-sparse-indexer-scoring-kernel/10-fp8-mqa-logits-sparse-indexer-scoring-kernel.pptx) | [row](https://app.notion.com/p/ac84a3f18a034bac9f7c33b63a8eb6e3) |

## Comm/Compute Overlap

| # | Technique | Source | Slides | PPTX | Notion |
|---|---|---|---|---|---|
| 11 | [1F1B / batch-level EP all-to-all overlap](decks/11-1f1b-batch-level-ep-all-to-all-overlap/) | Megatron-LM / MCore | [PDF](decks/11-1f1b-batch-level-ep-all-to-all-overlap/11-1f1b-batch-level-ep-all-to-all-overlap.pdf) (14 slides) | [PPTX](decks/11-1f1b-batch-level-ep-all-to-all-overlap/11-1f1b-batch-level-ep-all-to-all-overlap.pptx) | [row](https://app.notion.com/p/0e6015aa2a8449ff917574f6139df4c2) |
| 12 | [Ascend · comm/compute fusion and PD-disaggregated scheduling](decks/12-ascend-comm-compute-fusion-and-pd-disaggregated/) | GLM-5 series | [PDF](decks/12-ascend-comm-compute-fusion-and-pd-disaggregated/12-ascend-comm-compute-fusion-and-pd-disaggregated.pdf) (13 slides) | [PPTX](decks/12-ascend-comm-compute-fusion-and-pd-disaggregated/12-ascend-comm-compute-fusion-and-pd-disaggregated.pptx) | [row](https://app.notion.com/p/b1ba69567b7740fdbeead2013a32881c) |
| 13 | [Fused AllReduce + residual + RMSNorm](decks/13-fused-allreduce-residual-rmsnorm/) | HPC-Ops | [PDF](decks/13-fused-allreduce-residual-rmsnorm/13-fused-allreduce-residual-rmsnorm.pdf) (15 slides) | [PPTX](decks/13-fused-allreduce-residual-rmsnorm/13-fused-allreduce-residual-rmsnorm.pptx) | [row](https://app.notion.com/p/2a55d26c7c7a4f858f5816d7ca24b473) |
| 14 | [Wave-based fine-grained communication/compute overlap](decks/14-wave-based-fine-grained-communication-compute/) | DeepSeek V4 | [PDF](decks/14-wave-based-fine-grained-communication-compute/14-wave-based-fine-grained-communication-compute.pdf) (15 slides) | [PPTX](decks/14-wave-based-fine-grained-communication-compute/14-wave-based-fine-grained-communication-compute.pptx) | [row](https://app.notion.com/p/0a5b3fd612d24e81a538b34ef561d06c) |

## Expert-Parallel Communication

| # | Technique | Source | Slides | PPTX | Notion |
|---|---|---|---|---|---|
| 15 | [0-SM PP / CP / Engram communication primitives](decks/15-0-sm-pp-cp-engram-communication-primitives/) | DeepEP | [PDF](decks/15-0-sm-pp-cp-engram-communication-primitives/15-0-sm-pp-cp-engram-communication-primitives.pdf) (14 slides) | [PPTX](decks/15-0-sm-pp-cp-engram-communication-primitives/15-0-sm-pp-cp-engram-communication-primitives.pptx) | [row](https://app.notion.com/p/c3f064d3adbd46a983472f8d7992f155) |
| 16 | [Elastic GPU/CPU hybrid memory buffer](decks/16-elastic-gpu-cpu-hybrid-memory-buffer/) | DeepEP | [PDF](decks/16-elastic-gpu-cpu-hybrid-memory-buffer/16-elastic-gpu-cpu-hybrid-memory-buffer.pdf) (15 slides) | [PPTX](decks/16-elastic-gpu-cpu-hybrid-memory-buffer/16-elastic-gpu-cpu-hybrid-memory-buffer.pptx) | [row](https://app.notion.com/p/f8ca122ace3b41dba7d867f9c2731da3) |
| 17 | [ElasticBuffer unified API + analytic SM/QP budgeting](decks/17-elasticbuffer-unified-api-analytic-sm-qp/) | DeepEP | [PDF](decks/17-elasticbuffer-unified-api-analytic-sm-qp/17-elasticbuffer-unified-api-analytic-sm-qp.pdf) (16 slides) | [PPTX](decks/17-elasticbuffer-unified-api-analytic-sm-qp/17-elasticbuffer-unified-api-analytic-sm-qp.pptx) | [row](https://app.notion.com/p/288a723bba1048b2964f7c67a24d4a3a) |
| 18 | [FP8 dispatch + BF16 combine low-precision communication](decks/18-fp8-dispatch-bf16-combine-low-precision/) | DeepEP, DeepSeek V4 | [PDF](decks/18-fp8-dispatch-bf16-combine-low-precision/18-fp8-dispatch-bf16-combine-low-precision.pdf) (14 slides) | [PPTX](decks/18-fp8-dispatch-bf16-combine-low-precision/18-fp8-dispatch-bf16-combine-low-precision.pptx) | [row](https://app.notion.com/p/8f8e53cef45c40b58e4079637b461706) |
| 19 | [HybridEP + FlexDispatcher](decks/19-hybridep-flexdispatcher/) | Megatron-LM / MCore, DeepEP | [PDF](decks/19-hybridep-flexdispatcher/19-hybridep-flexdispatcher.pdf) (16 slides) | [PPTX](decks/19-hybridep-flexdispatcher/19-hybridep-flexdispatcher.pptx) | [row](https://app.notion.com/p/3e7e3ebd211b43769e3a8acb1883c7f6) |
| 20 | [MoonEP · dynamic redundant experts + online GPU planning + zero-copy static shapes](decks/20-moonep-dynamic-redundant-experts-online-gpu/) | MoonEP | [PDF](decks/20-moonep-dynamic-redundant-experts-online-gpu/20-moonep-dynamic-redundant-experts-online-gpu.pdf) (15 slides) | [PPTX](decks/20-moonep-dynamic-redundant-experts-online-gpu/20-moonep-dynamic-redundant-experts-online-gpu.pptx) | [row](https://app.notion.com/p/dfc27fd2e9d443c9a0a0efd215bdc3da) |
| 21 | [NCCL Gin backend (GPU-initiated RDMA)](decks/21-nccl-gin-backend-gpu-initiated-rdma/) | DeepEP | [PDF](decks/21-nccl-gin-backend-gpu-initiated-rdma/21-nccl-gin-backend-gpu-initiated-rdma.pdf) (16 slides) | [PPTX](decks/21-nccl-gin-backend-gpu-initiated-rdma/21-nccl-gin-backend-gpu-initiated-rdma.pptx) | [row](https://app.notion.com/p/50c0563d46924e00830d20a62dd6cbde) |
| 22 | [Network-side optimization: VL traffic isolation / adaptive routing / SM-free RDMA](decks/22-network-side-optimization-vl-traffic-isolation/) | DeepEP | [PDF](decks/22-network-side-optimization-vl-traffic-isolation/22-network-side-optimization-vl-traffic-isolation.pdf) (15 slides) | [PPTX](decks/22-network-side-optimization-vl-traffic-isolation/22-network-side-optimization-vl-traffic-isolation.pptx) | [row](https://app.notion.com/p/c0f6478d3ece4c3c8cef0965065c4c0a) |

## Frameworks & Platform Stacks

| # | Technique | Source | Slides | PPTX | Notion |
|---|---|---|---|---|---|
| 23 | [Ascend adaptation stack (GLM-5)](decks/23-ascend-adaptation-stack-glm-5/) | GLM-5 series, VeOmni | [PDF](decks/23-ascend-adaptation-stack-glm-5/23-ascend-adaptation-stack-glm-5.pdf) (14 slides) | [PPTX](decks/23-ascend-adaptation-stack-glm-5/23-ascend-adaptation-stack-glm-5.pptx) | [row](https://app.notion.com/p/3eda454335e04ef09f6a0b6991ee5608) |
| 24 | [DeepEP v2 (EP communication library)](decks/24-deepep-v2-ep-communication-library/) | DeepEP | [PDF](decks/24-deepep-v2-ep-communication-library/24-deepep-v2-ep-communication-library.pdf) (16 slides) | [PPTX](decks/24-deepep-v2-ep-communication-library/24-deepep-v2-ep-communication-library.pptx) | [row](https://app.notion.com/p/d471baa2bcdf452592855568ee99f726) |
| 25 | [DeepGEMM (unified tensor-core kernel library)](decks/25-deepgemm-unified-tensor-core-kernel-library/) | DeepGEMM | [PDF](decks/25-deepgemm-unified-tensor-core-kernel-library/25-deepgemm-unified-tensor-core-kernel-library.pdf) (15 slides) | [PPTX](decks/25-deepgemm-unified-tensor-core-kernel-library/25-deepgemm-unified-tensor-core-kernel-library.pptx) | [row](https://app.notion.com/p/c0133413563f4ee390f9e4fb55027780) |
| 26 | [HPC-Ops (Tencent Hunyuan inference operator library)](decks/26-hpc-ops/) | HPC-Ops | [PDF](decks/26-hpc-ops/26-hpc-ops.pdf) (15 slides) | [PPTX](decks/26-hpc-ops/26-hpc-ops.pptx) | [row](https://app.notion.com/p/85f1abe2180f4b86bf1e9ea6d23705ce) |
| 27 | [MoonEP (perfectly balanced EP communication library)](decks/27-moonep/) | MoonEP, Kimi K3 | [PDF](decks/27-moonep/27-moonep.pdf) (15 slides) | [PPTX](decks/27-moonep/27-moonep.pptx) | [row](https://app.notion.com/p/65b197b767bb4d5c9dc17bbce5c0ca13) |
| 28 | [VeOmni (model-centric omni-modal training framework)](decks/28-veomni/) | VeOmni | [PDF](decks/28-veomni/28-veomni.pdf) (15 slides) | [PPTX](decks/28-veomni/28-veomni.pptx) | [row](https://app.notion.com/p/9b9145984b6a433abf69f32d2c06db67) |

## GEMM & MoE Kernels

| # | Technique | Source | Slides | PPTX | Notion |
|---|---|---|---|---|---|
| 29 | [FP8 grouped GEMM (per-tensor / block-wise scaling)](decks/29-fp8-grouped-gemm/) | HPC-Ops | [PDF](decks/29-fp8-grouped-gemm/29-fp8-grouped-gemm.pdf) (13 slides) | [PPTX](decks/29-fp8-grouped-gemm/29-fp8-grouped-gemm.pptx) | [row](https://app.notion.com/p/4b75d7ab1bfc40239d6b9566c5d6fc14) |
| 30 | [FP8/FP4/BF16 dense GEMM + UE8M0 scaling layout](decks/30-fp8-fp4-bf16-dense-gemm-ue8m0-scaling-layout/) | DeepGEMM | [PDF](decks/30-fp8-fp4-bf16-dense-gemm-ue8m0-scaling-layout/30-fp8-fp4-bf16-dense-gemm-ue8m0-scaling-layout.pdf) (14 slides) | [PPTX](decks/30-fp8-fp4-bf16-dense-gemm-ue8m0-scaling-layout/30-fp8-fp4-bf16-dense-gemm-ue8m0-scaling-layout.pptx) | [row](https://app.notion.com/p/eedb72bc30e14aa8ba49b28b4344d0b9) |
| 31 | [Fused FP8 MoE low-latency pipeline](decks/31-fused-fp8-moe-low-latency-pipeline/) | HPC-Ops | [PDF](decks/31-fused-fp8-moe-low-latency-pipeline/31-fused-fp8-moe-low-latency-pipeline.pdf) (14 slides) | [PPTX](decks/31-fused-fp8-moe-low-latency-pipeline/31-fused-fp8-moe-low-latency-pipeline.pptx) | [row](https://app.notion.com/p/4d0436aff16240398bbe1c41ab56d5ad) |
| 32 | [K-axis grouped GEMM (MoE weight backward)](decks/32-k-axis-grouped-gemm-moe-weight-backward/) | DeepGEMM | [PDF](decks/32-k-axis-grouped-gemm-moe-weight-backward/32-k-axis-grouped-gemm-moe-weight-backward.pdf) (13 slides) | [PPTX](decks/32-k-axis-grouped-gemm-moe-weight-backward/32-k-axis-grouped-gemm-moe-weight-backward.pptx) | [row](https://app.notion.com/p/56c25e6b3cbe4425b251e2b59530b802) |
| 33 | [M-axis grouped GEMM (contiguous layout)](decks/33-m-axis-grouped-gemm-contiguous-layout/) | DeepGEMM | [PDF](decks/33-m-axis-grouped-gemm-contiguous-layout/33-m-axis-grouped-gemm-contiguous-layout.pdf) (13 slides) | [PPTX](decks/33-m-axis-grouped-gemm-contiguous-layout/33-m-axis-grouped-gemm-contiguous-layout.pptx) | [row](https://app.notion.com/p/3e1e63852a934fad8b5d53773913bce5) |
| 34 | [Masked grouped GEMM (CUDA Graph friendly)](decks/34-masked-grouped-gemm-cuda-graph-friendly/) | DeepGEMM, DeepEP | [PDF](decks/34-masked-grouped-gemm-cuda-graph-friendly/34-masked-grouped-gemm-cuda-graph-friendly.pdf) (13 slides) | [PPTX](decks/34-masked-grouped-gemm-cuda-graph-friendly/34-masked-grouped-gemm-cuda-graph-friendly.pptx) | [row](https://app.notion.com/p/d8af0edc72e64994bbb502dc9a3bfcea) |
| 35 | [Mega MoE fused MegaKernel](decks/35-mega-moe-fused-megakernel/) | DeepGEMM | [PDF](decks/35-mega-moe-fused-megakernel/35-mega-moe-fused-megakernel.pdf) (16 slides) | [PPTX](decks/35-mega-moe-fused-megakernel/35-mega-moe-fused-megakernel.pptx) | [row](https://app.notion.com/p/77562a16a085453aa50e5744e3382f7e) |
| 36 | [Route GEMM (BF16 x FP32 via dual-BF16 split)](decks/36-route-gemm-bf16-x-fp32-via-dual-bf16-split/) | HPC-Ops | [PDF](decks/36-route-gemm-bf16-x-fp32-via-dual-bf16-split/36-route-gemm-bf16-x-fp32-via-dual-bf16-split.pdf) (15 slides) | [PPTX](decks/36-route-gemm-bf16-x-fp32-via-dual-bf16-split/36-route-gemm-bf16-x-fp32-via-dual-bf16-split.pptx) | [row](https://app.notion.com/p/1ca14bb9a0ce4d439cc9a0686b60a02d) |

## Kernel Fusion & Runtime

| # | Technique | Source | Slides | PPTX | Notion |
|---|---|---|---|---|---|
| 37 | [Ascend · super-operator fusion (MLAPO / Sparse FA / MoE Mega-Fusion)](decks/37-ascend-super-operator-fusion/) | GLM-5 series | [PDF](decks/37-ascend-super-operator-fusion/37-ascend-super-operator-fusion.pdf) (14 slides) | [PPTX](decks/37-ascend-super-operator-fusion/37-ascend-super-operator-fusion.pptx) | [row](https://app.notion.com/p/5926e0ea48474805988cb4d1419bac18) |
| 38 | [CUDA Graphs (full-iteration / per-layer / TE graphed callables)](decks/38-cuda-graphs/) | Megatron-LM / MCore, TransformerEngine | [PDF](decks/38-cuda-graphs/38-cuda-graphs.pdf) (15 slides) | [PPTX](decks/38-cuda-graphs/38-cuda-graphs.pptx) | [row](https://app.notion.com/p/7d97c06866b54afdb590597a7dc39756) |
| 39 | [DeepJIT runtime compilation + PDL + determinism switches](decks/39-deepjit-runtime-compilation-pdl-determinism/) | DeepGEMM | [PDF](decks/39-deepjit-runtime-compilation-pdl-determinism/39-deepjit-runtime-compilation-pdl-determinism.pdf) (13 slides) | [PPTX](decks/39-deepjit-runtime-compilation-pdl-determinism/39-deepjit-runtime-compilation-pdl-determinism.pptx) | [row](https://app.notion.com/p/1ebac796b72b428f9c32afc18e7b5fd4) |
| 40 | [Fused Sampler (decode post-processing fusion)](decks/40-fused-sampler-decode-post-processing-fusion/) | HPC-Ops | [PDF](decks/40-fused-sampler-decode-post-processing-fusion/40-fused-sampler-decode-post-processing-fusion.pdf) (13 slides) | [PPTX](decks/40-fused-sampler-decode-post-processing-fusion/40-fused-sampler-decode-post-processing-fusion.pptx) | [row](https://app.notion.com/p/3070daaed63740e3be901cba2641e152) |
| 41 | [Mega Gate / Mega mHC fused kernels](decks/41-mega-gate-mega-mhc-fused-kernels/) | DeepGEMM, DeepSeek V4 | [PDF](decks/41-mega-gate-mega-mhc-fused-kernels/41-mega-gate-mega-mhc-fused-kernels.pdf) (15 slides) | [PPTX](decks/41-mega-gate-mega-mhc-fused-kernels/41-mega-gate-mega-mhc-fused-kernels.pptx) | [row](https://app.notion.com/p/0bb7a1acbf924329aab698ccc8c5ffee) |
| 42 | [Router / permutation / MLA-RoPE fused kernels](decks/42-router-permutation-mla-rope-fused-kernels/) | Megatron-LM / MCore, TransformerEngine | [PDF](decks/42-router-permutation-mla-rope-fused-kernels/42-router-permutation-mla-rope-fused-kernels.pdf) (15 slides) | [PPTX](decks/42-router-permutation-mla-rope-fused-kernels/42-router-permutation-mla-rope-fused-kernels.pptx) | [row](https://app.notion.com/p/9d82ddbf31244093bc930a84ed679c2d) |
| 43 | [TileLang DSL + batch-invariant deterministic kernels](decks/43-tilelang-dsl-batch-invariant-deterministic/) | DeepSeek V4 | [PDF](decks/43-tilelang-dsl-batch-invariant-deterministic/43-tilelang-dsl-batch-invariant-deterministic.pdf) (15 slides) | [PPTX](decks/43-tilelang-dsl-batch-invariant-deterministic/43-tilelang-dsl-batch-invariant-deterministic.pptx) | [row](https://app.notion.com/p/22c7e2d5f253481981afb6c6ee06e3f5) |

## Low-Precision & Quantization

| # | Technique | Source | Slides | PPTX | Notion |
|---|---|---|---|---|---|
| 44 | [Ascend · W4A8/W8A8 hybrid quantization stack](decks/44-ascend-w4a8-w8a8-hybrid-quantization-stack/) | GLM-5 series | [PDF](decks/44-ascend-w4a8-w8a8-hybrid-quantization-stack/44-ascend-w4a8-w8a8-hybrid-quantization-stack.pdf) (15 slides) | [PPTX](decks/44-ascend-w4a8-w8a8-hybrid-quantization-stack/44-ascend-w4a8-w8a8-hybrid-quantization-stack.pptx) | [row](https://app.notion.com/p/38d6d033f1b943b3b79615c26f4f0a2e) |
| 45 | [FP4 QAT (MXFP4 expert weights + indexer QK path)](decks/45-fp4-qat-mxfp4-expert-weights-indexer-qk-path/) | DeepSeek V4 | [PDF](decks/45-fp4-qat-mxfp4-expert-weights-indexer-qk-path/45-fp4-qat-mxfp4-expert-weights-indexer-qk-path.pdf) (13 slides) | [PPTX](decks/45-fp4-qat-mxfp4-expert-weights-indexer-qk-path/45-fp4-qat-mxfp4-expert-weights-indexer-qk-path.pptx) | [row](https://app.notion.com/p/ad31c31051744ea2b53ea97c1417ac71) |
| 46 | [FP8 training (per-tensor / blockwise) and FP8 activation compression](decks/46-fp8-training-and-fp8-activation-compression/) | TransformerEngine, Megatron-LM / MCore, Kimi K2 | [PDF](decks/46-fp8-training-and-fp8-activation-compression/46-fp8-training-and-fp8-activation-compression.pdf) (16 slides) | [PPTX](decks/46-fp8-training-and-fp8-activation-compression/46-fp8-training-and-fp8-activation-compression.pptx) | [row](https://app.notion.com/p/8406549e9fd645e49d2f2e90a838f1bd) |
| 47 | [MXFP8 micro-scaling training](decks/47-mxfp8-micro-scaling-training/) | TransformerEngine | [PDF](decks/47-mxfp8-micro-scaling-training/47-mxfp8-micro-scaling-training.pdf) (15 slides) | [PPTX](decks/47-mxfp8-micro-scaling-training/47-mxfp8-micro-scaling-training.pptx) | [row](https://app.notion.com/p/02c9479032474c7e8e387fb2e6b7a964) |
| 48 | [NVFP4 pretraining recipe](decks/48-nvfp4-pretraining-recipe/) | TransformerEngine, Megatron-LM / MCore | [PDF](decks/48-nvfp4-pretraining-recipe/48-nvfp4-pretraining-recipe.pdf) (15 slides) | [PPTX](decks/48-nvfp4-pretraining-recipe/48-nvfp4-pretraining-recipe.pptx) | [row](https://app.notion.com/p/eae9504538664595866784566f5d14cd) |
| 49 | [Precision-aware optimizer (BF16/FP8 optimizer states)](decks/49-precision-aware-optimizer/) | Megatron-LM / MCore, TransformerEngine | [PDF](decks/49-precision-aware-optimizer/49-precision-aware-optimizer.pdf) (15 slides) | [PPTX](decks/49-precision-aware-optimizer/49-precision-aware-optimizer.pptx) | [row](https://app.notion.com/p/893ccbb87b354e7e8fedbc0a18ca7117) |

## Memory & Offload Scheduling

| # | Technique | Source | Slides | PPTX | Notion |
|---|---|---|---|---|---|
| 50 | [Distributed checkpointing and elastic-resource training](decks/50-distributed-checkpointing-and-elastic-resource/) | VeOmni, Kimi K2 | [PDF](decks/50-distributed-checkpointing-and-elastic-resource/50-distributed-checkpointing-and-elastic-resource.pdf) (13 slides) | [PPTX](decks/50-distributed-checkpointing-and-elastic-resource/50-distributed-checkpointing-and-elastic-resource.pptx) | [row](https://app.notion.com/p/a065908961604600811834b0c1800744) |
| 51 | [Dynamic batching / sequence packing](decks/51-dynamic-batching-sequence-packing/) | VeOmni, Megatron-LM / MCore, GLM-5 series | [PDF](decks/51-dynamic-batching-sequence-packing/51-dynamic-batching-sequence-packing.pdf) (13 slides) | [PPTX](decks/51-dynamic-batching-sequence-packing/51-dynamic-batching-sequence-packing.pptx) | [row](https://app.notion.com/p/5cfd3adb77f74e0c8d09435a3af5d9b3) |
| 52 | [Fine-grained selective recomputation](decks/52-fine-grained-selective-recomputation/) | Megatron-LM / MCore | [PDF](decks/52-fine-grained-selective-recomputation/52-fine-grained-selective-recomputation.pdf) (15 slides) | [PPTX](decks/52-fine-grained-selective-recomputation/52-fine-grained-selective-recomputation.pptx) | [row](https://app.notion.com/p/2e31e16f18ce45edba1e3d9058756c08) |
| 53 | [Memory-efficient token permutation](decks/53-memory-efficient-token-permutation/) | Megatron-LM / MCore | [PDF](decks/53-memory-efficient-token-permutation/53-memory-efficient-token-permutation.pdf) (15 slides) | [PPTX](decks/53-memory-efficient-token-permutation/53-memory-efficient-token-permutation.pptx) | [row](https://app.notion.com/p/3a91e54ff5c7400e9b2e65e0ba6479f5) |
| 54 | [Pipeline-aware activation / optimizer offload](decks/54-pipeline-aware-activation-optimizer-offload/) | Megatron-LM / MCore, Kimi K2, Kimi K3 | [PDF](decks/54-pipeline-aware-activation-optimizer-offload/54-pipeline-aware-activation-optimizer-offload.pdf) (15 slides) | [PPTX](decks/54-pipeline-aware-activation-optimizer-offload/54-pipeline-aware-activation-optimizer-offload.pptx) | [row](https://app.notion.com/p/431ab65669934bd2af65da396bb96f7f) |

## MoE & Routing

| # | Technique | Source | Slides | PPTX | Notion |
|---|---|---|---|---|---|
| 55 | [Loss-free balance routing + sigmoid gating](decks/55-loss-free-balance-routing-sigmoid-gating/) | GLM-4.5, DeepSeek V3.2, Megatron-LM / MCore | [PDF](decks/55-loss-free-balance-routing-sigmoid-gating/55-loss-free-balance-routing-sigmoid-gating.pdf) (16 slides) | [PPTX](decks/55-loss-free-balance-routing-sigmoid-gating/55-loss-free-balance-routing-sigmoid-gating.pptx) | [row](https://app.notion.com/p/efa722c88bed4a4cb1baa6d8d0a0d6fa) |
| 56 | [Quantile Balancing](decks/56-quantile-balancing/) | Kimi K3 | [PDF](decks/56-quantile-balancing/56-quantile-balancing.pdf) (14 slides) | [PPTX](decks/56-quantile-balancing/56-quantile-balancing.pptx) | [row](https://app.notion.com/p/b59b1fb31a654c95a2957772ad9e4d92) |
| 57 | [Stable LatentMoE](decks/57-stable-latentmoe/) | Kimi K3 | [PDF](decks/57-stable-latentmoe/57-stable-latentmoe.pdf) (14 slides) | [PPTX](decks/57-stable-latentmoe/57-stable-latentmoe.pptx) | [row](https://app.notion.com/p/fafccf6521d44c569d9d5cfad419049f) |
| 58 | [Ultra-sparse MoE (fine-grained routed experts + shared expert)](decks/58-ultra-sparse-moe/) | DeepSeek V4, Kimi K2, GLM-5 series | [PDF](decks/58-ultra-sparse-moe/58-ultra-sparse-moe.pdf) (13 slides) | [PPTX](decks/58-ultra-sparse-moe/58-ultra-sparse-moe.pptx) | [row](https://app.notion.com/p/b2283eba9cf14a8fb2fa3f7b2ba4127f) |

## Model Shape & Depth Scaling

| # | Technique | Source | Slides | PPTX | Notion |
|---|---|---|---|---|---|
| 59 | [Attention Residuals (AttnRes / Block AttnRes)](decks/59-attention-residuals-attnres-block-attnres/) | Kimi K3 | [PDF](decks/59-attention-residuals-attnres-block-attnres/59-attention-residuals-attnres-block-attnres.pdf) (15 slides) | [PPTX](decks/59-attention-residuals-attnres-block-attnres/59-attention-residuals-attnres-block-attnres.pptx) | [row](https://app.notion.com/p/8b86f64fd2d44d4bac732d2321f7722a) |
| 60 | [Deep-and-narrow + more attention heads + partial RoPE](decks/60-deep-and-narrow-more-attention-heads-partial/) | GLM-4.5 | [PDF](decks/60-deep-and-narrow-more-attention-heads-partial/60-deep-and-narrow-more-attention-heads-partial.pdf) (13 slides) | [PPTX](decks/60-deep-and-narrow-more-attention-heads-partial/60-deep-and-narrow-more-attention-heads-partial.pptx) | [row](https://app.notion.com/p/ba32c8b828d748ea9f32a65e15598576) |
| 61 | [mHC: manifold-constrained hyper-connections](decks/61-mhc-manifold-constrained-hyper-connections/) | DeepSeek V4, GLM-5 series | [PDF](decks/61-mhc-manifold-constrained-hyper-connections/61-mhc-manifold-constrained-hyper-connections.pdf) (13 slides) | [PPTX](decks/61-mhc-manifold-constrained-hyper-connections/61-mhc-manifold-constrained-hyper-connections.pptx) | [row](https://app.notion.com/p/99ccc6cd324244c0bf114f19b5440674) |
| 62 | [MTP (multi-token prediction) layer](decks/62-mtp-multi-token-prediction-layer/) | DeepSeek V4, GLM-4.5, GLM-5 series, Megatron-LM / MCore | [PDF](decks/62-mtp-multi-token-prediction-layer/62-mtp-multi-token-prediction-layer.pdf) (15 slides) | [PPTX](decks/62-mtp-multi-token-prediction-layer/62-mtp-multi-token-prediction-layer.pptx) | [row](https://app.notion.com/p/354b71e013554721a039936a8fff8b2a) |

## Optimizer & Stability

| # | Technique | Source | Slides | PPTX | Notion |
|---|---|---|---|---|---|
| 63 | [Anticipatory routing (asynchronous router parameters)](decks/63-anticipatory-routing/) | DeepSeek V4 | [PDF](decks/63-anticipatory-routing/63-anticipatory-routing.pdf) (12 slides) | [PPTX](decks/63-anticipatory-routing/63-anticipatory-routing.pptx) | [row](https://app.notion.com/p/ecc924eb12db4b0fa39609cf42a53369) |
| 64 | [Distributed engineering of Muon at scale](decks/64-distributed-engineering-of-muon-at-scale/) | DeepSeek V4, GLM-5 series | [PDF](decks/64-distributed-engineering-of-muon-at-scale/64-distributed-engineering-of-muon-at-scale.pdf) (13 slides) | [PPTX](decks/64-distributed-engineering-of-muon-at-scale/64-distributed-engineering-of-muon-at-scale.pptx) | [row](https://app.notion.com/p/9069ea0863a741658d052a19a186169b) |
| 65 | [MuonClip (Muon + QK-Clip)](decks/65-muonclip-muon-qk-clip/) | Kimi K2 | [PDF](decks/65-muonclip-muon-qk-clip/65-muonclip-muon-qk-clip.pdf) (15 slides) | [PPTX](decks/65-muonclip-muon-qk-clip/65-muonclip-muon-qk-clip.pptx) | [row](https://app.notion.com/p/63cf5bee70464919acbac2e6ed917cdd) |
| 66 | [QK-Norm](decks/66-qk-norm/) | GLM-4.5 | [PDF](decks/66-qk-norm/66-qk-norm.pdf) (15 slides) | [PPTX](decks/66-qk-norm/66-qk-norm.pptx) | [row](https://app.notion.com/p/319387d19fc64c6d8079c56a5c3b7a69) |

## Parallelism Strategy

| # | Technique | Source | Slides | PPTX | Notion |
|---|---|---|---|---|---|
| 67 | [Asymmetric VPP / custom pipeline layout](decks/67-asymmetric-vpp-custom-pipeline-layout/) | Megatron-LM / MCore | [PDF](decks/67-asymmetric-vpp-custom-pipeline-layout/67-asymmetric-vpp-custom-pipeline-layout.pdf) (14 slides) | [PPTX](decks/67-asymmetric-vpp-custom-pipeline-layout/67-asymmetric-vpp-custom-pipeline-layout.pptx) | [row](https://app.notion.com/p/0b667c3fda7240be99b87c3b35fdcfcd) |
| 68 | [Context Parallelism / Dynamic CP](decks/68-context-parallelism-dynamic-cp/) | Megatron-LM / MCore, TransformerEngine | [PDF](decks/68-context-parallelism-dynamic-cp/68-context-parallelism-dynamic-cp.pdf) (16 slides) | [PPTX](decks/68-context-parallelism-dynamic-cp/68-context-parallelism-dynamic-cp.pptx) | [row](https://app.notion.com/p/82d104d9bcc74fab931db687fed76e81) |
| 69 | [KDA Context Parallelism (KCP)](decks/69-kda-context-parallelism-kcp/) | Kimi K3 | [PDF](decks/69-kda-context-parallelism-kcp/69-kda-context-parallelism-kcp.pdf) (13 slides) | [PPTX](decks/69-kda-context-parallelism-kcp/69-kda-context-parallelism-kcp.pptx) | [row](https://app.notion.com/p/a5e78a14357748deb3032985bef70062) |
| 70 | [Megatron-FSDP / HSDP (with EP support)](decks/70-megatron-fsdp-hsdp-with-ep-support/) | Megatron-LM / MCore | [PDF](decks/70-megatron-fsdp-hsdp-with-ep-support/70-megatron-fsdp-hsdp-with-ep-support.pdf) (15 slides) | [PPTX](decks/70-megatron-fsdp-hsdp-with-ep-support/70-megatron-fsdp-hsdp-with-ep-support.pptx) | [row](https://app.notion.com/p/bbdbb22301864a05bcbc8e4097a6b284) |
| 71 | [MoE Parallel Folding](decks/71-moe-parallel-folding/) | Megatron-LM / MCore | [PDF](decks/71-moe-parallel-folding/71-moe-parallel-folding.pdf) (15 slides) | [PPTX](decks/71-moe-parallel-folding/71-moe-parallel-folding.pptx) | [row](https://app.notion.com/p/ef4d75b8eee841a79ada63b79bff78d8) |

## Notes

- Decks summarise public papers, repositories and documentation; numbers are quoted from those sources (see each deck's `notes/sources.md`). Diagrams are redrawn; no paper figures are reproduced.
- Where a Notion row disagreed with its primary source, the slides follow the source and the difference is listed under *Discrepancies* in `notes/sources.md`; all of them are collected in [`docs/notion-discrepancies.md`](docs/notion-discrepancies.md).
- Every deck folder links to its Notion row; each Notion row embeds the PDF, attaches the PPTX and links back here.
