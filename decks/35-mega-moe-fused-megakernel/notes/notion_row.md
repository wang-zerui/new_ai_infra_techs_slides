# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/77562a16a085453aa50e5744e3382f7e
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)
Last edited (Notion metadata): 2026-10-12T19:13:49Z

## Properties
- Technique: Mega MoE fused MegaKernel
- Category: GEMM & MoE Kernels
- Core Mechanism: Fuses EP dispatch, linear1 (FP8 x FP4 / FP8 x FP8), SwiGLU, linear2 and EP combine into one mega-kernel, using symmetric memory so NVLink communication overlaps Tensor-Core compute inside a single kernel.
- Impact / Measured Results: Eliminates kernel boundaries and intermediate buffer write-backs; single-stream overlap is friendlier to numerical reproducibility than dual-stream or micro-batch splitting.
- Parent Library / Framework: DeepGEMM (https://app.notion.com/p/c0133413563f4ee390f9e4fb55027780 = deck 25)
- Reference: https://github.com/deepseek-ai/DeepGEMM
- Source: DeepGEMM

## Body
### Papers & slides (PDF)
- DeepSeek-V4 technical report (PDF) — https://arxiv.org/pdf/2606.19348
- Flux: fine-grained comm/compute overlap (PDF) — https://arxiv.org/pdf/2406.06858
### Public resources
- DeepGEMM repo — https://github.com/deepseek-ai/DeepGEMM (Mega MoE kernel)
- PyTorch SymmetricMemory — https://dev-discuss.pytorch.org/t/pytorch-symmetricmemory-harnessing-nvlink-programmability-with-ease/2798 (the comm-in-kernel primitive it relies on)
- Podcast: DeepSeek V4 and Ascend performance — https://www.youtube.com/watch?v=y2vz0REr5sQ (discussion of Mega MoE fusion)
