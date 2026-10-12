# Notion row (read-only, fetched 2026-10-13)

Page: FP8/FP4/BF16 dense GEMM + UE8M0 scaling layout
URL: https://app.notion.com/p/eedb72bc30e14aa8ba49b28b4344d0b9
Database: 🏗️ Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties

- **Technique**: FP8/FP4/BF16 dense GEMM + UE8M0 scaling layout
- **Category**: GEMM & MoE Kernels
- **Core Mechanism**: "fp8_gemm_{nt,nn,tn,tt}; SM90 supports only the NT layout with FP32 scale
  factors, while SM100 supports all layouts with 4x UE8M0 packed into an int; LHS scale factors must
  be TMA-aligned and transposed."
- **Impact / Measured Results**: "Peaked at 1550 TFLOPS on H800, matching or beating expert-tuned
  libraries with a minimal implementation."
- **Parent Library / Framework**: DeepGEMM (deck 25,
  https://app.notion.com/p/c0133413563f4ee390f9e4fb55027780)
- **Reference**: https://github.com/deepseek-ai/DeepGEMM
- **Source**: DeepGEMM

## Page body

Papers & slides (PDF)
- DeepSeek-V3 (fine-grained FP8 scaling) (PDF) — https://arxiv.org/pdf/2412.19437
- FP8 formats for deep learning (PDF) — https://arxiv.org/pdf/2209.05433

Public resources
- DeepGEMM repo — https://github.com/deepseek-ai/DeepGEMM (layout and scaling requirements)
- Exploring DeepGEMM FP8 kernels — https://kingsleykim.dev/blog/deepgemm/ (Hopper FP8 walkthrough)
