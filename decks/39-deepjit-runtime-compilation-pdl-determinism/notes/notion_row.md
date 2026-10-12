# Notion row (read-only, fetched 2026-10-13)

Page: DeepJIT runtime compilation + PDL + determinism switches
URL: https://app.notion.com/p/1ebac796b72b428f9c32afc18e7b5fd4
Database: 🏗️ Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties

- **Technique**: DeepJIT runtime compilation + PDL + determinism switches
- **Category**: Kernel Fusion & Runtime
- **Core Mechanism**: "All kernels are JIT-compiled at runtime (no CUDA compilation at install time)
  through a low-CPU-overhead C++ JIT module with on-disk cache; runtime knobs include set_num_sms /
  set_tc_util / set_pdl and use_deterministic_algorithms."
- **Impact / Measured Results**: "Per-shape configuration and SM budget caps let GEMMs coexist with
  communication kernels; determinism switches serve reproducible training."
- **Parent Library / Framework**: DeepGEMM (deck 25,
  https://app.notion.com/p/c0133413563f4ee390f9e4fb55027780)
- **Reference**: https://github.com/deepseek-ai/DeepGEMM
- **Source**: DeepGEMM

## Page body

Papers & slides (PDF)
- TileLang (PDF) — https://arxiv.org/pdf/2504.17577
- DeepSeek-V3 technical report (PDF) — https://arxiv.org/pdf/2412.19437

Public resources
- DeepGEMM repo — https://github.com/deepseek-ai/DeepGEMM (DeepJIT, PDL and determinism switches)
- Exploring DeepGEMM FP8 kernels — https://kingsleykim.dev/blog/deepgemm/ (JIT design notes)
