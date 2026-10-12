# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/e18d1e0afe6f4d3d9c37bfdf815b0738
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties

- **Technique**: IndexShare (cross-layer shared indexer)
- **Category**: Attention & Long Context
- **Core Mechanism**: Every four sparse-attention layers reuse the top-k selection of a single indexer, avoiding repeated indexing computation.
- **Impact / Measured Results**: 2.9x lower per-token FLOPs at 1M context; the improved MTP adds up to 20% more accepted length in speculative decoding.
- **Reference**: https://arxiv.org/abs/2603.12201
- **Source**: GLM-5 series

## Body

### Papers & slides (PDF)
- IndexShare (PDF): https://arxiv.org/pdf/2603.12201
- DeepSeek-V3.2 technical report (PDF): https://arxiv.org/pdf/2512.02556

### Public resources
- GLM-5 blog — https://z.ai/blog/glm-5 (official announcement)
- GLM-5 GitHub — https://github.com/zai-org/GLM-5 (model cards and reports)
- DSA background explainer — https://sebastianraschka.com/llm-architecture-gallery/deepseek-sparse-attention/ (what the shared indexer does)
