# Notion row (fetched 2026-10-13, read-only copy)

URL: https://app.notion.com/p/65b197b767bb4d5c9dc17bbce5c0ca13
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)
Page last edited: 2026-10-12T19:13:41Z

## Properties

- **Technique:** MoonEP (perfectly balanced EP communication library)
- **Category:** Frameworks & Platform Stacks
- **Core Mechanism:** Container entry: Moonshot's MIT-licensed EP communication library released with Kimi K3, aiming for identical compute per rank under any routing skew. Sub-techniques: dynamic redundant-expert balancing, online GPU planning kernel, zero-copy static shapes.
- **Impact / Measured Results:** Static shapes and no per-layer host sync; communication buffer drops from a worst-case S x K x R to a fixed S x K; increasingly outperforms DeepEP v2 as routing imbalance grows on H20/EP=8.
- **Reference:** https://github.com/MoonshotAI/MoonEP
- **Source:** MoonEP, Kimi K3

## Body

### Papers & slides (PDF)
- Kimi K3 technical report (PDF): https://arxiv.org/pdf/2607.24653
- DeepSeek-V3 (EP baseline) (PDF): https://arxiv.org/pdf/2412.19437

### Public resources
- MoonEP repo: https://github.com/moonshotAI/moonep — MIT-licensed source, benchmarks vs DeepEP
- MarkTechPost: Moonshot open-sources MoonEP: https://www.marktechpost.com/2026/07/29/moonshot-ai-open-sources-moonep-a-perfectly-balanced-expert-parallelism-library-for-moe-training/
- Hacker News discussion: https://news.ycombinator.com/item?id=49071019

## Child technique row
- Deck 20: MoonEP · dynamic redundant experts + online GPU planning + zero-copy static shapes
  (https://app.notion.com/p/dfc27fd2e9d443c9a0a0efd215bdc3da), whose "Parent Library / Framework" points here.
