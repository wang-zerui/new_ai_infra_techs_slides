# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/ff95a3d4ac0745728425571a7344ee70
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties

- **Technique**: Staged context extension + annealing / cooldown
- **Category**: Attention & Long Context
- **Core Mechanism**: Main pretraining on short sequences (K2: 4k->32k->128k with YaRN; K3: 8k->64k), stretched to 256k->1M only during cooldown / mid-training, with constant LR plus cosine decay and final annealing.
- **Impact / Measured Results**: Concentrates expensive long-context compute on a small tail of tokens; this is how K3 gets its 1M context.
- **Reference**: https://arxiv.org/abs/2607.24653
- **Source**: Kimi K2, Kimi K3, GLM-4.5

## Body

### Papers & slides (PDF)
- Kimi K3 technical report (PDF): https://arxiv.org/pdf/2607.24653
- Kimi K2 technical report (PDF): https://arxiv.org/pdf/2507.20534

### Public resources
- Kimi K2 blog: https://moonshotai.github.io/Kimi-K2/ — 15.5T-token pretraining recipe
- Kimi K3 blog: https://www.kimi.com/en/blog/kimi-k3 — 1M context staging
- GLM-4.5 blog: https://z.ai/blog/glm-4.5 — mid-training and cooldown
