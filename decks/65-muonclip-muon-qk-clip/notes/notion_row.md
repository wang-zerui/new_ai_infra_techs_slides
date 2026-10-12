# Notion row (fetched 2026-10-13, read-only)

URL: https://app.notion.com/p/63cf5bee70464919acbac2e6ed917cdd
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties

- **Technique**: MuonClip (Muon + QK-Clip)
- **Category**: Optimizer & Stability
- **Core Mechanism**: Token-efficient Muon updates plus QK-Clip: after every step the Q/K projection weights are rescaled against threshold tau, killing attention-logit explosion at the source.
- **Impact / Measured Results**: Zero loss spikes across K2's full 15.5T-token pretraining (1T MoE), with no downstream degradation from clipping.
- **Reference**: https://arxiv.org/abs/2507.20534
- **Source**: Kimi K2

## Body

### Papers & slides (PDF)
- Kimi K2 technical report (PDF): https://arxiv.org/pdf/2507.20534
- Muon is scalable for LLM training (PDF): https://arxiv.org/pdf/2502.16982

### Public resources
- Kimi K2 blog: https://moonshotai.github.io/Kimi-K2/ — MuonClip and zero loss spikes
- Keller Jordan: Muon optimizer: https://kellerjordan.github.io/posts/muon — the base optimizer
- Muon is scalable for LLM training (PDF): https://arxiv.org/pdf/2502.16982
