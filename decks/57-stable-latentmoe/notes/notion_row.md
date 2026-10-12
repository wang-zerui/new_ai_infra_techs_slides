# Notion row — Stable LatentMoE (deck 57)

Fetched 2026-10-13 (page last edited 2026-10-12T19:13:40Z) from
https://app.notion.com/p/fafccf6521d44c569d9d5cfad419049f
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026). Read-only; not edited.

## Properties

| Property | Value |
|---|---|
| Technique | Stable LatentMoE |
| Category | MoE & Routing |
| Core Mechanism | Experts down-project into a latent space (an MLA analogue), with SiTU-GLU to suppress activation blow-up and RMSNorm on routed experts for numerical stability. |
| Impact / Measured Results | Trains stably with 16 of 896 routed experts active per token (~56x sparsity); the key enabler of K3's width scaling. |
| Reference | https://arxiv.org/abs/2607.24653 |
| Source | Kimi K3 |

## Body

### Papers & slides (PDF)
- Kimi K3 technical report (PDF): https://arxiv.org/pdf/2607.24653
- DeepSeek-V2 / MLA (PDF): https://arxiv.org/pdf/2405.04434

### Public resources
- Kimi K3 blog: https://www.kimi.com/en/blog/kimi-k3 — architecture overview
- Kimi K3 repo: https://github.com/MoonshotAI/Kimi-K3
- SemiAnalysis: Kimi K3 deep dive: https://newsletter.semianalysis.com/p/kimi-k3-the-manos-the-mythos-the — width scaling and latent experts
