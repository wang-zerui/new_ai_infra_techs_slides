# Notion row — Quantile Balancing (deck 56)

Fetched 2026-10-13 (page last edited 2026-10-12T19:13:40Z) from
https://app.notion.com/p/b59b1fb31a654c95a2957772ad9e4d92
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026). Read-only; not edited.

## Properties

| Property | Value |
|---|---|
| Technique | Quantile Balancing |
| Category | MoE & Routing |
| Core Mechanism | Adjusts the routing distribution directly from quantile statistics instead of aux-loss-free bias updates (unstable at ~1k experts); needs no auxiliary loss. |
| Impact / Measured Results | Keeps load balanced at ~10^3 experts, avoids expert under-training and directly improves EP training throughput. |
| Reference | https://arxiv.org/abs/2607.24653 |
| Source | Kimi K3 |

## Body

### Papers & slides (PDF)
- Kimi K3 technical report (PDF): https://arxiv.org/pdf/2607.24653
- Auxiliary-loss-free load balancing (PDF): https://arxiv.org/pdf/2408.15664

### Public resources
- Kimi K3 blog: https://www.kimi.com/en/blog/kimi-k3 — routing at ~1k experts
- Kimi K3 repo: https://github.com/MoonshotAI/Kimi-K3
- HF blog: MoE load-balancing strategies: https://huggingface.co/blog/NormalUhr/moe-balance — context for why bias updates break down
