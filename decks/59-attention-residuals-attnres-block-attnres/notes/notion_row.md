# Notion row — Attention Residuals (AttnRes / Block AttnRes) (deck 59)

Fetched 2026-10-13 (page last edited 2026-10-12T19:13:40Z) from
https://app.notion.com/p/8b86f64fd2d44d4bac732d2321f7722a
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026). Read-only; not edited.

## Properties

| Property | Value |
|---|---|
| Technique | Attention Residuals (AttnRes / Block AttnRes) |
| Category | Model Shape & Depth Scaling |
| Core Mechanism | Per-layer learnable pseudo-queries aggregate the embedding and all preceding layer outputs with softmax-attention weighting instead of equal-weight residual addition; the Block variant groups layers to cut memory from O(Ld) to O(Nd), with two-stage computation and cache-based pipelined communication. |
| Impact / Measured Results | Relieves PreNorm depth dilution - output magnitude and gradients stay uniform with depth; improves every task on Kimi Linear 48B / 1.4T tokens. |
| Reference | https://arxiv.org/abs/2603.15031 |
| Source | Kimi K3 |

## Body

### Papers & slides (PDF)
- Attention Residuals (PDF): https://arxiv.org/pdf/2603.15031
- Hyper-Connections (PDF): https://arxiv.org/pdf/2409.19606

### Public resources
- Attention-Residuals repo: https://github.com/MoonshotAI/Attention-Residuals — code and results
- Moonshot announcement thread: https://x.com/Kimi_Moonshot/status/2033378587878072424 — summary figures
- Kimi-Linear repo: https://github.com/MoonshotAI/Kimi-Linear — validation architecture
