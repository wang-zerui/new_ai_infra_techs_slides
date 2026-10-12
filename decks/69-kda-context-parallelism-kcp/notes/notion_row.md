# Notion row (read-only copy, fetched 2026-10-12T19:13Z)

URL: https://app.notion.com/p/a5e78a14357748deb3032985bef70062
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties

- **Technique:** KDA Context Parallelism (KCP)
- **Category:** Parallelism Strategy
- **Source:** Kimi K3
- **Reference:** https://arxiv.org/abs/2607.24653
- **Core Mechanism:** Decomposes each sequence chunk into two locally computable terms - the cumulative transition
  applied to the entry state, and the state generated locally from zero - so only fixed-size recurrent states need an
  all-gather.
- **Impact / Measured Results:** Communication is independent of sequence length (softmax-attention CP must ship KV
  blocks that grow with length); compute scales linearly to 1M context.

## Body

### Papers & slides (PDF)
- Kimi K3 technical report (PDF): https://arxiv.org/pdf/2607.24653
- Ring Attention (PDF): https://arxiv.org/pdf/2310.01889

### Public resources
- Kimi K3 repo: https://github.com/MoonshotAI/Kimi-K3 (architecture and training notes)
- Kimi K3 blog: https://www.kimi.com/en/blog/kimi-k3
- SemiAnalysis: Kimi K3 deep dive: https://newsletter.semianalysis.com/p/kimi-k3-the-manos-the-mythos-the
  (linear-attention state and parallelism)
