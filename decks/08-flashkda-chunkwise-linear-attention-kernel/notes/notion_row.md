# Notion row (read-only copy, fetched 2026-10-12T19:13Z)

URL: https://app.notion.com/p/addf5d5fcfad445081c78ebac0e50d60
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties

- **Technique:** FlashKDA (chunkwise linear-attention kernel)
- **Category:** Attention Kernels
- **Source:** Kimi K3
- **Reference:** https://arxiv.org/abs/2607.24653
- **Core Mechanism:** A CUTLASS-based chunkwise kernel that overlaps intra-chunk computation with cross-chunk
  recurrent state propagation instead of stalling between two alternating phases; shared by training and prefill
  and auto-dispatched as a flash-linear-attention backend.
- **Impact / Measured Results:** Lets linear attention actually saturate the hardware on long training sequences -
  a prerequisite for using KDA in a 2.8T model.

## Body

### Papers & slides (PDF)
- Kimi K3 technical report (PDF): https://arxiv.org/pdf/2607.24653
- Gated DeltaNet (PDF): https://arxiv.org/pdf/2412.06464

### Public resources
- Kimi-Linear repo: https://github.com/MoonshotAI/Kimi-Linear (FlashKDA kernels)
- Kimi K3 blog: https://www.kimi.com/en/blog/kimi-k3
- SemiAnalysis: Kimi K3 deep dive: https://newsletter.semianalysis.com/p/kimi-k3-the-manos-the-mythos-the
  (chunkwise linear attention)
