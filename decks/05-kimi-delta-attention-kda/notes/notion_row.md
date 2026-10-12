# Notion row (read-only copy, fetched 2026-10-12T19:13Z)

URL: https://app.notion.com/p/0a25e512d50542b08af4e090523e14b2
Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)

## Properties

- **Technique:** Kimi Delta Attention (KDA)
- **Category:** Attention & Long Context
- **Source:** Kimi K3
- **Reference:** https://arxiv.org/abs/2607.24653
- **Core Mechanism:** A Gated DeltaNet refinement: the delta rule plus channel-wise forget gates maintain a
  fixed-size recurrent state; three KDA layers alternate with one gated-MLA layer, fully NoPE (position
  information comes implicitly from gated decay).
- **Impact / Measured Results:** State does not grow with length, so 1M context needs no RoPE interpolation;
  with AttnRes and Stable LatentMoE it gives roughly 2.5x better overall scaling efficiency than K2.

## Body

### Papers & slides (PDF)
- Kimi K3 technical report (PDF): https://arxiv.org/pdf/2607.24653
- Gated DeltaNet (PDF): https://arxiv.org/pdf/2412.06464

### Public resources
- Kimi-Linear repo: https://github.com/MoonshotAI/Kimi-Linear (KDA reference implementation)
- SemiAnalysis: Kimi K3 deep dive: https://newsletter.semianalysis.com/p/kimi-k3-the-manos-the-mythos-the
  (KDA origins, gated delta rule)
- DigitalOcean: hardware-aware Kimi Linear: https://www.digitalocean.com/community/tutorials/kimi-linear-moonshot-ai
  (hands-on tutorial)
