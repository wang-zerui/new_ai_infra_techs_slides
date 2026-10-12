# Story — deck 06: Staged context extension + annealing / cooldown

- **Where it lives** (prosperity): frontier MoE pretraining runs that ship long windows: Kimi K2
  (1.04T MoE, 32.6B active, 15.5T tokens, 128K window), GLM-4.5 (355B MoE, 32B active, 23T tokens,
  128K window), Kimi K3 (2.8T MoE, 104B active, 1M window, 8× K2's training context).
- **Pain number** (betrayal): attention work per token grows linearly with the window, so a 128K window
  costs **32×** the attention work per token of a 4K window (analytic; K2 and GLM-4.5 both pretrain at 4K
  and ship 128K). K2's own measurement shows attention dominates at 128K: doubling heads 64→128 adds 83%
  inference FLOPs at 128K.
- **Failed suitors**:
  1. Train at the target window from the start: every one of ~16–23T tokens pays the long-window
     attention, and genuinely long documents are scarce, so most long windows would be packed short
     documents (K3 §3.4: long data is scarce and must be upsampled).
  2. Train short and only stretch positions at the end (RoPE rescaling / interpolation): "length alone
     does not confer long-range capability" (K3 §3.4); the model must practise on data whose answers are
     spread across the window.
- **Villain**: a cost/learning mismatch. Per-token cost scales with the window, but almost everything
  the model learns (knowledge, language, code) is learnable in short windows.
- **Turn / insight**: long-range skill is a late, small adaptation: learn knowledge in short windows at
  high LR, then spend the long windows only in the low-LR tail, on a small, upsampled slice of data that
  needs them (≤ 0.5% of tokens at the longest window in K2 and GLM-4.5).
- **Mechanisms**:
  ❶ A decaying LR schedule that creates a low-LR tail (K2: WSD, constant 2e-4 for 10T then cosine to
     2e-5 over 5.5T, then anneal 2e-5→7e-6; GLM-4.5 and K3: cosine). Answers: where in the run to put
     length.
  ❷ A length ladder inside the tail (K2: 4K anneal 400B → 32K 60B; GLM-4.5: 4K → 32K for 1T → 128K
     for 100B; K3: 8K → 64K in pretraining, 256K → 1M in cooldown), each rung fed with upsampled long and
     synthetic long-dependency data. Answers: the model must adapt gradually and needs real long data.
  ❸ Re-fit positions per rung (K2: YaRN 32K→128K; GLM-4.5: RoPE base 10K→1M at 32K; K3: NoPE MLA +
     KDA, nothing to retune). Answers: positional encodings trained at 4K do not cover 128K.
- **Release**: the run-average attention work per token stays at **1.03×** (K2) and **1.44×** (GLM-4.5)
  of a 4K-only run, versus **32×** for a 128K window throughout (analytic, token-weighted from published
  stage budgets). The 128K stage is 0.4% of tokens in both. End-to-end the windows hold up: K2-Instruct
  MRCR 55.0% at 128K; K3 BrowseComp 90.4% with the full 1M window and no context management.
- **Limits**: no report ablates staged vs. unstaged; K3 publishes no stage budgets; K2's last 4× (32K →
  128K) is YaRN, untrained; long data is scarce and must be upsampled or synthesized; the tail still needs
  long-sequence infrastructure (context parallelism); the LR shape is contested (K2 WSD vs. GLM-4.5/K3
  cosine); and the idea does not transfer to RL output length (GLM-4.5: progressive 16K→64K RL lost to a
  single 64K stage, 80.6% vs 83.4% AIME24).

## Slide plan (15 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Frontier MoEs pretrain at 4–8K, serve 128K–1M | prosperity | chart_windows |
| 3 | Long windows multiply attention work per token | pain | `%% big: 32×` |
| 4 | Training long throughout or never both fail | failed suitors / villain | fig_statusquo (SVG) |
| 5 | Insight: long-range skill is a late, small adaptation | turn | boxed sentence (build) |
| 6 | Three runs, one shape: short bulk, long tail | overview | fig_ladder (SVG) |
| 7 | ❶ The LR schedule decays into a low-LR tail | mechanism | chart_lr |
| 8 | ❷ The length ladder fills the last few percent | mechanism | chart_timeline |
| 9 | ❸ Each rung re-fits positions, or needs none | mechanism | fig_positions (SVG) |
| 10 | Staging keeps attention work near a 4K run | release | chart_span (analytic) |
| 11 | The trained windows hold up end to end | result | two-col numbers |
| 12 | Cosine beat WSD once each was tuned | result / nuance | text + callout |
| 13 | The cost: data, infra, untested ablations | limits | two-col |
| 14 | Summary | | QR |
