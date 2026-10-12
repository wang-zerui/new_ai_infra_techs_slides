# Story — deck 67: Asymmetric VPP / custom pipeline layout

- **Where it lives** (prosperity): Megatron-Core (flag since v0.13); interleaved 1F1B pipelining (VPP) is the
  default way to shrink the pipeline bubble (bubble ≈ (p−1)/(v·m), worth up to 10% throughput in PTD-P);
  DeepSeek-V3 recipes on 1,024 H100 (PP8) and 256 GB200 (PP4).
- **Pain number** (betrayal): **61** — DeepSeek-V3's decoder count is prime, so no uniform split over
  PP × VPP stages exists; and the stack is heterogeneous (embedding, 3 dense, 58 MoE, MTP, loss).
  (No source measures the imbalance cost directly; the stake is quantified analytically on the results slide.)
- **Failed suitors**:
  1. Uniform VPP (`--num-layers-per-virtual-pipeline-stage`): needs layer count divisible by PP·VPP; counts
     layers, not cost.
  2. Edge knobs (`--decoder-first/last-pipeline-num-layers`, `--account-for-embedding/loss-in-pipeline-split`):
     adjust only the end ranks, and treat embedding/loss as one whole transformer layer.
- **Villain**: the assumption that a pipeline stage is "N identical layers" — stage = count, not cost.
- **Turn / insight**: a pipeline is just an ordered list of stages, each holding some layers; if the user writes
  the list, any layer count and any mix can be balanced by cost (3 dense ≈ 2 MoE; isolate MTP).
- **Mechanisms**:
  ❶ A layout string (E, t, m, L, `|`, `*n`) unrolls into one entry per stage; VPP = stages / PP.
  ❷ Interleaved placement: stage s → PP rank s mod p, chunk s div p (same order as uniform VPP).
  ❸ Cost-aware placement rules + validator (E first, L last, MTP in a rank's last chunk, counts match).
- **Release**: VPP becomes available for the 61-layer model: analytic bubble 5.5% → 1.4% for the H100
  recipe (PP8, m=128, v=4); both DeepSeek-V3 recipes (368 and 1,048 TFLOPS/GPU) ship custom layouts;
  VPP is also the prerequisite for 1F1B EP overlap when PP > 1 (deck 11).
- **Limits**: balance is hand-written (no cost model), more P2P traffic (× v), experimental flag, MTP placement
  restrictions, no encoder layers; FSDP+EP avoids PP stage design altogether (deck 70).

## Slide plan (14 logical slides)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Interleaving shrinks the pipeline bubble v-fold | prosperity | fig_vpp |
| 3 | DeepSeek-V3 has 61 decoder layers: a prime | pain | `%% big: 61` |
| 4 | Uniform VPP splits by count, not by cost | villain | fig_model |
| 5 | Old knobs only nudge the edge stages | failed suitors | two-col |
| 6 | Insight: a pipeline is just a list of stages | turn | boxed sentence |
| 7 | One string places every layer on every chunk | overview | fig_layout_1 |
| 8 | ❶ The string unrolls into one entry per stage | mechanism | fig_grammar |
| 9 | ❷ Stage s lands on rank s mod p | mechanism | fig_layout_2 |
| 10 | ❸ Group light layers, isolate heavy ones | mechanism | two-col |
| 11 | VPP4 cuts the analytic bubble 4×: 5.5% → 1.4% | release | chart bubble |
| 12 | Both DeepSeek-V3 recipes ship custom layouts | adoption | bullets |
| 13 | The cost: balance by hand, v× more P2P | limits | two-col |
| 14 | Summary | | QR |
