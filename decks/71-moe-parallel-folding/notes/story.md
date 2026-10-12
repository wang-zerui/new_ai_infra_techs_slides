# Story — deck 71: MoE Parallel Folding

- **Where it lives** (prosperity): Megatron-Core's MoE stack; every large fine-grained MoE recipe in the
  Megatron-Core report (DeepSeek-V3 on 1,024 H100 and 256 GB200 uses folding with expert TP = 1, EP64).
- **Pain number** (betrayal): with one shared mapping, the best Megatron-Core configuration for a
  fine-grained MoE (Mixtral-8x22B-G8T8, 128 H100) reaches only **17.1% MFU**.
- **Failed suitors**:
  1. One mapping with high TP: attention is happy, but experts are sliced into thin shards; ETP
     communication dominates (> 70% of MoE latency once ETP×EP leaves the node, fine-grained).
  2. EP carved out of DP (EP ⊆ DP): EP is capped by DP (EP ≤ 8 on a 256-GPU TP4·CP2·DP8·PP4 job),
     CP8+EP8 needs 64 GPUs, and the all-to-all leaves NVLink (5–10× less bandwidth).
- **Villain**: the assumption that every layer of a block shares one parallel mapping
  (World = TP×CP×DP×PP with EP ⊆ DP) — the dense-sparse mismatch.
- **Turn / insight**: an MoE layer computes every token independently; whatever way attention split the
  tokens (batch, sequence, TP-SP), each GPU just holds a bag of tokens. So the MoE layer may re-factor the
  same GPUs any way it wants; only PP has to agree.
- **Mechanisms**:
  ❶ Two rank grids over the same GPUs (attention TP×CP×DP×PP, MoE ETP×EP×EDP×PP, PP shared) — answers EP ≤ DP.
  ❷ A reshape, not a collective, at the boundary + a token-level dispatcher that is agnostic to the
     attention mapping (A2A-V over EP, AG-V/RS-V over ETP) — answers "how do tokens move between grids".
  ❸ Separate gradient/optimizer groups (expert grads reduce over EDP, scaled EDP/DP) — answers correctness.
- **Release**: MFU 17.1% → **28.8%** on the fine-grained model (1.68×, derived); +2.8 to +3.7 points on
  the three others; gap widens with scale (25.5% vs 13.4% at 1,024 GPUs); FP8 631.7 vs 575.1 TFLOPS.
- **Limits**: PP must be identical; ETP defaults to TP (set it to 1); a folded EP larger than the NVLink
  domain still pays cross-node all-to-all (pair with EP overlap, deck 11); sub-sequence token dropping;
  the dispatcher must support dynamic shapes.

## Slide plan (15 logical slides)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Attention and experts want opposite parallelism | prosperity | fig_needs |
| 3 | Fine-grained MoE stalls at 17% MFU | pain | `%% big: 17.1%` |
| 4 | Sharing one mapping caps EP at DP | villain | fig_fold_1 (64-rank grids) |
| 5 | Neither knob setting fixes both layers | failed suitors | two-col text |
| 6 | Insight: experts see a bag of tokens | turn | boxed sentence |
| 7 | Folding gives MoE its own grid on the same GPUs | overview | fig_fold_2 |
| 8 | ❶ Two rank grids tile the same GPUs | mechanism | fig_ranks (8 GPUs) |
| 9 | ❷ A reshape, not a collective, joins the grids | mechanism | fig_flow |
| 10 | ❸ Expert gradients reduce over their own group | mechanism | text |
| 11 | Fine-grained MoE: 17.1% → 28.8% MFU | release | chart mfu |
| 12 | At 1,024 GPUs: 25.5% vs 13.4% MFU | release | chart scaling |
| 13 | DeepSeek-V3 production recipes use folding | adoption | bullets |
| 14 | The cost: one PP, explicit ETP, NVLink-sized EP | limits | two-col |
| 15 | Summary | | QR |
