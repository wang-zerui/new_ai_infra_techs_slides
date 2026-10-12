# Story — deck 51: Dynamic batching / sequence packing

- **Where it lives** (prosperity): every modern training mix has variable-length samples: multimodal
  samples with different image/video token counts, long-context stages (GLM-5 extends context from
  4K to 200K in mid-training), RL traces from hundreds to tens of thousands of tokens. VeOmni (child of
  deck 28) ships dynamic batching + packing; Megatron-Core ships the THD packed format and Dynamic-CP.
- **Pain number** (betrayal): padding a batch of 4K, 8K and 32K sequences to 32K wastes **~60%** of
  computation and memory (Megatron-Core MoE report §6.4).
- **Failed suitors**:
  1. Pad to the longest sample: the [B, S_max] rectangle computes and stores pad tokens.
  2. Concatenate naively: no padding, but attention and positions leak across sample boundaries.
  3. Fixed sample count per micro-batch: packed lengths differ per rank, so ranks finish at different
     times and wait at the gradient sync (VeOmni docs).
- **Villain**: the dense [batch, seq_len] tensor shape assumes every sample has the same length.
- **Turn / insight**: only attention needs to know where a sample starts and ends; every other op is
  per token. Flatten the micro-batch into one token stream and hand attention cumulative offsets
  (cu_seqlens).
- **Mechanisms**:
  ❶ Packing: concatenate, restart `position_ids` per sample, derive `cu_seqlens`, mask the first label of
     each later sample; varlen FlashAttention computes block-diagonal causal attention without a mask tensor.
  ❷ Dynamic batch size: a buffer (≥ 500 samples by default) fills each micro-batch to a token budget by
     first-fit, counting total or loss-carrying tokens, with a 1.5× physical cap; ranks get equal tokens.
  ❸ THD with SP/CP: the packed stream is padded to a multiple of the SP/CP size and split; Megatron
     keeps padded + unpadded cu_seqlens; Dynamic-CP picks the CP size per micro-batch (deck 68).
- **Release**: packing cuts memory **40–60%** and lifts throughput **1.5–2×** vs padding (RL with thinking
  models, Megatron-Core); balancing packs with Dynamic-CP adds **1.48×** (GitHub) and **1.25×**
  (CommonCrawl) TFLOPS/GPU on Llama 13B, and **>35%** end to end at multi-thousand-GPU scale.
- **Limits**: equal tokens ≠ equal FLOPs (attention is quadratic), so DP imbalance remains without a
  workload-aware scheduler; needs varlen kernels; static-shape features (torch.compile, CUDA graphs)
  need `pad_to_length`; loss must be averaged per valid token; dyn_bsz changes samples per step
  (VeOmni recommends it off for SFT); VeOmni's batching buffer is not checkpointed yet.

## Slide plan (13 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Real batches mix 4K and 32K samples | prosperity | fig_lengths (SVG) |
| 3 | Padding wastes most of the batch | pain | `%% big: ~60%` |
| 4 | Padding wastes FLOPs; concat leaks attention | villain + suitors | fig_padding (SVG, 2 panels) |
| 5 | Insight: only attention needs sample boundaries | turn | fig_ops (SVG) |
| 6 | Pack, budget, then shard the token stream | overview | fig_pipeline (SVG) |
| 7 | ❶ cu_seqlens turn one stream into four samples | mechanism | fig_pack (SVG) |
| 8 | ❷ Fill each batch to a token budget | mechanism | fig_dynbsz (SVG) |
| 9 | ❸ A packed stream splits cleanly across SP ranks | mechanism | fig_thd (SVG) |
| 10 | Packing: 1.5–2× throughput, 40–60% less memory | release | chart pack_gain |
| 11 | Balancing the packs adds up to 1.48× | result | chart dyncp |
| 12 | The cost: kernels, static shapes, uneven FLOPs | limits | two-col |
| 13 | Summary | | QR arxiv.org/abs/2508.02317 |
