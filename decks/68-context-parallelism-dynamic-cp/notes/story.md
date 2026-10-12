# Story — deck 68: Context Parallelism / Dynamic CP

- **Where it lives** (prosperity): Megatron-Core + Transformer Engine, every GPT-path model (dense and MoE);
  long-context and reasoning/RL training at 16K-64K+ tokens, where attention (SDPA) reaches 69% of FLOPs at 64K
  and activation memory grows about linearly with sequence length.
- **Pain number** (betrayal): the usual way to fit a long sequence, full activation recomputation, adds
  **~30%** training time (Megatron CP guide).
- **Failed suitors**:
  1. Full recomputation: fits memory, pays ~30% for a second forward pass.
  2. More TP: shards GEMMs until they are too short to hide TP communication; TP32 at 32K runs at ~0.85x of
     the full-recompute baseline (CP guide Fig. 2, read from the chart).
  3. (Megatron) sequence parallelism only splits LayerNorm/Dropout activations and is tied to the TP degree.
- **Villain**: memory knobs that cut along the hidden dimension (TP) or along time (recompute) pay for every
  op, although only one op, attention, couples tokens.
- **Turn / insight**: every module except attention is token-local, so the sequence can be split with no
  change to them; attention only needs the other shards' K/V, which can stream in a ring underneath the
  attention compute.
- **Mechanisms** (wave 1, static CP):
  ❶ Zigzag (load-balanced) chunking: 2×CP chunks, rank r holds r and 2CP−1−r — answers the causal-mask
     imbalance (contiguous split gives the last rank 7× the first rank's work at CP=4, analytic).
  ❷ Ring KV exchange (P2P) overlapped with blockwise attention; backward re-circulates KV and dKV —
     answers "attention needs all KV" without storing the full KV.
  ❸ Comm backends + hierarchical CP (p2p / all_gather / a2a / a2a+p2p) — answers topology: a2a inside
     NVLink, ring across nodes.
- **Wave 2 (dark moment again)**: variable-length packed batches: a static CP size is set by the longest
  sequence; short sequences are over-sharded and their ring traffic is exposed; equal-token micro-batches
  carry unequal S² attention work (DP imbalance).
- **Turn 2**: resizing CP moves no parameters or optimizer state, only token slices and the CP group.
  ❹ Dynamic CP: pre-built CP groups (powers of two up to dp×cp), per-micro-batch cp_size chosen by an async
     cost-model/solver/simulator scheduler.
- **Release**: static CP: TP8CP4 ≈1.2x vs full recompute at 32K (BF16 MHA), ≈1.3x with FP8+GQA (same currency
  as the ~30% pain: speed relative to full recompute). Dynamic CP: 1.48x / 1.25x TFLOPS/GPU vs packing-only
  on Llama-13B (GitHub / CommonCrawl), >35% end-to-end at multi-thousand-GPU scale.
- **Limits**: ring must be hidden by attention compute (short chunks / IB spans expose it); seq length must
  be divisible by 2×CP; CP does not shard weights; Dynamic CP: power-of-2 groups, no CUDA Graphs,
  FlashAttention preferred (cuDNN recompiles on every shape), no MLA yet, FSDP+PP unsupported.

## Slide plan (16 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Long sequences make attention dominate | prosperity | fig_hook (SVG) |
| 3 | Recomputing to fit long context costs ~30% | pain | `%% big: ~30%` |
| 4 | Both memory knobs slow training down | failed suitors | fig_statusquo (SVG) |
| 5 | Insight: only attention mixes tokens | turn | boxed sentence (build) |
| 6 | CP splits every activation along the sequence | overview | fig_overview (SVG) |
| 7 | ❶ Zigzag chunks balance the causal triangle | mechanism | fig_zigzag_1/2 (SVG build) |
| 8 | ❷ KV circulates in a ring under attention | mechanism | fig_ring (SVG) |
| 9 | ❸ Hierarchical CP matches comm to topology | mechanism | fig_hier (SVG) |
| 10 | At 32K, TP8 + CP4 runs ≈1.2× faster | release 1 | chart cp_speedup |
| 11 | Packed variable-length batches break static CP | wave-2 dark moment | fig_dcp_problem (SVG) |
| 12 | ❹ Dynamic CP picks the CP size per micro-batch | turn 2 + mechanism | fig_dcp_fix (SVG) |
| 13 | An async scheduler plans every micro-batch | mechanism | fig_dcp_sched (SVG) |
| 14 | Dynamic CP lifts throughput up to 1.48× | release 2 | chart dcp_tflops |
| 15 | The cost: comm must hide under attention | limits | two-col |
| 16 | Summary | | QR |
