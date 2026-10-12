# Story — deck 59: Attention Residuals (AttnRes / Block AttnRes)

- **Where it lives** (prosperity): every PreNorm LLM adds each layer's output into one residual stream with
  weight 1. Kimi K3 (93 layers, 2.8T MoE) replaces this with Block AttnRes: 8 blocks of 12 layers, 9 depth
  sources counting the embedding; validated first on Kimi Linear 48B / 3B active, 1.4T tokens.
- **Pain number** (betrayal): a standard-residual (PreNorm) model needs **1.25×** the compute to reach the
  validation loss of the same model with Block AttnRes (scaling-law fit over five sizes, 194M–528M active).
  Symptom: baseline per-block output magnitude grows monotonically with depth, and gradients pile up in the
  earliest layers (Fig. 5).
- **Failed suitors**:
  1. Re-weight the recurrence (Highway, ReZero, LayerScale, DeepNorm): each layer still sees only h_{l−1}.
  2. Dense cross-layer access with static weights (DenseFormer): 1.767 vs baseline 1.766, no gain.
  3. Multi-stream mixing (mHC, deck 61): input-dependent, 1.747, but 34d residual I/O per layer vs 3d.
  4. Full AttnRes itself at scale: O(Ld) memory and pipeline traffic.
- **Villain**: depth-wise aggregation with fixed unit weights: h_l = h_1 + Σ f_i(h_i) is a single compressed
  state (an RNN over depth); under PreNorm its magnitude grows O(L), diluting each layer, and no layer can
  selectively retrieve an earlier one.
- **Turn / insight**: depth is a short sequence (L < 1000). The residual is depth-wise *linear* attention with
  all-ones weights; replace it with *softmax* attention over earlier layer outputs, queried by one learned
  vector per layer. Measured: input-dependent softmax weights matter (static mixing 1.749, DenseFormer 1.767,
  softmax 1.737).
- **Mechanisms**:
  ❶ AttnRes op: α_{i→l} = softmax_i(w_lᵀ RMSNorm(v_i)), h_l = Σ α_{i→l} v_i over the embedding and all earlier
     outputs; w_l zero-initialized (answers: selective, content-dependent access).
  ❷ Block AttnRes: sum layers within a block, attend over N ≈ 8 block summaries + the running partial sum;
     O(Ld) → O(Nd) (answers: memory and pipeline traffic).
  ❸ Systems: cross-stage caching under interleaved PP (send only new blocks) + two-phase inference (batched
     inter-block attention, online-softmax merge) + sequence-sharded prefill (answers: training/inference
     overhead).
- **Release**: 1.25× compute advantage; lower loss at all five sizes (−0.020 to −0.026); Kimi Linear 48B:
  GPQA-Diamond +7.5, Math +3.6, HumanEval +3.1, MMLU +1.1 (MMLU-Pro unchanged); overhead < 4% training with
  PP, < 2% inference latency; residual I/O 5.5d per layer vs 34d for mHC.
- **Limits**: Block (1.746) trails Full (1.737) on the 16-layer ablation; input-dependent queries are better
  (1.731) but cost a d×d projection and sequential decoding; needs PP caching and kernel work; prefill block
  cache (15 GB at 128K tokens before sharding); prefers deeper-narrower shapes that cost inference latency;
  no K3-specific ablation published.

## Slide plan (15 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Every layer adds into one residual stream | prosperity | fig_stack_std (SVG) |
| 3 | Fixed residuals cost 1.25× compute | pain | `%% big: 1.25×` |
| 4 | Standard residuals mix depth with fixed 1s | villain | fig_matrix_1 (SVG) |
| 5 | Static depth weights do not help | failed suitors | chart ablation.py |
| 6 | Insight: depth is a short sequence, so attend over it | turn | fig_matrix_2 (SVG) |
| 7 | K3 attends over 9 depth sources per layer | overview | fig_stack_block (SVG) |
| 8 | ❶ One learned query per layer picks its sources | mechanism | fig_op (SVG) |
| 9 | ❷ Blocks shrink depth memory from L to N | mechanism | fig_block (SVG) |
| 10 | ❸ Pipeline stages send only new blocks | mechanism | fig_pipe (SVG) |
| 11 | Block AttnRes lowers loss at every scale | release | chart scaling.py |
| 12 | +7.5 GPQA-Diamond on a 48B MoE | release | chart downstream.py |
| 13 | Overhead: < 4% training, < 2% inference | release / cost | chart io.py |
| 14 | The price: coarser blocks, pipeline plumbing | limits | two-col |
| 15 | Summary | | QR |
