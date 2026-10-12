# Story — deck 45, FP4 QAT (MXFP4 expert weights + CSA indexer QK path)

- **Where it lives** (prosperity): DeepSeek-V4 post-training. The model is FP8 everywhere (E4M3,
  128 × 128 blocks) except two hot spots that go to FP4: routed MoE expert weights and the QK path of
  CSA's lightning indexer (deck 01).
- **Pain number** (betrayal): ~**97%** of V4-Pro's 1.6T parameters are routed-expert weights, ~1.55 TB
  even at FP8 (analytic). Second hot spot: at 1M tokens the indexer does 14× the multiply-adds of a CSA
  layer's core attention (analytic, deck 01).
- **Failed suitors**: (1) stay FP8: 1 byte per weight, the full 1.55 TB; (2) quantize to FP4 after
  training (PTQ): RL samples from one model and users get another, accuracy loss is unadapted;
  (3) V3.2-style FP8 indexer with FP32 scores: the 1M-token scan stays heavy.
- **Villain**: a precision split between training and serving. Training stacks are built for FP8, so
  FP4 normally appears only after training, where nothing can adapt to it.
- **Turn / insight**: FP4 → FP8 dequantization is exact (E4M3 has 2 more exponent bits than E2M1, so the
  1×32 FP4 sub-block scales fold into a 128×128 FP8 block when their ratio is bounded). QAT can therefore
  ride the unmodified FP8 training stack.
- **Mechanisms**:
  ❶ Expert-weight QAT — challenge: train against FP4 without an FP4 training stack → FP32 master → MXFP4
     → exact FP8 → existing FP8 GEMMs; gradient w.r.t. the FP8 weight goes straight to the master (STE).
  ❷ FP4 indexer QK path — challenge: the 1M-token scan dominates CSA → Q and compressed K cached, loaded
     and multiplied in FP4 (Hadamard-rotated in the reference code), scores FP32 → BF16.
  ❸ Native FP4 in rollout and serving — challenge: RL must sample the model users get → rollout and
     inference load real FP4 weights; teacher and reference models are QAT'ed too.
- **Release**: top-k selector **2×** faster at **99.7%** KV recall; released V4-Pro weights **0.54 B per
  parameter** (865 GB for 1.6T) vs V3.2's 1.03 (derived from checkpoint sizes).
- **Limits**: no FLOP gain on current hardware (FP4×FP8 peak = FP8×FP8; ~1/3 gain only on future
  hardware); exactness is conditional on scale ratios (verified empirically); post-training only; no
  FP4-vs-FP8 accuracy ablation published.

## Storyboard (13 logical slides, 8 minutes)

| # | Beat | Headline (as built) |
|---|---|---|
| 1 | Title | FP4 QAT: MXFP4 Experts + Indexer QK |
| 2 | Hook | V4 runs two hot spots in FP4 |
| 3 | Pain (97%, analytic) | Expert weights are nearly all of V4 |
| 4 | Status quo | FP8 is too big; FP4 after training drifts |
| 5 | Insight | Key idea: FP4 → FP8 dequantization is exact |
| 6 | Overview | Three FP4 paths, one set of bits |
| 7 | ❶ | ❶ Experts train through FP4, compute in FP8 |
| 8 | ❷ | ❷ The indexer scores blocks in FP4 |
| 9 | ❸ | ❸ Rollouts and serving share the FP4 bits |
| 10 | Result (2×, 99.7%) | Top-k selection runs 2× faster |
| 11 | Result (0.54 B/param) | V4-Pro ships at 0.54 bytes per parameter |
| 12 | Limits | Cost: no FLOP gain yet, and a scale caveat |
| 13 | Summary | Summary |
