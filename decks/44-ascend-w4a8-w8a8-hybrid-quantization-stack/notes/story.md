# Story — deck 44: Ascend · W4A8/W8A8 hybrid quantization stack

- **Where it lives** (prosperity): GLM-5, 744B parameters (40B active, 256 experts), served on Huawei
  Ascend through vLLM-Ascend and SGLang; the target is one Atlas 800 A3 node (128 GB × 8 = 1,024 GB) with
  200K-token contexts.
- **Pain number** (betrayal): **1.49 TB** of BF16 weights (744e9 × 2 B, derived) against a 1,024 GB node —
  and the KV cache for a 200K context has to come out of whatever is left.
- **Failed suitors**:
  1. INT8 everywhere (W8A8): 744 GB of weights fit, but leave only ~280 GB of the node for KV cache,
     activations and buffers — the long-context case is where the model is supposed to earn its keep.
  2. 4 bits everywhere: activations carry outlier channels orders of magnitude larger than the rest, so
     INT4 activations (or INT4 on the shared attention path) wreck accuracy; this is the exact failure
     QuaRot and SmoothQuant were written for.
- **Villain**: outlier channels in the activations — a property of the *basis* the model happens to be
  written in, not of the information it carries.
- **Turn / insight**: a rotation is free. Orthogonal Q satisfies QQᵀ = I, so folding Hadamard rotations
  into the weights leaves the network's output unchanged while spreading every outlier channel across all
  channels. Once the basis is outlier-free, the remaining decisions are pure bookkeeping: what to
  calibrate and which tensors can afford 4 bits.
- **Mechanisms**:
  ❶ **Rotate** — QuaRot: randomized Hadamard transforms fused into the weight matrices (computational
    invariance), so there are no outlier features left to protect.
  ❷ **Calibrate** — Flex_AWQ_SSZ: an AWQ-style per-channel scaling search on calibration data, with an
    asymmetric (scale/offset) weight quantizer; run by msModelSlim.
  ❸ **Assign per tensor** — W4A8 for the MoE experts (97% of the weights, each touched by few tokens),
    W8A8 for the attention and dense-MLP blocks every token crosses; the plan is a per-weight table in the
    checkpoint, and W4A8 at runtime is int4 weights packed two per byte with per-token int8 activations.
  ❹ **Close the loop** — INT4 QAT in the SFT stage, with one kernel shared by training and offline weight
    quantization so training and inference are bitwise identical.
- **Release**: weights drop from 1,488 GB (BF16) to **382 GB** (derived): the model occupies a third of
  one Atlas 800 A3 node instead of 73%, which is why the deployment guide tells you to use the w4a8
  weights for 200K contexts. The report's own end-to-end claim: **50%** lower long-sequence deployment
  cost and a single domestic node comparable to "dual-GPU international clusters" — for the whole
  co-optimization set, not quantization alone.
- **Limits**: no published accuracy numbers for the quantized GLM-5 (the report says only "without
  significant accuracy loss"); calibration and QAT cost real training time; per-channel only (per-group
  W4A8 is rejected by the runtime); W4A8 is an MoE-expert scheme, so the attention path stays at 8 bits;
  and the fused MLA pre-processing operator (deck 37) is recommended for w8a8 but "may not benefit" w4a8.

## Slide plan (15 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | GLM-5 has to serve 744B weights from one node | prosperity | bullets |
| 3 | 1.49 TB of weights, 1,024 GB of node | pain | `%% big:` (derived) |
| 4 | Both obvious precision plans fail | failed suitors | fig_plans (SVG, two panels) |
| 5 | Insight: the outliers live in the basis | turn | boxed sentence |
| 6 | Four steps turn 16-bit weights into W4A8 | overview | fig_pipeline (SVG) |
| 7 | ❶ Rotating the model deletes the outliers | mechanism | fig_rotate (SVG) |
| 8 | ❷ Calibration searches the scales per layer | mechanism | fig_calib (SVG) |
| 9 | ❸ The experts hold 97% of the weights | mechanism | chart param_share |
| 10 | ❸ W4A8 is int4 weights, int8 per-token activations | mechanism | fig_w4a8 (SVG) |
| 11 | ❹ INT4 QAT makes training match inference | mechanism | bullets + boxed |
| 12 | 382 GB of weights leaves the node to the KV cache | release | chart weight_memory |
| 13 | What is not published: the accuracy | honesty | bullets |
| 14 | The cost: calibration, QAT, and a frozen plan | limits | two-col |
| 15 | Summary | | QR |
