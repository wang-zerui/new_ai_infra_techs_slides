# Story — deck 23: Ascend adaptation stack (GLM-5) [container entry]

- **Where it lives** (prosperity): GLM-5, a 744B-parameter MoE (40B active, 256 experts, DSA sparse
  attention, 200K context), released with day-one full-stack adaptation to seven Chinese accelerator
  platforms; the report walks the Ascend Atlas stack as its case study.
- **Pain number** (betrayal): the weights alone are **1.4 TiB** at BF16, while one Atlas 800 A3 node holds
  **1.0 TiB** of NPU memory — before a single KV-cache byte. (Analytic, from the architecture table;
  the report publishes no memory breakdown.)
- **Failed suitors**:
  1. Spread BF16 weights over two nodes: doubles the hardware per replica and pushes tensor-parallel and
     expert-parallel traffic onto the slower inter-node fabric — exactly the exposure the deployment is
     trying to avoid.
  2. Port the GPU stack unchanged: the model's own shape was tuned to a GPU roofline (the report notes
     DeepSeek-V3's head count follows the H800 roofline and "is inappropriate for other hardware"), and
     a sparse-attention model lands on the NPU as dozens of small pre/post-processing kernels.
- **Villain**: a model shaped by, and a kernel stack written for, one vendor's roofline — memory capacity,
  interconnect bandwidth and the Vector/Cube unit split all differ on the target accelerator.
- **Turn / insight**: fitting a frontier MoE on one domestic node is not a kernel problem or a precision
  problem — it is three problems at three layers, and the report attacks exactly one resource at each:
  memory (quantization), compute density (super-operator fusion), and the exposed time around them
  (engine-level overlap and scheduling).
- **Mechanisms** (the three pillars; each is a sibling deck):
  ❶ W4A8/W8A8 mixed quantization — INT4 experts, INT8 attention/MLP → the weights fit one node (deck 44).
  ❷ Super-operator fusion — Lightning Indexer, Sparse Flash Attention, MLAPO (13 ops → 1), Vector and
     Cube units in parallel (deck 37).
  ❸ Comm/compute fusion and PD-disaggregated scheduling — FlashComm splits AllReduce, DP attention + EP,
     prefix cache, MTP, asynchronous scheduling (deck 12).
- **Release**: the report claims one domestic node reaches performance "comparable to dual-GPU
  international clusters" and **50%** lower deployment cost in long-sequence scenarios.
- **Limits**: the claim comes with no baseline model, benchmark, metric or configuration; only Ascend of
  the seven platforms is described; everything in §5 is inference-side (the report never says GLM-5 was
  trained on domestic accelerators); and no accuracy table is given for the W4A8 weights.

## Slide plan (14 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | GLM-5 ships day one on seven domestic platforms | prosperity | 2 bullets + fig_platforms |
| 3 | The weights alone outgrow the node | pain | `%% big: 1.4 TiB` (analytic) |
| 4 | A GPU-shaped stack does not land on an NPU | villain / suitors | fig_mismatch (before panel) |
| 5 | Insight: three layers, three different resources | turn | boxed sentence |
| 6 | The stack: quantize, fuse, schedule | overview | fig_stack (sibling-deck refs) |
| 7 | ❶ Mixed W4A8/W8A8 puts the weights on one node | pillar 1 | fig_quant + deck 44 pointer |
| 8 | ❷ Super-operators replace kernel swarms | pillar 2 | fig_fuse + deck 37 pointer |
| 9 | ❸ Scheduling hides what is left | pillar 3 | fig_sched + deck 12 pointer |
| 10 | Only INT4 experts fit a 744B model on one node | evidence | chart weight_memory (analytic) |
| 11 | The report's claim: one node, half the long-sequence cost | release | big quote + caveat bullets |
| 12 | What the report does not tell you | honesty / limits | two-col text |
| 13 | The recipes are upstream and in production | adoption | bullets (vLLM Ascend, press claim) |
| 14 | Summary | | QR |
