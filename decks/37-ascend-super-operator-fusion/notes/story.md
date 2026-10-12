# Story — deck 37: Ascend · super-operator fusion

- **Where it lives** (prosperity): GLM-5, a 744B / 40B-active MoE with DeepSeek Sparse Attention,
  deployed from day one on seven Chinese chip platforms; the case study is Huawei Ascend Atlas, where the
  whole model must serve from a single Atlas 800T A3 machine through vLLM-Ascend and SGLang.
- **Pain number** (betrayal): the decode path is a crowd of tiny operators. MLA pre-processing alone is
  **13 small ops per attention layer × 80 layers ≈ 1,000 kernels per decoded token** (derived), each one
  handing its intermediate to the next through HBM, and each one using either the Cube unit or the Vector
  unit but not both.
- **Failed suitors**:
  1. Fix the host: async dispatch (TaskQueue) and CPU core binding remove dispatch stalls on Ascend
     (Pangu Ultra MoE §4.5) — the device still re-reads every intermediate from HBM.
  2. Reuse GPU-shaped library kernels: they cover the dense attention body, not DSA's pre- and
     post-processing (score/ReLU/TopK, MLA pre-processing, KV gather), so the sparse path stays a chain of
     small ops — which is why GLM-5's team wrote custom fusion kernels "to overcome the computational
     bottlenecks of sparse attention on Ascend NPUs" (§5).
- **Villain**: op-by-op execution on a split-unit core. The DaVinci AI core has a Cube unit for matmul and
  a Vector unit for element-wise work; a chain of small ops serializes them and pays HBM at every boundary.
- **Turn / insight**: the chain's intermediates are small and short-lived — they only exist because the ops
  are separate. One super-operator keeps them on chip and lets the Vector unit work on one tile while the
  Cube unit works on another. (GLM-5 §5: MLAPO "utilizing parallel processing between Vector and Cube
  units"; Lightning Indexer "allowing the NPU to overlap computation with memory access".)
- **Mechanisms**:
  ❶ **MLAPO** — 13 pre-processing ops (quantize → q down-proj → RMSNorm → q up-proj → RoPE → absorb BMM →
     kv down-proj → RMSNorm → RoPE → KV-cache scatter) become one super operator.
  ❷ **Lightning Indexer** — score calculation + ReLU + TopK in one kernel; the 32 × S score matrix never
     reaches HBM.
  ❸ **Sparse Flash Attention** — selecting the top-2048 KV rows and attending over them happen in the same
     kernel, so the gather overlaps the attention math.
- **Release**: the report's own currency is end-to-end: a single domestic node reaches performance
  "comparable to dual-GPU international clusters" and cuts long-sequence deployment cost by **50%** — for
  the whole co-optimization set (quantization + fusion + engine), not fusion alone. On the same silicon,
  the one public ablation (Pangu Ultra MoE, 718B training on 6K Ascend NPUs) puts fused operators at the
  last **+9.5 points** of a 58.7% cumulative MFU gain (derived from Table 5).
- **Limits**: no per-kernel numbers are published; each super-operator is hand-written against one model's
  shapes and one NPU generation (A2/A3 vs 950 support differs); fusion freezes the op graph, so model
  changes mean new kernels; the Notion row's "MoE Mega-Fusion" is not in the GLM-5 report (the stack does
  fuse grouped matmul + SwiGLU + quant, but that is vLLM-Ascend, not a GLM-5 claim).

## Slide plan (14 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | GLM-5 ships on domestic NPUs, not only GPUs | prosperity | bullets |
| 3 | One decoded token launches ~1,000 tiny kernels | pain | `%% big:` (derived) |
| 4 | Op-by-op execution idles half of the AI core | villain | fig_chain_1 (SVG, Cube/Vector lanes) |
| 5 | Insight: the intermediates only exist because the ops do | turn | boxed sentence |
| 6 | Three super-operators carry GLM-5's decode path | overview | fig_overview (SVG) |
| 7 | ❶ MLAPO collapses 13 pre-processing ops into one | mechanism | fig_chain_2 (same canvas, fused) |
| 8 | ❷ The indexer scores, rectifies and ranks in one kernel | mechanism | fig_indexer (SVG) |
| 9 | ❸ Sparse FA gathers the top-2048 KV and attends | mechanism | fig_sfa (SVG) |
| 10 | Fusing the indexer removes ~17 MB per decoded token | release (analytic) | chart indexer_traffic |
| 11 | On Ascend, fused operators were the last +9.5 MFU points | release (measured, other workload) | chart ascend_mfu |
| 12 | GLM-5: one domestic node, half the long-context cost | release (end-to-end) | `%% big: 50%` |
| 13 | The cost: hand-written kernels, frozen shapes | limits | two-col text |
| 14 | Summary | | QR |
