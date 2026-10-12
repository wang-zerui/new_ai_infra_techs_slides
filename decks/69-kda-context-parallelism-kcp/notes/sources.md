# Sources: deck 69, KDA Context Parallelism (KCP)

## Sources read

| Key | Source | Where | What was used |
|---|---|---|---|
| K3 | Kimi Team, "Kimi K3: Open Frontier Intelligence", arXiv 2607.24653 (https://arxiv.org/abs/2607.24653); cached text `$CACHE/arxiv-2607.24653.txt` | §3.4 (context curriculum, CP pointer); §5.1 intro; §5.1.1 (intra-device CP); §5.1.2 (KCP, Eq. 17, footnote 2); §5.2 (parallelism stack); Table 1; refs [73], [115], [116], [141], [144] | Problem statement, why summation fails, affine decomposition, all-gather + prefix-scan merge, use in pre-training |
| PR691 | fla-org/flash-linear-attention PR #691 "[CP] Add cp for gdn and kda" (created 2025-12-19, merged 2026-01-20), description via `gh api` (saved in `$SCRATCH/research/69/`) | "Performance" (KDA forward and forward+backward), communication-volume sentence, "conv1d", "Drawbacks" | Measured times and efficiencies at 32K, CP = 4, 64 heads, head dim 128, H800; payload CP x H x K x (K + V); conv halo; drawbacks |
| FLA-CP | flash-linear-attention `fla/ops/cp/README.md`, `comm.py`, `chunk_delta_h.py` (main branch, fetched 2026-10-13) | "CP Architecture", "Pre-Process Forward/Backward", "Transition Matrix", "Initial State Memory Optimization", "Acknowledgments"; `chunk_gated_delta_rule_fwd_h_pre_process` (hm = [H, K, V + K] fp32, all-gather, merge kernel), zigzag layout | Fragment formulas, merge rule, backward direction, FP32 chain, packed-document handling, conv halo API, independence from Moonshot's internal KCP |
| MEG | Megatron-LM context_parallel.md (cached `$CACHE/megatron-context_parallel.md`) | overview | Softmax CP all-gathers KV (ring P2P) and reduce-scatters KV gradients |
| K3README | https://github.com/MoonshotAI/Kimi-K3 README (via `gh api`) | model summary table | 96 attention heads, 69 KDA + 24 Gated MLA, context length 1,048,576 |
| FKDA | https://github.com/MoonshotAI/FlashKDA README | Kernel API | "Currently requires K = V = 128" (used to infer K3's KDA head dimension) |
| not read | Yaoyu Wang, "Context Parallelism for DeltaNet" (notion.site, K3 ref. [144]); LASP / LASP-2 (arXiv 2404.02882, 2502.07563); Ring Attention (arXiv 2310.01889) | - | JS-rendered page / not needed: their role is taken from K3 §5.1.2's description; no numbers from them are on slides |

## Numbers on slides

| Slide | Number | Source location | Note |
|---|---|---|---|
| 2 | context 8K -> 1M during training | K3 §3.4 (8K -> 64K pre-training, 256K -> 1M cooldown) | |
| 2 | PP, EP, ZeRO, CP | K3 §5.2 first sentence | |
| 2 (fig) | 69 of 93 KDA, 24 of 93 Gated MLA; up to 1M tokens | K3 Table 1; K3README | |
| 3 | 55% (55.12%) fwd+bwd, 37% (37.15%) fwd, head-split all-to-all CP | PR691 "KDA" block; rate = non-CP time / (CP time x 4): 36.193/(16.415 x 4), 7.839/(5.276 x 4) | 32K tokens, CP = 4, 64 heads, head dim 128, H800 |
| 4 | M_t = (I - beta k k^T) Diag(alpha) | K3 §5.1.2 | |
| 5, 7 | S_out = M S_in + S~ | K3 Eq. 17 | |
| 7 (fig) | S~ step and M_[t] = Diag(gamma) - (Gamma (.) K)^T W | FLA-CP README "Pre-Process Forward" | |
| 8 | payload per GPU H x d_k x (d_k + d_v) | PR691 ("communication volume of CP x H x K x (K + V)"); FLA-CP chunk_delta_h.py (`hm = [HV, K, V + K]` fp32) | |
| 8 | 12 MiB per GPU per layer | derived: 96 x 128 x (128 + 128) x 4 B = 12,582,912 B = 12 MiB | assumes 96 KDA heads (K3 Table 1 "Attention Heads 96") and head dim 128 (FlashKDA supports only K = V = 128 and K3 uses FlashKDA), FP32 as in FLA |
| 9 | W - 1 tail tokens (conv window W) | FLA-CP README (`conv1d_kernel_size`), PR691 "conv1d" section | W itself not stated for K3; figure shows W = 4 as illustration |
| 10 | 86% (86.26) and 76% (75.65) KCP; 55% / 37% all-to-all | PR691 | |
| 10 | 36.2 ms -> 10.5 ms (36.193 -> 10.490), 16.4 ms (16.415) | PR691 forward + backward | |
| 10 | 1.56x faster than head-split CP | derived: 16.415 / 10.490 = 1.565 | |
| 10 (notes) | forward 7.839 -> 2.591 ms vs 5.276 ms | PR691 | |
| 11 | 512x; 448 MiB vs 0.875 MiB per head per layer at 1M tokens, 8 GPUs | analytic, derived: softmax (P-1)/P x T x 2 x 128 x 2 B = 7/8 x 1,048,576 x 512 B = 448 MiB; KCP (P-1) x 128 x 256 x 4 B = 7 x 131,072 B = 0.875 MiB; ratio T / (256 P) = 512 | comparator: MHA-equivalent K and V in BF16 with the same head dim; MLA/GQA would shrink KV by a constant factor (MEG notes GQA/MQA reduce CP volume) |
| 12 (chart + notes) | KCP 0.125 MiB (P = 2) to 7.9 MiB (P = 64); softmax 256 to 504 MiB at T = 1M | analytic, same formulas | |
| 12 | three extra operators, one all-gather; "lossy" vs all-to-all; payload grows with CP | PR691 "Drawbacks" and communication paragraph | |
| 12 (notes) | initial_state / output_final_state not supported in CP mode | FLA-CP README Quick Start note | |

## Discrepancies

1. **Notion "compute scales linearly to 1M context".** K3 §5.1.2 states that KCP "achieves linear compute scaling"
   and §3.4 that CP makes million-token training tractable, but no KCP measurement at 1M (or any other length) is
   published. The only public timing is FLA's 32K, CP = 4 benchmark (PR #691). The deck shows the 32K measurement and
   labels the 1M comparison as analytic.
2. **Which implementation the numbers describe.** K3 footnote 2 says "The KDA implementation is available in FLA PR
   #691"; FLA's own README says its CP code "was independently contributed to FLA and is a separate codebase from the
   internal Moonshot implementation" (where it is called KCP, "Kimi Context Parallel"). The measured numbers are
   therefore for the open FLA implementation, not necessarily Moonshot's internal one. Stated on slides 10 and 12.
3. **Name.** K3 expands KCP as "KDA Context Parallelism" (as Notion does); FLA's README calls it "Kimi Context
   Parallel". Same technique.
4. **Notion body: "Kimi K3 repo - architecture and training notes".** The Kimi-K3 README contains a model summary and
   usage notes but nothing about KCP; all mechanism detail comes from the K3 report and FLA.
5. **Precision.** Neither Notion nor K3 mention precision; FLA's PR calls the outputs and gradients "lossy" relative to
   all-to-all CP and its README requires the M chain in FP32. Mentioned in the costs slide notes.
