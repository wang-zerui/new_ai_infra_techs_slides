# Sources — deck 06: Staged context extension + annealing / cooldown

## Sources read

| Tag | Source | Where |
|---|---|---|
| [K3] | Kimi Team, "Kimi K3: Open Frontier Intelligence", arXiv 2607.24653. https://arxiv.org/abs/2607.24653 | Abstract (p.1); §2.1.2 Gated MLA / NoPE (p.5); §3.2 Scaling Law (p.10) + Table 1 (p.11); §3.3 Training Recipe (p.11-12); §3.4 Long-Context Extension (p.12); §5.1.2 KDA Context Parallelism (p.18); §6.1.3 Evaluation Configurations (p.26); Table 2 (p.27) |
| [K2] | Kimi Team, "Kimi K2: Open Agentic Intelligence", arXiv 2507.20534. https://arxiv.org/abs/2507.20534 | Abstract (p.1); §2.3 Model Architecture + Table 2 (p.6); §2.3 "Number of Attention Heads" (p.7); §2.5 Training recipe (p.9); §4 Evaluations, configuration (p.15); App. results text (p.29) |
| [GLM] | GLM-4.5 Team, "GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models", arXiv 2508.06471. https://arxiv.org/abs/2508.06471 | Abstract (p.1); Fig. 3 (p.4); §2.2-2.3 pre-/mid-training (p.4-5); §2.4 Hyper-Parameters (p.5); §3.2 Reasoning RL, Fig. 6 (p.7-8) |
| [N] | Notion row (notes/notion_row.md) | properties + body links |

All three PDFs were read from the shared cache (`$CACHE/arxiv-*.txt`). Blogs listed on the Notion row were
not fetched; every number on the slides comes from the three reports.

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 1.04T total / 32.6B active | Kimi K2 parameters | [K2] Table 2 p.6 (text says "32 billion activated") |
| 15.5T tokens | K2 pretraining tokens | [K2] abstract, §2.5 p.9 |
| 4,096-token (4K) window | K2 pretraining window | [K2] §2.5 p.9 |
| 10T tokens at constant LR 2e-4 after a 500-step warm-up | K2 WSD stable phase | [K2] §2.5 p.9 |
| 5.5T tokens, cosine decay 2e-4 → 2e-5 | K2 WSD decay phase | [K2] §2.5 p.9 |
| weight decay 0.1; global batch 67M tokens | K2 recipe | [K2] §2.5 p.9 |
| anneal: LR 2e-5 → 7e-6, 400B tokens at 4K, then 60B tokens at 32K | K2 annealing + long-context activation | [K2] §2.5 p.9 |
| YaRN to 128K | K2 context extension beyond the trained 32K | [K2] §2.5 p.9 |
| 128K | K2 evaluation window for long-context tasks | [K2] §4 p.15 |
| MRCR 55.0% vs 50.8% (DeepSeek-V3-0324) | K2-Instruct long-context retrieval at 128K | [K2] p.29 (also p.17) |
| FRAMES 77.1%, LongBench v2 49.1%, "around 2%" behind DeepSeek-V3-0324 | K2-Instruct long-context reasoning (speaker notes only) | [K2] p.29 |
| 83% | extra inference FLOPs at 128K when heads go 64 → 128 (expert count fixed at 384) | [K2] §2.3 p.7 (speaker notes only) |
| 355B total / 32B active | GLM-4.5 parameters | [GLM] abstract p.1 |
| 23T tokens | GLM-4.5 pre- + mid-training tokens | [GLM] abstract p.1 |
| 15T @4K, 7T @4K, 500B @32K, 500B @32K, 100B @128K | GLM-4.5 stages: general corpus; code & reasoning; repo-level code; synthetic reasoning; long context & agent | [GLM] Fig. 3 p.4 (stage windows "4K 4K 32K 32K 128K"); §2.3 p.4-5 |
| 4,096 → 32,768 → 131,072 | GLM-4.5 max sequence length per phase | [GLM] §2.3 p.5, §2.4 p.5 |
| LR warm-up 0 → 2.5e-4, cosine decay to 2.5e-5 at end of mid-training | GLM-4.5 schedule (cosine, not WSD) | [GLM] §2.4 p.5 |
| WSD "underfitting in the stable stage", worse SimpleQA / MMLU | GLM-4.5 reason for cosine | [GLM] §2.4 p.5 |
| batch 16M → 64M tokens over the first 500B tokens; WD 0.1 | GLM-4.5 recipe | [GLM] §2.4 p.5 |
| RoPE base 10,000 → 1,000,000 when moving to 32K | GLM-4.5 positional change | [GLM] §2.4 p.5 |
| upsample long documents; best-fit packing in mid-training | GLM-4.5 long-context data | [GLM] §2.3 p.5 |
| 80.6% vs 83.4% AIME24 avg@32 | GLM-4.5 RL: multi-stage (16K→32K→48K→64K) vs single-stage 64K output length; smaller experimental model | [GLM] §3.2 Fig. 6 p.8 |
| 2.8T (2.78T) total / 104B (104.2B) active | Kimi K3 parameters | [K3] abstract p.1, Table 1 p.11 |
| 1M window; "8×" K2's 128K training context | K3 training context length | [K3] Table 1 p.11 |
| 93 layers = 69 KDA + 24 MLA | K3 attention-layer composition | [K3] Table 1 p.11 |
| 8K → 64K in pretraining; 256K → 1M in cooldown; "four-stage curriculum" | K3 progressive context extension | [K3] §3.3 p.12, §3.4 p.12. The four stages are read as 8K, 64K, 256K, 1M (the report names these four lengths and "four-stage"). |
| cosine LR with 1% linear warm-up; WD 0.1 | K3 recipe | [K3] §3.3 p.11 |
| cosine beats WSD under independently tuned hyper-parameters | K3 scaling-law study | [K3] §3.2 p.10 |
| NoPE on all MLA layers; no RoPE rescaling or interpolation | K3 positions | [K3] §2.1.2 p.5, §3.4 p.12 |
| long data upsampled; synthetic permuted/concatenated long-dependency data | K3 long-context data | [K3] §3.4 p.12 |
| "small fraction of the overall training budget" | K3 long-sequence share (no number) | [K3] §3.4 p.12 |
| BrowseComp 90.4% with full 1M window, no context management (91.2% with compaction at 300K) | K3 long-context use | [K3] §6.1.3 p.26, Table 2 p.27 |
| AA-LCR 74.7 | K3 long-context reasoning | [K3] Table 2 p.27 (first column = Kimi K3) |

## Derived numbers (labelled "derived" or "analytic" on the slides)

1. **32× / 128× attention work per token (analytic).** Causal attention lets each token read every earlier
   token in its window, so the attention-score work per token is proportional to the window length L
   (mean L/2 keys). Ratio 128K / 4K = 131,072 / 4,096 = **32×** (K2 and GLM-4.5: 4K pretraining window,
   128K final window). K3: 1M / 8K = 1,048,576 / 8,192 = **128×**, which applies only to its 24 MLA
   (global) layers; its 69 KDA layers have constant per-token cost (deck 05). This counts attention only,
   not the MLP/expert FLOPs, which do not depend on L; it assumes attention spans the whole packed window
   (no per-document masking), i.e. an upper bound.
2. **Token shares of the long stages (derived).**
   - K2 total tokens = 15.5T + 0.4T + 0.06T = **15.96T**. The anneal starts at 2e-5, which is where the 5.5T
     cosine ends, so we place it after the 15.5T. If it were inside the 15.5T the shares change by < 0.02
     percentage points. 32K stage share = 0.06 / 15.96 = **0.38%** (≈ 0.4%).
   - GLM-4.5 total = 15 + 7 + 0.5 + 0.5 + 0.1 = **23.1T** (the paper rounds to 23T). 128K stage share =
     0.1 / 23.1 = **0.43%** (≈ 0.4%); ≥ 32K share = 1.1 / 23.1 = **4.8%**.
3. **Run-average attention work per token relative to a 4K-only run (analytic, token-weighted).**
   mean(L) / 4K = Σ tokens_i · (L_i / 4K) / Σ tokens_i.
   - K2: (15.9 · 1 + 0.06 · 8) / 15.96 = 16.38 / 15.96 = **1.03×** (1.026).
   - GLM-4.5: (22 · 1 + 1.0 · 8 + 0.1 · 32) / 23.1 = 33.2 / 23.1 = **1.44×** (1.437).
   - 128K window on every token: **32×**.
   - Side note (speaker notes): in GLM-4.5 the 128K stage is 0.43% of tokens but 3.2 / 33.2 = 9.6% of
     the run's attention work; the two 32K stages are 8 / 33.2 = 24%.
4. **K2 warm-up length (notes only).** 500 steps × 67M tokens = 33.5B tokens (0.2% of the run); drawn in
   the LR chart but not stated on a slide.
5. **K2 YaRN factor.** Not stated in [K2]; slides say only "YaRN to 128K" (from the trained 32K).

## Charts

- `charts/chart_windows.py`: bulk-pretraining window vs final window per model (4K/128K, 4K/128K,
  8K/1M), log2 axis. Numbers from the table above.
- `charts/chart_lr.py`: LR vs cumulative tokens. K2: linear warm-up over 33.5B tokens (derived), constant
  2e-4 to 10T, cosine to 2e-5 at 15.5T, then a straight dashed segment to 7e-6 at 15.96T (the anneal's
  shape is not published; drawn as a straight line and marked as such). GLM-4.5: cosine from 2.5e-4 to
  2.5e-5 over 23.1T (warm-up length not published, omitted). K3 not plotted (peak/final LR and token
  budget not published).
- `charts/chart_timeline.py`: training window vs cumulative tokens (step chart) for K2 and GLM-4.5 using
  the stage budgets above; K2's YaRN extension to 128K drawn dotted (no training tokens). K3 not plotted
  (no stage budgets).
- `charts/chart_span.py`: the analytic run-average attention work per token (derived numbers 3).

## Discrepancies (Notion row vs primary sources)

1. **LR schedule.** Notion: "constant LR plus cosine decay and final annealing". That is K2's WSD schedule
   ([K2] §2.5). K3 uses cosine decay with a 1% linear warm-up ([K3] §3.3) and its scaling-law study
   favours cosine over WSD ([K3] §3.2); GLM-4.5 also uses cosine and reports WSD underfits ([GLM] §2.4).
   Slides present WSD as K2's choice and cosine as K3's and GLM-4.5's.
2. **"this is how K3 gets its 1M context" / "small tail of tokens".** K3 says the long-sequence work is a
   "small fraction of the overall training budget" but publishes no stage token budgets. The quantified
   tails on the slides come from K2 (60B at 32K) and GLM-4.5 (100B at 128K). K3's 1M also relies on NoPE
   + KDA (no positional rescaling, [K3] §3.4), not on staging alone.
3. **K2 "4k->32k->128k with YaRN".** Only 4K and 32K are trained stages; 128K is reached with YaRN on top
   of the 32K stage ([K2] §2.5). Slides draw the 128K step dotted, as an extension, not a training stage.
4. **"Measured results".** Notion's impact line has no measurement, and none of the three reports ablates
   staged vs. unstaged training. Slides label the cost comparison "analytic" and present end-to-end
   long-context scores as outcomes, not as an ablation.
5. **"cooldown / mid-training" for 256K→1M.** K3 calls its phase "cooldown"; GLM-4.5 calls its long stages
   "mid-training" and runs them within one cosine schedule. Slides use "low-LR tail" for both.
6. **K2 token total.** [K2] says 15.5T total and separately describes a 460B anneal + long-context stage
   "towards the end of pre-training"; whether the 460B is inside the 15.5T is not stated (see derived 2).
