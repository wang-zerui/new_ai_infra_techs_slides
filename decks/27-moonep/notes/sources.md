# Sources for deck 27 (MoonEP library overview)

All downloads live in `$SCRATCH/research/27/` (MoonEP tarball of `master` at commit d86fda4, 2026-10-12)
and the shared cache `$SCRATCH/research/cache/`.

## Sources read

| Tag | Source | What was used |
|---|---|---|
| [README] | MoonEP README, https://github.com/MoonshotAI/MoonEP (cache: `gh-moonep-README.md`, identical to repo master) | three properties, notation, maxvio definition, perf claims, API, weight/grad buffer layout |
| [CHART-COMM] | `figure/comm_vs_deepep.png` in the repo (rendered by `benchmarks/bench_vs_deepep.py`) | labeled bar totals (µs) for dispatch/combine fwd/bwd at maxvio 0.2/1/10/20 |
| [CHART-E2E] | `figure/e2e_vs_deepep.png` in the repo | labeled iteration-time percentages and "deepep OOM" / "pass" at maxvio 20 |
| [BENCH] | `benchmarks/bench_vs_deepep.py` docstring | what each op includes; DeepEP v2 elastic `do_expand=True`; 32 SMs; padding 128; warmup 20 / iters 50; single node; lognormal routing |
| [API] | `moonep/api.py` (`_create_context`, `Buffer.__init__`) | NvS = align(S·K + (token_padding−1)·2·epn, token_padding); default num_sms 32 |
| [PLAN] | `moonep/planning.py` docstring + kernel body; `tests/planning_reference.py` | one cooperative GPU launch; rank 0 gathers tokens-per-expert over NVLink, balances, multicasts the plan (`multimem.st`) |
| [K3] | Kimi K3 technical report, arXiv 2607.24653 (cache: `arxiv-2607.24653.txt`) | §5.2.1 Perfectly Balanced EP MoE Training; App. E proof; Table 1 architecture; abstract |
| [DEEPEP] | DeepEP README (cache: `gh-deepep-README.md`, V2.5 version) | `do_cpu_sync` semantics; V2.5 adds `lb_prefetch_weights` / `lb_reduce_grads` citing MoonEP |
| [DEEPEP-GIT] | `gh api repos/deepseek-ai/DeepEP/commits` | V2.5 commit def8651 dated 2026-09-29 |
| [GIT] | `gh api repos/MoonshotAI/MoonEP` and `/commits` (queried 2026-10-13) | created 2026-07-24, MIT, 1,170 stars, 138 forks; multi-node NVLink fabric 2026-08-06 (#23); MXFP4 expert-weight prefetch 2026-08-13 (#29) |
| [ARXIV-K3] | arXiv API `export.arxiv.org/api/query?id_list=2607.24653` | K3 report published 2026-07-27 (v2 2026-08-07) |
| [V3] | DeepSeek-V3 technical report, arXiv 2412.19437 (cache) | §3.4 redundant experts re-planned "periodically (e.g., every 10 minutes)" from load statistics; 32 redundant experts for prefill |
| [EPLB] | DeepSeek EPLB README (cache: `gh-eplb-README.md`) | plans from estimated loads; "moving average of historical statistics" |
| [MCORE] | Megatron-Core MoE report, arXiv 2603.07685 (cache) §4.3.7 | dynamic shapes need a device-to-host sync; worst-case buffers "O(EP_size)" larger; ECHO clones hot experts to spare slots |

## Numbers on slides

| Slide | Number | Source location |
|---|---|---|
| 2 | Kimi K3: 2.8T total parameters (2.78T in Table 1), 104B activated, 896 routed experts, 16 active per token | [K3] abstract; Table 1 |
| 2 | Kimi K2 predecessor: 384 routed experts, top-8 | [K3] Table 1 (used only in notes) |
| 3 | DeepEP v2 end-to-end training OOMs at maxvio 20; MoonEP "pass" | [CHART-E2E] red "deepep OOM" marker at 20.0; [README] "until training OOMs at high imbalance" |
| 3 | maxvio 20 = hottest expert at 21× the average load | [README] maxvio = max_e(T_e / T̄) − 1; [CHART-COMM] x-axis legend "k = hottest expert at (k+1)x load" |
| 3, 12 | e2e setup: EP=8, seq 16k, 9 layers, H=7168, E=256, top-8, H'=2048, H20 | [CHART-E2E] title; [README] "Both benchmarks run on H20 with EP=8" |
| 4 | running example: 4 ranks, S·K = 8 copies each, loads 12/12/4/4 | illustrative (constructed for the figure, not a measurement; same example as deck 20) |
| 5 | DeepSeek-V3 redundant experts adjusted "e.g., every 10 minutes" from online statistics | [V3] §3.4.1 (Prefilling); [EPLB] "moving average of historical statistics" |
| 5 | ECHO / UltraEP preset redundant-expert count or per-rank token cap; training stops if no feasible plan; cap needs manual tuning | [K3] §5.2.1; ECHO plans per forward from current load ([MCORE] §4.3.7); UltraEP plans from current post-gating load ([DEEPEP] Expert load balancing) |
| 5 | Quantile Balancing bias "takes effect only in the next step" | [K3] §2.3.3 |
| 6 | each under-loaded GPU filled from one overloaded GPU, so its replicas come from that GPU's E/R experts | [K3] App. E proof of Theorem 1 (key lemma) |
| 6 | balanced plan always exists with ≤ E/R redundant experts per rank; bound essentially tight (⌈E(R−1)/R²⌉ ≈ E/R) | [K3] §5.2.1 and App. E Theorems 1-2 |
| 6, 10 | every rank receives exactly S × K tokens | [README] property 1; [K3] §5.2.1 |
| 9 | planning ≈ 40-50 µs ≈ 2% of the ≈ 2.4-2.6 ms dispatch fwd | **estimated**: pixel heights of the green "MoonEP planning" segment in [CHART-COMM] dispatch-fwd panel, calibrated by linear fit of the 8 labeled bar tops (slope 3.25 µs/px, residual ≤ 6 µs): 42, 42, 52, 52 µs at maxvio 0.2/1/10/20; totals 2598-2644 µs (labeled). Slide says "≈, read off the README chart". |
| 9 | plan computed by one GPU launch; host not involved per layer | [PLAN] module docstring ("A single cooperative grid launch ... Phase A/B/C/D in one kernel"); [README] "no per-layer MoE host synchronization" |
| 10 | buffer sizes (analytic): MoonEP [NvS, H] BF16 = 1.12 GB at EP=8 (0.94 GB of it is the S·K payload); copy-free worst case S·K·R·H·2 B = 7.52 GB at EP=8, 15.0 GB at 16, 30.1 GB at 32, 60.1 GB at 64 | **derived** from [K3] §5.2.1 ("requires a communication buffer of size S × K × R, whereas MoonEP requires only a fixed S × K buffer") and [API] NvS formula; config S=8192, K=8, H=7168, E=384 = the [CHART-COMM] config; BF16 = 2 B. NvS(EP) = 89,984 (4), 77,824 (8), 71,680 (16), 68,608 (32), 67,072 (64) slots. Script: `charts/buffer_vs_ep.py`. |
| 10 | DeepEP v2 `do_cpu_sync=True` gets exact sizes via CPU sync, "commonly used for training"; `False` allocates to configured capacity | [DEEPEP] "Example use in model training and inference" |
| 11 | per-layer EP comm (dispatch fwd + dispatch bwd + combine fwd + combine bwd, MoonEP incl. planning + prefetch): MoonEP 9,044 / 9,068 / 9,105 / 8,670 µs; DeepEP v2 8,870 / 8,902 / 9,453 / 10,175 µs at maxvio 0.2 / 1 / 10 / 20 | **derived** sums of the 16 labeled bar values in [CHART-COMM]: MoonEP d_f 2598/2644/2629/2413, d_b 2526/2548/2550/2340, c_f 1975/1932/1955/1954, c_b 1945/1944/1971/1963; DeepEP v2 d_f 2296/2299/2408/2604, d_b 2311/2310/2421/2624, c_f 2126/2142/2306/2468, c_b 2137/2151/2318/2479 |
| 11 | DeepEP v2 grows 15% from maxvio 0.2 to 20 (10,175 / 8,870 = 1.147); MoonEP 15% lower at maxvio 20 ((10,175 − 8,670)/10,175 = 14.8%); MoonEP 2% higher at maxvio 0.2 (9,044 / 8,870 = 1.020) | derived from the row above |
| 11 | comm setup: H20, EP=8, E=384, H=7168, K=8, S=8192/rank, 32 SMs | [CHART-COMM] title; [BENCH] |
| 11 | combine fwd: MoonEP 1,932-1,975 µs; DeepEP v2 2,126 → 2,468 µs | [CHART-COMM] labels (used in notes) |
| 12 | MoonEP iteration-time saving vs DeepEP v2: −1.9% (0.2), −0.5% (1), +3.9% (5), +7.6% (10), +11.1% (15); maxvio 20: DeepEP OOM, MoonEP pass | [CHART-E2E] labels. Checked meaning: label = (T_DeepEP − T_MoonEP)/T_DeepEP, e.g. at 15: (1.127 − 1.002)/1.127 = 11.1% (plotted points read approximately). |
| 13 | repo public 2026-07-24, MIT; 1,170 stars, 138 forks (2026-10-13) | [GIT] |
| 13 | Kimi K3 report on arXiv 2026-07-27 | [ARXIV-K3] |
| 13 | K3 pre-training uses MoonEP | [K3] §1 and §5.2.1 |
| 13 | DeepEP V2.5 (2026-09-29) adds `lb_prefetch_weights` / `lb_reduce_grads` "support the expert-replication approach explored by MoonEP and UltraEP" | [DEEPEP] News; [DEEPEP-GIT] |
| 13 | multi-node NVLink fabric support (2026-08-06), MXFP4 expert-weight prefetch (2026-08-13), Zhenwu PPU "under review" | [GIT] commits #23, #29; [README] Supported Devices |
| 14 | prefetch ≈ 0.5-0.75 ms of each dispatch fwd/bwd | **estimated** from purple segment pixel heights in [CHART-COMM] (same calibration as slide 9): 725 / 754 / 732 / 523 µs (fwd), 721 / 754 / 728 / 520 µs (bwd) |
| 14 | prefetch pool = epn expert weights per projection, shared by all layers (+ FP32 grad reduce buffer of the same shape) | [README] Weight buffer / Gradient buffers |
| 14 | pool size at E=384, EP=8 (slide: "4.2 + 8.5 GB at EP=8"), H=7168, H'=2048: 4.2 GB BF16 weights + 8.5 GB FP32 grads | **derived**: 48 × 3 × 7168 × 2048 × 2 B = 4.23 GB; × 4 B = 8.46 GB |
| 14 | NVLink domain only (VMM/NVLink buffers + NVSwitch multicast; no RDMA path); tests require NVLink | [API]/`buffer.py` (create_nvl_dist_tensor, multicast); [README] Build & Test; [BENCH] "Single node only" |
| 14 | per-expert token counts inside a rank stay skewed; K3 adds a workload-aware expert-GEMM scheduler | [K3] §5.2.1 "Expert-GEMM scheduling and overlap" |
| 14 | balanced-case cost: −1.9% e2e at maxvio 0.2 | [CHART-E2E] |

## Context number (speaker notes of slide 12 only)

- `benchmarks/bench_comm.py` and `tests/generate_topk_routing.py` docstrings label lognormal sigma 0.1 "near balanced",
  1 "typical dropless-MoE skew", 5 "near-degenerate". Re-simulating the generator (lognormal expert weights, top-K
  without replacement via Gumbel-top-k, S=8192, K=8, R=8, 5 seeds; `$SCRATCH/scripts/sim_maxvio.py`) gives realized
  maxvio 7.9-13.0 (mean 10.7) at E=384 and 5.2-12.2 (mean 9.2) at E=256 for sigma = 1; sigma = 0.5 gives 1.8-3.4.
  **Derived**, synthetic routing only.

## Derivations (scripts)

- `charts/buffer_vs_ep.py`: NvS = ceil((S·K + 127·2·E/R)/128)·128; bytes = NvS·H·2; worst = S·K·R·H·2.
- `charts/comm_total.py`: sums of the four labeled ops per maxvio (table above).
- `charts/e2e_saving.py`: the five labeled percentages plus the OOM marker.
- Pixel estimates: `$SCRATCH/scripts/measure_bars.py` (segment extents by colour) + linear calibration on labeled tops.

## Discrepancies (Notion row vs primary sources)

1. Notion (deck 20 row, inherited here): "every EP rank receives exactly S x K / EP tokens". Primary sources: each rank
   receives exactly **S × K** tokens, with S = input tokens *per rank* ([README] property 1, [K3] §5.2.1). Slides use S × K.
2. Notion: "communication buffer drops from a worst-case S x K x R to a fixed S x K" matches [K3], but the actual
   allocation is NvS = S·K plus per-expert-group padding slack (≤ 127·2·E/R slots) rounded to 128 ([API]): 77,824 slots
   instead of 65,536 at the benchmark config. Slide 10 plots the real NvS and says "S·K + padding".
3. Notion: "increasingly outperforms DeepEP v2 as routing imbalance grows". True for combine and for the totals at high
   imbalance, but at low imbalance MoonEP's dispatch (with planning + prefetch) is slower (2,598 vs 2,296 µs at maxvio
   0.2) and e2e iteration time is 1.9% worse at maxvio 0.2 ([CHART-COMM], [CHART-E2E]). Slides show both sides.
4. K3 says "S is the sequence length" in §5.2.1; README/API define S as input tokens per rank. Same thing when one
   sequence per rank; slides use the README definition.
5. Notion (deck 20 row) says static shapes are "CUDA Graph friendly"; neither README nor K3 claims CUDA Graph support
   (K3 says static shapes remove the per-layer host sync and "alleviate" launch overhead). Not claimed on slides.

## Summary slide (15) numbers
- "−15% EP comm" = slide 11 at maxvio 20; "11.1% faster steps" = slide 12 at maxvio 15; "1.1 GB vs 7.5 GB" = slide 10
  at EP=8 (analytic); "OOMs at maxvio 20" = slide 3.
