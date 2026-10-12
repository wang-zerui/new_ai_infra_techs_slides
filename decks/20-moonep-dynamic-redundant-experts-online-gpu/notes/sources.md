# Sources for deck 20 (MoonEP mechanisms deep dive)

Downloads: MoonEP tarball of `master` (commit d86fda4, 2026-10-12) in `$SCRATCH/research/27/repo/MoonEP-master/`
(shared with deck 27); shared cache `$SCRATCH/research/cache/`. Scripts in `$SCRATCH/scripts/`.

## Sources read

| Tag | Source | What was used |
|---|---|---|
| [README] | MoonEP README (cache `gh-moonep-README.md`) | three properties; notation S, K, E, R, epn, NvS, H, H′; weight/grad buffer layout; API; zero_copy semantics |
| [API] | `moonep/api.py` (`_create_context`, `Buffer.dispatch`, `_run_dispatch_on_current_stream`) | NvS formula; dispatch = inter-rank sync + planning + dispatch + epilogue; plan reuse skips planning |
| [PLAN] | `moonep/planning.py` (module docstring, `PlanningKernel.kernel`) | one cooperative launch; Phase A gather of tokens-per-expert to rank 0; rank 0 balance loop in one warp, per-home-rank expert split, layout (cu_seqlens, experts_to_copy, zero-fill ranges), plan multicast via `multimem.st`; other ranks counting-sort their top-k by expert, rank 1 also sorts rank 0's; destination per token copy; dedup encoding |
| [REF] | `tests/planning_reference.py` | readable reference of the same algorithm: balance = group_tokens − S·K; fill loop; expert split (largest quota ← expert with most remaining tokens); layout of 2·epn groups; dst = first dest rank whose cumulative allocation covers the token's global index (`searchsorted`) |
| [DISP] | `moonep/dispatch.py`, `dispatch_epilogue.py`, `combine.py`, `combine_prologue.py` docstrings | one NVLink copy per (token, destination GPU); local fan-out of duplicates; fused FP32 pre-reduction of duplicates before combine; padding rows zero-filled |
| [PF] | `moonep/prefetch.py`, `moonep/grad_reduce.py` docstrings | owner-push TMA weight prefetch; home rank pulls FP32 replica grads over NVLink, accumulates, then each rank zeroes its own consumed slots |
| [CHART-COMM] | `figure/comm_vs_deepep.png` + `benchmarks/bench_vs_deepep.py` | labeled dispatch/combine totals vs DeepEP v2; bar segments (comm / planning / prefetch) |
| [K3] | Kimi K3 report arXiv 2607.24653 (cache) | §5.2.1 (perfect balance, E/R bound, ILP references + near-optimal GPU kernel, zero copy, S×K vs S×K×R, sync-free static shapes, workload-aware expert-GEMM scheduler); App. E Theorems 1-2 |
| [V3] | DeepSeek-V3 report arXiv 2412.19437 (cache) §3.4 | redundant experts from online statistics, adjusted "e.g., every 10 minutes"; dynamic redundancy "requires more careful optimization of the algorithm ... and the fusion with the dispatch kernel" |
| [EPLB] | EPLB README (cache) | plan from estimated loads, "moving average of historical statistics" |
| [DEEPEP] | DeepEP README (cache, V2.5) | `do_cpu_sync=True` exact sizes via CPU sync (training/prefill); `False` = configured capacity |
| [MCORE] | Megatron-Core MoE report arXiv 2603.07685 §4.3.7 | dynamic shapes need a device-to-host sync; worst-case buffer O(EP size) larger |

## Numbers on slides

Slide numbers are logical slides (slide 7 is a 3-step build, so PDF page = slide + 2 from slide 8 on).

| Slide | Number | Source location |
|---|---|---|
| 2 | benchmark shape S=8192 tokens/GPU, K=8, E=384, R=8, H=7168 | [CHART-COMM] title; [README] "H20 with EP=8" |
| 2 | S×K = 65,536 token copies per GPU; E/R = 48 experts per GPU | **derived**: 8192 × 8; 384 / 8 |
| 2 | Kimi K3 MoE: 896 routed experts, 16 per token | [K3] Table 1 (notes) |
| 3 | DeepSeek-V3 re-plans redundant experts periodically, "e.g., every 10 minutes", from online load statistics | [V3] §3.4.1 |
| 4 | dynamic redundancy needs a better global routing algorithm and fusion with the dispatch kernel (paraphrase) | [V3] §3.4.2 last sentences |
| 4 | DeepEP v2 `do_cpu_sync=True` reads exact sizes on the CPU, "commonly used for training and prefill" | [DEEPEP] |
| 4 | worst-case copy-free buffer S×K×R vs MoonEP S×K | [K3] §5.2.1 |
| 5 | ≤ E/R redundant experts per GPU always suffice; some routings need ⌈E(R−1)/R²⌉ | [K3] App. E Theorems 1, 2 |
| 5 | chart: E/R = 96 / 48 / 24 / 12 / 6 and ⌈E(R−1)/R²⌉ = 72 / 42 / 23 / 12 / 6 for E=384 at R = 4 / 8 / 16 / 32 / 64 | **analytic** from the two formulas; `charts/replica_bound.py` |
| 5 | ≤ R−1 fills; each GPU filled at most once | [K3] App. E proof of Theorem 1 |
| 2, 7, 8, 10, 11 | running example R=4, E=8 (E/R=2), S=4, K=2 → S×K=8; expert loads E0..E7 = 9,3,7,5,2,2,2,2; GPU loads 12,12,4,4; per-source counts in the table below | **illustrative**, constructed for the slides; plan computed by hand with the [REF] algorithm (tie-break: smaller index first, as in the kernel's argmax/argmin) |
| 7 | balance = load − S×K: +4, +4, −4, −4; fills z[0→2] = 4, z[1→3] = 4; split: 4 of E0 → GPU 2, 4 of E2 → GPU 3; result 8,8,8,8 | as above |
| 8 | E0's 9 copies in global order (GPU0: 3, GPU1: 2, GPU2: 2, GPU3: 2); cumulative allocation over GPUs [5, 5, 9, 9]; copies 0-4 stay on GPU 0, 5-8 go to GPU 2 | as above ([REF] `searchsorted(alloc_cumsum[e], g, right=True)`) |
| 9 | planning kernel: one cooperative launch, 32 SMs default; rank 0 plans, multicasts | [PLAN]; [API] default num_sms 32 |
| 9 | planning ≈ 40-50 µs ≈ 2% of dispatch fwd (2.4-2.6 ms) | **estimated** from [CHART-COMM] green segments (linear calibration on labeled tops, slope 3.25 µs/px): 42 / 42 / 52 / 52 µs at maxvio 0.2 / 1 / 10 / 20; shares 1.6 / 1.6 / 2.0 / 2.2% of the labeled totals 2598 / 2644 / 2629 / 2413 µs |
| 10 | NvS = align(S·K + (P−1)·2·E/R, P); toy P = 4 → NvS = 20, cu_seqlens on GPU 2 = [4, 8, 12, 12] | [API] formula; toy values **derived** |
| 10 | benchmark NvS = 77,824 slots (65,536 real + padding slack), P = 128 | **derived** from [API] formula |
| 11 | pool: E/R slots per projection, shared by all layers; FP32 grad reduce buffer of the same shape | [README] weight / gradient buffers |
| 12 | headline "prefetch up to 29%" = max estimated prefetch share (754 / 2,644 = 28.5%) |  **estimated**, see next rows |
| 12 | dispatch fwd totals: MoonEP 2,598 / 2,644 / 2,629 / 2,413 µs vs DeepEP v2 2,296 / 2,299 / 2,408 / 2,604 µs | [CHART-COMM] labels |
| 12 | MoonEP segments (estimated): comm 1,832 / 1,852 / 1,849 / 1,839; planning 42 / 42 / 52 / 52; prefetch 725 / 754 / 732 / 523 µs | **estimated** from pixel heights (same calibration); prefetch share 28 / 29 / 28 / 22% |
| 12 | raw dispatch comm ≈ 20% below DeepEP v2's dispatch at maxvio 0.2 (1,832 vs 2,296) | derived from the estimate above and a labeled total |
| 13 | combine fwd: MoonEP 1,975 / 1,932 / 1,955 / 1,954 µs; DeepEP v2 2,126 / 2,142 / 2,306 / 2,468 µs | [CHART-COMM] labels |
| 13 | DeepEP v2 +16% from maxvio 0.2 to 20 (2468/2126 = 1.161); MoonEP within 2.2% (1975/1932); −21% at maxvio 20 ((2468−1954)/2468 = 20.8%) | derived |
| 14 | pool at E=384, EP=8, H=7168, H′=2048: 4.2 GB BF16 + 8.5 GB FP32 | **derived** 48 × 3 × 7168 × 2048 × 2 B / × 4 B |
| 14 | per-expert token counts inside a GPU stay skewed; K3 adds a workload-aware expert-GEMM scheduler | [K3] §5.2.1 |
| 14 | plan is exact-feasible but only near-optimal (vs offline ILP references) | [K3] §5.2.1 "Online planning" |
| 15 | summary repeats slides 3, 5, 9, 12, 13 | — |

### Running-example token table (illustrative)

| source GPU | E0 | E1 | E2 | E3 | E4 | E5 | E6 | E7 | sum |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 1 | 2 | 1 | 0 | 1 | 0 | 0 | 8 |
| 1 | 2 | 1 | 2 | 1 | 1 | 0 | 1 | 0 | 8 |
| 2 | 2 | 0 | 2 | 2 | 0 | 1 | 0 | 1 | 8 |
| 3 | 2 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 8 |
| total | 9 | 3 | 7 | 5 | 2 | 2 | 2 | 2 | 32 |

Home GPU of expert e = e // 2. Plan (by the [REF] algorithm): alloc[E0] = {GPU0: 5, GPU2: 4}, alloc[E2] = {GPU1: 3,
GPU3: 4}, all other experts stay home. E2's 7 copies in global order (GPU0: 2, GPU1: 2, GPU2: 2, GPU3: 1), cumulative
allocation [0, 3, 3, 7]: copies 0-2 stay on GPU 1, 3-6 go to GPU 3.

## Discrepancies (Notion row vs primary sources)

1. Notion: "every EP rank receives exactly **S x K / EP** tokens". [README] and [K3]: exactly **S × K** tokens per rank,
   with S = input tokens per rank. Slides use S × K.
2. Notion: "a fixed static S x K buffer". The real allocation is NvS = S·K plus per-group padding slack (≤ (P−1)·2·E/R
   rows, P = 128) rounded to P ([API]); 77,824 vs 65,536 slots at the benchmark shape. Slide 10 shows the padding.
3. Notion: "makes GroupedGEMM a fully static load". The per-GPU total is static, but per-expert segment sizes
   (`cu_seqlens`) still change every micro-batch; K3 adds a workload-aware expert-GEMM scheduler because per-expert
   counts inside a rank stay skewed ([K3] §5.2.1). Slide 14 says so.
4. Notion: "online planning overhead is about zero". The README says "negligible"; the README chart's planning segment
   is ≈ 40-50 µs (≈2% of dispatch), and the weight prefetch that planning triggers is ≈ 0.5-0.75 ms per dispatch
   (estimated), which the Notion row does not mention. Slide 12 shows both.
5. Notion: "CUDA Graph friendly". Not claimed by [README] or [K3] (K3: static shapes remove per-layer host sync and
   alleviate launch overhead). Not claimed on slides.
6. Notion: "with no host involvement" is correct per layer; the host still allocates the NVLink/VMM buffers once at
   `Buffer` construction ([API]).
