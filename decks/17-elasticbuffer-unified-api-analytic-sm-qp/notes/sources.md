# Sources: ElasticBuffer unified API + analytic SM/QP budgeting — deck 17

Research copies: `$SCRATCH/research/24/` (shared DeepEP research: READMEs, PR bodies, source snapshots) and
`$SCRATCH/research/17/` (model re-implementations `sm_model.py`, `sm_model_v20.py`).
`$SCRATCH` = `$SCRATCH`.

## Sources read

| Tag | Source | What was used |
|---|---|---|
| [R2] | DeepEP README at the V2 release, commit b306af06af (PR #605, 2026-04-29), `$SCRATCH/research/24/README_b306af06af.md`, https://github.com/deepseek-ai/DeepEP/blob/b306af06af/README.md | "V2 release" features (unified `ElasticBuffer`, analytic SM/QP, 24 → 4-6 SMs), Performance table, "Example use in inference decoding" (handle caching) |
| [PR605] | https://github.com/deepseek-ai/DeepEP/pull/605 | same list + "CUDA graph compatible, or faster CPU sync in saving memory mode" |
| [R25] | DeepEP README, `main` (V2.5), cached `$SCRATCH/research/cache/gh-deepep-README.md` | `EPBuffer` init example (`get_theoretical_num_sms`), dispatch/combine examples, `do_cpu_sync`, expanded layout, deferred epilogues, Requirements |
| [PR763] | https://github.com/deepseek-ai/DeepEP/pull/763 (V2.5) | `ElasticBuffer` split into `EPBuffer` / `EngramBuffer` / `PPBuffer` / `BucketBuffer`; deferred epilogues; cached expanded layouts |
| [EP25] | `deep_ep/buffers/ep.py` on `main` (`$SCRATCH/research/24/src_main/deep_ep/buffers/ep.py`) | `EPHandle` docstring (psum semantics in expand mode), `__init__` (QP allocation 65/129/17, `prefer_overlap_with_compute`), `get_theoretical_num_sms` (lines 431-541), `get_theoretical_num_qps` (543-560), `dispatch` docstring |
| [ENV25] | `deep_ep/utils/envs.py` on `main` | `get_sm_read_gbs` = 180, `get_sm_write_gbs` = 45, `get_nvlink_gbs` (nvidia-smi link sum × 0.9), `get_rdma_gbs` (ibstat rate / 8), `check_fast_rdma_atomic_support` (CA type MT4131) |
| [EL20] | `deep_ep/buffers/elastic.py` at b306af06af (`$SCRATCH/research/24/src_v2/`) | V2.0 version of the model: per-SM 200 / 50 GB/s defaults, dispatch terms only, ×1.25, align to 2 |
| [QP] | `deep_ep/include/deep_ep/comm/barrier.cuh` lines 57-86 on `main` (same code in `common/comm.cuh` at b306af06af) | `get_qp_mode`: notify warps on QP 0; "e.g., 3 SMs and 10 QPs — SM 0: 0 3 6 9, SM 1: 1 4 7, SM 2: 2 5 8" (CTA-exclusive), else all SMs share all QPs (GPU sharing) |
| [EPH] | `csrc/buffers/ep.hpp` on `main` | buffer sizing for direct vs hybrid (fixed max-capacity layouts), channels per SM from shared memory, `kNumMaxChannelsPerSM = 8` |
| [V1code] | `deep_ep/buffer.py` at tag v1.2.1 (`$SCRATCH/research/24/src_v1/buffer.py`) | `num_sms: int = 20`; `get_dispatch_config` / `get_combine_config` with 10 EP sizes (2, 4, 8, 16, 24, 32, 64, 128, 144, 160), `# TODO: automatically tune`, `assert num_ranks in config_map, 'Unsupported number of EP ranks'`; low-latency `num_qps_per_rank` must equal local experts |
| [R1] | DeepEP README, V1 era (commit 2be0d4f8e9) | `Buffer.set_num_sms(24)` example; "Auto-tuning on your cluster: … run all the tests and use the best auto-tuned configuration"; CX7 400 Gb/s ≈ 50 GB/s; low-latency decode example |
| [MCore] | Megatron-LM MoE README, cached `$SCRATCH/research/cache/megatron-moe-README.md` line 311 | FlexDispatcher `deepepv2` backend "Uses the NCCL Gin-based ElasticBuffer dispatch/combine API" (slide 2 notes) |
| [V3] | DeepSeek-V3 report, arXiv 2412.19437 (`$SCRATCH/research/cache/arxiv-2412.19437.txt`) | 256 routed experts, top-8 (§4.2, p.22); NVLink 160 GB/s (§3.2.2, p.13) |

## Numbers on slides

Slide numbers are logical slides. PDF pages: slides 1-4 = p1-4, 5 = p5-6 (build), 6 = p7, 7 = p8-9 (build), 8 = p10, 9 = p11-13 (builds), 10-16 = p14-20.

| Slide (PDF p.) | Number | Source |
|---|---|---|
| 3 pain | **24 SMs** per GPU for V3-like training with V1 | [R2] New features: "For V3-like legacy training, SM usage reduced from 24 to 4 - 6 while maintaining equivalent or better performance"; [R1] `Buffer.set_num_sms(24)` |
| 3 | V1 table: 10 EP sizes, 5 knobs per entry | [V1code] `Config(num_sms, nvl_send, nvl_recv, rdma_send, rdma_recv)`, 10 keys |
| 4 figure | V1 dispatch config rows: EP 8 → 6/256, 6/128; 16 → 36/288, 20/128; 32 → 32/288, 32/128; 64 → 20/288, 28/128; 128 → 20/560, 32/128 | [V1code] lines 233-242 |
| 4 | "48" row = any size outside the table (illustrative choice of 48) → `assert … 'Unsupported number of EP ranks'` | [V1code] line 244 |
| 4 | low-latency QPs = local experts; default 20 SMs | [V1code] `__init__` docstring; line 30 |
| 5 insight figure | ≈ 45 GB/s HBM writes per SM (V2.5 default; 50 in V2.0); NIC 50 GB/s for 400 Gb/s | [ENV25] `get_sm_write_gbs`; [EL20] signature default; [R1] "CX7 InfiniBand 400 Gb/s … (~50 GB/s maximum bandwidth)" |
| 7 ❶ figure `fig_layout` | toy example: 3 local experts, `expert_alignment = 4`, psum = [3, 6, 13], 16 rows exact vs 24-row capacity | **illustrative**; semantics from [EP25] `EPHandle.psum_num_recv_tokens_per_expert` docstring ("psum[i] equals the aligned cumulative count of experts before i plus the actual count of expert i") and [R25] (`do_cpu_sync=False`: "outputs are allocated to the configured capacity, and the expert GEMMs must use `handle.psum_num_recv_tokens_per_expert`") |
| 8 ❷ figure `fig_budget` | +30 %, round to 4, min 4; QPs direct min(SMs, 9), hybrid 16 × SMs + 1 | [EP25] lines 530-560 |
| 9 worked example (`fig_worked_*`) | E[GPUs per token] = **7.27**, E[nodes per token] = **3.61** (EP 8×4, 256 experts, top-8) | **derived** with the code's formula `G·(1 − C(N − N/G, k)/C(N, k))`, N = 256, k = 8, G = 32 / 4 |
| 9 | per delivered byte: RDMA **0.37 B**, NVLink **0.875 B**; RDMA-bound because 0.373/50 > 0.875/160 (holds for any NVLink > 117 GB/s) | **derived** from the code's traffic terms; NVLink 160 GB/s from [V3] §3.2.2 |
| 9 | delivered rate 50 / 0.373 = **134 GB/s**; SM writes 1.26 B per delivered byte → **169 GB/s**; ÷ 50 GB/s per SM (V2.0) = **3.4 SMs**; × 1.25 = 4.2 → **6 SMs** (ceil, even) | **derived**, V2.0 constants [EL20] |
| 9 | release table runs EP 8×4 with **6 SMs** at **61 GB/s** | [R2] Performance table row 2 |
| 9 | V2.5 constants give **8 SMs** for the same case (raw 4.50 from the combine-write term, × 1.3 = 5.85 → 6 → align 4 → 8) | **derived**, [EP25] + [ENV25] |
| 10 ❸ figure `fig_qp` | 3 SMs × 10 QPs mapping (0 3 6 9 / 1 4 7 / 2 5 8) | [QP] source comment |
| 10 | allocated QPs: hybrid 65 (NIC CA type MT4131, fast RDMA atomics) or 129; direct 17; QP 0 = notify warps in hybrid | [EP25] `__init__` lines 327-335; [QP] |
| 11 big | **24 → 4–6** SMs, equivalent or better performance | [R2] New features |
| 12 chart `rdma_bw` | SM90 EP 8×2: 90 / 81 GB/s, 12 SMs; SM90 EP 8×4: 61 / 61 GB/s, 6 SMs; SM100 EP 8×2: 90 / 91 GB/s, 12 SMs | [R2] Performance table rows 1-3 (8K tokens, hidden 7168, top-8, FP8 dispatch, BF16 combine, logical bandwidth incl. local traffic) |
| 13 chart `sm_model` (analytic) | model output 8, 8, 4, 4 SMs for EP 8×4 / 8×8 / 8×16 / 8×32 vs V1's fixed 24 | **analytic**: `charts/sm_model.py` re-implements [EP25] `get_theoretical_num_sms` with RDMA 50 GB/s, NVLink 160 GB/s, per-SM 180/45 GB/s, 132 SMs; RDMA-bound at all four sizes for NVLink > 117 GB/s; V1 line from [R1] |
| 14 chart `nvlink_bw` | SM100 EP8: 726 / 740 GB/s at 64 SMs; 643 / 675 GB/s at 24 SMs; **89 % / 91 %** (derived 643/726, 675/740); 2.7× fewer SMs (64/24) | [R2] Performance table rows 4-5 |
| 14 | `prefer_overlap_with_compute=False` raises the SM count to at least 64 | [EP25] line 531 (`max(num_sms, 64)`) |
| 15 limits | balanced gate only; `num_scaleout_topk` must be 0; "TODO: support do_expand and allow_multiple_reduction"; "TODO: use architecture-specific values"; bandwidth from `nvidia-smi` / `ibstat` | [EP25] comments; [ENV25] |
| 15 | "Buffer size consumption is larger than V1" | [R2] Notes |

## Discrepancies (Notion row vs primary sources; the primary source wins on slides)

1. **"Communication SMs drop from 24 to 4-6 at equal or better performance in V3-style training"** — matches [R2]
   verbatim in substance. OK.
2. **"726/740 GB/s on SM100 NVLink EP8"** — matches [R2] row 4, but that row uses **64 SMs** ("Max perf");
   the minimum-SM row is 643/675 GB/s at 24 SMs. The Notion row omits the SM count; the slides show both rows.
3. **"handle caching reuses routing metadata"** — true for V2.0 ([R2] decode example: "reusing routing metadata
   across iterations when the gating decisions remain unchanged"). V2.5 ([R25]) narrows it: "A new routing decision
   needs a fresh dispatch; decoding alone does not make an old routing handle reusable"; the cached handle is for
   replaying the *same* routing (e.g., backward). Slides use the V2.5 wording.
4. **"ElasticBuffer"** — renamed/split into `EPBuffer` (+ Engram/PP/Bucket buffers) in V2.5 ([PR763]). Slides say
   "EPBuffer (ElasticBuffer in V2.0)".
5. **"with a new GEMM layout"** — the V2.5 README calls it the "expanded layout for grouped expert GEMMs"
   (`do_expand=True`, `expert_alignment`); slides use that name.
6. The **analytic model's constants changed between V2.0 and V2.5** (per-SM 200/50 → 180/45 GB/s, headroom ×1.25 → ×1.3,
   rounding 2 → 4, combine terms added); not a Notion discrepancy, but the worked example states which constants it uses.
