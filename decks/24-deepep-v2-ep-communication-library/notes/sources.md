# Sources: DeepEP v2 (EP communication library) — deck 24

Research copies live in `$SCRATCH/research/24/` (README history, PR bodies, source snapshots).
`$SCRATCH` = `$SCRATCH`.

## Sources read

| Tag | Source | What was used |
|---|---|---|
| [R25] | DeepEP README, current `main` (V2.5, commit 93eb6eb, 2026-09-30), https://github.com/deepseek-ai/DeepEP ; cached at `$SCRATCH/research/cache/gh-deepep-README.md` | News (V2.5 / V2), New features, Notes, Requirements, EPBuffer examples, Bucket, LB, Engram, PP, env vars (`EP_REUSE_NCCL_COMM`, `EP_DISABLE_GIN`, `EP_*_RDMA_SL`), Network configurations, Experimental branches, Community forks |
| [R2] | DeepEP README at the V2 release, commit b306af06af (PR #605, 2026-04-29), `$SCRATCH/research/24/README_b306af06af.md` | "V2 release" news, New features list, Notes, "Still on-going features", **Performance table**, "V2 vs V1" sentence |
| [PR605] | PR #605 "[Public release 26/04] Introducing EPv2", https://github.com/deepseek-ai/DeepEP/pull/605 (merged 2026-04-29; +14,948 / −3,932 lines, 106 files) | Same feature list + perf table as [R2]; adds "CUDA graph compatible, or faster CPU sync in saving memory mode" |
| [PR763] | PR #763 "DeepEP V2.5", https://github.com/deepseek-ai/DeepEP/pull/763 (merged 2026-09-29) | Buffer split (EPBuffer/EngramBuffer/PPBuffer/BucketBuffer), BufferAllocator, LB primitives, DeepJIT, V1 + NVSHMEM removed; requirements |
| [R1] | DeepEP README, V1 era, commit 2be0d4f8e9 (2025-10-21), `$SCRATCH/research/24/README_2be0d4f8e9.md` | V1 performance tables (H800, CX7), V1 example `Buffer.set_num_sms(24)`, "Auto-tuning on your cluster" notice, fixed-size-buffer notice (issue #39), V1 roadmap "SM-free kernels" |
| [V1code] | `deep_ep/buffer.py` at tag v1.2.1, `$SCRATCH/research/24/src_v1/buffer.py` | `num_sms = 20` default; `get_dispatch_config` / `get_combine_config` tables with 10 EP sizes (2..160) marked `# TODO: automatically tune`; assert "Unsupported number of EP ranks"; low-latency mode `num_qps_per_rank` must equal local experts; NVSHMEM env vars + 3 all-gathers at init |
| [V1nvshmem] | `third-party/README.md` at v1.2.1 | NVSHMEM ≥ 3.3.9, IBGDA via driver regkey + reboot or GDRCopy, NVSHMEM SLA |
| [V25code] | `deep_ep/buffers/ep.py`, `deep_ep/utils/envs.py`, `csrc/buffers/ep.hpp`, `csrc/kernels/comm/context.cpp`, `deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh`, `deep_ep/include/deep_ep/common/compiled.cuh` on `main`, `$SCRATCH/research/24/src_main/` | `get_theoretical_num_sms` / `get_theoretical_num_qps` formulas; QP allocation 65/129/17; hybrid dispatch warp roles (notify / scale-out / forward), TMA + `gin.put` over the rail team + TMA stores to NVLink peers; fixed-capacity buffer sizing |
| [V2code] | `deep_ep/buffers/elastic.py`, `csrc/kernels/backend/nccl.cu`, `deep_ep/include/deep_ep/common/{comm,handle}.cuh` at b306af06af, `$SCRATCH/research/24/src_v2/` | V2.0 SM model constants (200/50 GB/s per SM, ×1.25), Gin device-communicator setup (contexts = QPs, rail vs full connection, traffic class = SL), symmetric window via `ncclMemAlloc` + `ncclCommWindowRegister` |
| [V3] | DeepSeek-V3 Technical Report, arXiv 2412.19437, cached `$SCRATCH/research/cache/arxiv-2412.19437.txt` | §3.2.1 computation:communication ≈ 1:1; §3.2.2 NVLink 160 GB/s vs IB 50 GB/s, ≤ 4 nodes per token, 3.2 experts/node, 20 SMs suffice, 10 channels, warp specialization; §3.4.2 decode EP320 with IBGDA; §3.5.1 "20 out of the 132 SMs", hardware wish to unify IB and NVLink |
| [GIN] | Hamidouche et al., "GPU-Initiated Networking for NCCL", arXiv 2511.15076v2, `$SCRATCH/research/21/arxiv-2511.15076.txt` | NCCL 2.28 Device API (LSA / Multimem / GIN), GDAKI vs Proxy backends (used only for one sentence in deck 24; details in deck 21) |
| [MCore] | Megatron-LM MoE README, cached `$SCRATCH/research/cache/megatron-moe-README.md` line 311 | FlexDispatcher backend `deepepv2` "Uses the NCCL Gin-based ElasticBuffer dispatch/combine API" (adoption) |

## Numbers on slides (with exact source locations)

Slide numbers are logical slides; PDF pages in parentheses (slides 5 and 8 have a build step).

| Slide (PDF p.) | Number | Source |
|---|---|---|
| 2 (p2) hook | 256 routed experts, top-8 | [V3] §4.2 Hyper-Parameters, PDF p.22 ("256 routed experts … 8 experts will be activated for each token") |
| 2 (p2) | at most 4 nodes per token | [V3] §3.2.2, p.13 ("limit each token to be dispatched to at most 4 nodes") |
| 2 (p2) | NVLink 160 GB/s, IB 50 GB/s | [V3] §3.2.2, p.13 |
| 3 (p3) pain | **20 of 132** H800 SMs for communication | [V3] §3.5.1, p.20 ("we allocate 20 out of the 132 SMs available in the H800 GPU for this purpose") |
| 3 (p3) | 1:1 computation-to-communication ratio | [V3] §3.2.1, p.12 ("approximately 1:1") |
| 3 (p3) headline | **15%** | derived: 20 / 132 = 0.152 |
| 4 (p4) figure | `set_num_sms(24)`; 10 hand-tuned EP sizes; QPs = local experts | [R1] training example (`Buffer.set_num_sms(24)`); [V1code] `get_dispatch_config`/`get_combine_config` tables with keys 2, 4, 8, 16, 24, 32, 64, 128, 144, 160 and `# TODO: automatically tune`; [V1code] `__init__` docstring ("the low-latency mode requires that this number equals to the number of local experts") |
| 4 notes | V1 default `num_sms = 20` | [V1code] line 30 |
| 5 (p5-6) insight | one SM writes ≈ 45–50 GB/s (model constant) | [V25code] `deep_ep/utils/envs.py` `get_sm_write_gbs()` returns 45; [V2code] `elastic.py` default `sm_write_gbs = 50` |
| 5 (p5-6) | 400 Gb/s NIC drains 50 GB/s | [R1] "CX7 InfiniBand 400 Gb/s RDMA network card (~50 GB/s maximum bandwidth)"; [V25code] `get_rdma_gbs` returns `rate / 8` |
| 7 (p8) ❶ | — (API semantics only: `do_cpu_sync`, expanded layout, cached handle) | [R25] "Example use in model training and inference"; [V25code] `EPBuffer.dispatch` docstring |
| 8 (p9-10) ❷ | 4 notify warps per SM; one warp = one channel; channel → QP mapping | [V25code] `csrc/kernels/ep/dispatch.hpp:20` (`kNumNotifyWarps = 4`); `hybrid_dispatch.cuh` lines 57, 77 ("a warp is a channel"); `comm::get_qp_mode` in [V2code] `comm.cuh` |
| 9 (p11) ❸ figure | +30 % headroom, round to 4; QPs direct min(SMs, 9), hybrid 16 × SMs + 1 | [V25code] `get_theoretical_num_sms` (`align(max(4, ceil(num_sms * 1.3)), 4)`), `get_theoretical_num_qps` |
| 10 (p12) ❹ | — (mechanism: Gin context per QP, symmetric window, signal ordered after puts on the same context) | [V25code] `context.cpp` (`reqs.ginContextCount = num_allocated_qps`, `ncclCommWindowRegister`); [GIN] §III-A "Ordering Semantics" |
| 10 notes | NCCL ≥ 2.30.4 (V2), ≥ 2.32.3 (V2.5) | [R2] Requirements; [R25] Requirements |
| 12 (p14) chart `rdma_bw` | SM90 CX7 EP 8×2: 90 / 81 GB/s, 12 SMs; SM90 CX7 EP 8×4: 61 / 61 GB/s, 6 SMs; SM100 CX7 EP 8×2: 90 / 91 GB/s, 12 SMs; 8K tokens, hidden 7168, top-8, FP8 dispatch, BF16 combine; logical bandwidth incl. local-rank traffic | [R2] / [PR605] "Performance" table rows 1-3 and the note under it |
| 12 (p14) headline | **6–12 SMs**, **61–91 GB/s** | range over the same three rows |
| 13 (p15) big | **24 → 4–6** SMs, equivalent or better performance | [R2] New features ("For V3-like legacy training, SM usage reduced from 24 to 4 - 6 while maintaining equivalent or better performance") |
| 13 (p15) | up to **1.3×** peak, up to **4×** fewer SMs vs V1 | [R2] Performance ("Comparing with V1, V2 achieves up to 1.3x peak performance, while saving up to 4x SM count") |
| 14 (p16) chart `nvlink_bw` | SM100 EP8 NVLink: 726 / 740 GB/s with 64 SMs (max perf); 643 / 675 GB/s with 24 SMs (min #SM) | [R2] Performance table rows 4-5 |
| 14 (p16) | **89 %**, **91 %**; **2.7×** fewer SMs | derived: 643/726 = 0.886, 675/740 = 0.912; 64/24 = 2.67 |
| 15 (p17) | Megatron-LM FlexDispatcher `deepepv2` backend | [MCore] line 311 |
| 15 (p17) | DeepEP-Ascend: same API on Ascend 950 | [R25] News |
| 15 (p17) | UCCL-EP (community fork), Mori-EP (experimental branch, ROCm) | [R25] Experimental branches, Community forks |
| 15 (p17) | Hopper+, CUDA 13.1, NCCL 2.32.3 (PyTorch 2.10 in notes) | [R25] Requirements; [PR763] Notes |
| 15 (p17) | larger buffers than V1; 0-SM RDMA low-latency EP dropped; EP needs SMs | [R2] Notes; [R25] Notes ("EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported") |
| 15 notes | SM model assumes a balanced gate | [V25code] comment in `get_theoretical_num_sms` ("this is for balanced gate … For V3.0's group-limited gate, please do not use this function") |
| 15 notes | InfiniBand fully tested, RoCE theoretical | [R25] Network configurations |
| 16 (p18) summary | 20 of 132; 24 → 4–6; 61–91 GB/s from 6–12 SMs | as above |

## Things deliberately NOT put on slides
- V1 bandwidth tables ([R1]: 153/158 GB/s NVLink; 43-58 GB/s RDMA) are not compared bar-to-bar with the V2 table:
  different token count (4096 vs 8K), top-4-group routing in V1, and V2 reports *logical* bandwidth including
  local-rank traffic. Only the release notes' own V1→V2 ratios (1.3×, 4×, 24 → 4-6) are used.
- Larger-EP numbers: the V2 README explicitly omits them ("We omit results for larger EP configurations").

## Discrepancies (Notion row vs primary sources; the primary source wins on slides)

1. **"Several times lower SM usage than V1"** — consistent with [R2] ("several times fewer SM resources",
   "saving up to 4x SM count"). OK.
2. **"scale-up/out domains up to EP2048"** — matches the V2 release README [R2]; the current V2.5 README [R25] no longer
   states a number ("Larger scale-up & scale-out domain support"), and no published benchmark goes beyond EP32 (8×4). The deck therefore does not put EP2048 on a slide.
3. **"FP8 dispatch with IB virtual-lane isolation and adaptive routing"** — FP8 dispatch and VL isolation are documented;
   note the adaptive-routing advice changed between V1 and V2: V1 [R1] recommended adaptive routing only under heavy
   load and static routing under light load; V2 [R2]/[R25] recommends enabling adaptive routing "under all network load
   conditions". Slides use the V2 wording.
4. **Child list "elastic GPU/CPU buffers"** — in the V2 release this was listed under "Still on-going features"
   ([R2]); V2.5 ships multi-layer Engram storage on GPU **or** CPU ([R25] News), not the transparent hybrid
   GPU+CPU virtual address space. Slide 11 says "RDMA gets from GPU or CPU tables" and the notes say the hybrid buffer was listed as on-going in V2.
5. **Child list "0-SM PP/CP/Engram"** — "0 SM" wording is from the V2 release ([R2]); the V2.5 README drops the 0-SM
   label: PP send/recv uses TMA copies on SMs around a Gin put, Engram uses RDMA gets, and only the bucket all-gather is
   copy-engine driven (`num_sms=0`) while reduce-scatter/all-reduce use SMs ([R25] Bucket collectives). Slide 11 says
   "all-gather on copy engines, reduce-scatter on SMs" and the notes attribute the 0-SM label to the V2 release.
6. Notion names the API **ElasticBuffer**; V2.5 renamed/split it into `EPBuffer`, `EngramBuffer`, `PPBuffer`,
   `BucketBuffer` ([R25] News, [PR763]). Slides use `EPBuffer` and mention the rename.
