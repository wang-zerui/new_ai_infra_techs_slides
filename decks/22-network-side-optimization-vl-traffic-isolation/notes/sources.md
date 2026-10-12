# Sources — deck 22: Network-side optimization (VL traffic isolation / adaptive routing / SM-free RDMA)

Downloads live in `$SCRATCH/research/22/` (PR bodies `pr_*.md`, README versions `README_<sha>.md`, papers and blog text).

## Sources read

| Tag | Source | Where |
|---|---|---|
| [RD] | DeepEP README, current main (cached `gh-deepep-README.md`, V2.5, commit 93eb6eb238) | "Notes" (zero-SM RDMA EP not supported); "Network configurations" (traffic isolation: EP vs other; `sl_idx`, precedence `EP_OVERRIDE_RDMA_SL` > `sl_idx` > `EP_DEFAULT_RDMA_SL` > NCCL default; SL-to-VL mapping set by the fabric; adaptive routing; congestion control; PCI atomic mode); env-var table; tests default `NCCL_IB_SL=1`, `EP_OVERRIDE_RDMA_SL=1`; "Experimental branches" (AntGroup-Opt: Normal-SMFree #347, LL-SBO #483, LL-Layered #500) |
| [RD-V2] | DeepEP README at b306af06af (2026-04-29, EPv2 release) | News ("SM usage reduced from 24 to 4 - 6", "0 SM RDMA low-latency EP is no longer supported", "up to 1.3x peak performance, while saving up to 4x SM count"); Network configurations (AR "under all network load conditions"; CC "disabled because it hurts maximum bandwidth"; low-priority VLs) |
| [RD-V1a] | DeepEP README at ebfe47e46f (initial commit, 2025-02-24) | Network configurations: three VL classes (normal / low-latency / other) via `NVSHMEM_IB_SL`; AR only for low-latency kernels, "may lead to deadlocks or data corruption" for normal internode kernels; CC disabled |
| [RD-V1b] | DeepEP README at 0eee87b8ca (2025-07-16) and 567632dd59 (2026-02-03) | AR "can completely eliminate network congestion caused by routing conflicts, but ... introduces additional latency": enable under heavy load, static routing under light load; CC disabled ("not observed significant congestion in our production environment"); V1 example `Buffer.set_num_sms(24)`; LL kernels' receive hook "without any SM occupation" |
| [AR-FIX] | DeepEP commit 458cdcb22a "Fix AR bugs for normal kernels" (2025-03-05) | AR support for normal kernels |
| [PR347] | DeepEP PR #347 "Feature/sm free normal kernel" (AntGroup, base `antgroup-opt`, state **open**, fetched 2026-10-13) and its comments | Motivation, design (`return_recv_hook`, one channel per SM), §3.1 setup (2/4 nodes × 8 H20, 4096 tokens, h 7168, top-8), §3.2 tables, §3.3 costs (≈270 MB RDMA buffer per rank; NVLink-bound recv/send phases); maintainer comment (sphish, 2025-08-13) |
| [PR390] | DeepEP PR #390 "Single Batch Overlap (SBO)" (closed unmerged into main; re-landed as PR #483 into `antgroup-opt`, merged 2025-11-21) | §2 overlap design (signal per block_m tokens, atomic increments, polling combine send; Dispatch Recv overlapped with shared expert); §4 SGLang evaluation (5 nodes × 8 H20; decode DP-attn 16 + EP16; bs 32) |
| [PR500] | DeepEP PR #500 "opt ll dispatch layered algo" (merged into `antgroup-opt` 2025-12-04) | description (same-rail RDMA then NVLink forwarding for LL dispatch; on by default; `DEEPEP_DISABLE_LL_DISPATCH_OPT=1`; conflicts with elasticity PR #370); benchmark chart image (H20 EP16 vs batch size; H200 bs 128 vs EP size; read qualitatively only); comment "test pass on one/two/four 8*H200" |
| [LMSYS-ANT] | LMSYS blog "SGLang × Ant Group" 2025-09-26 (https://www.lmsys.org/blog/2025-09-26-sglang-ant-group/) | "SBO (Single-batch-overlap)" (why not TBO; out-of-order arrival from "NIC multi-QP scheduling, network congestion and multi-path routing"); Decode results ("+8%–10% improvement in the BS=20–56 range", 16× H20) |
| [V3] | DeepSeek-V3 Technical Report, arXiv 2412.19437 (cached) | §3.2.2 (IB 50 GB/s vs NVLink 160 GB/s; same-index GPU then NVLink forwarding; 20 SMs, 10 channels, warp-specialized forwarding), §3.5.1 (20 of 132 SMs; SM tasks list), §3.4.2 (decode EP via direct point-to-point IB with IBGDA) |
| [ISCA] | Insights into DeepSeek-V3, ISCA'25, arXiv 2505.09343 | §4.4.1 (up to 20 SMs for communication in training; inference EP entirely through NIC RDMA), §4.5 (KV-cache PCIe traffic contends with EP), §5.1 + Fig. 3 (multi-plane fat-tree: 8 GPU–NIC planes, separate RoCE storage plane, 64-port 400G IB switches), §5.1.2 + Fig. 7 (DeepEP on MPFT, 16–128 GPUs), §5.2.1 Table 5 (IB 2.8/3.7 µs), §5.2.2 (ECMP vs AR vs static routing; incast; VOQ), §6.3 (adaptive routing; lossless networks and HOL blocking) |
| [FF] | Fire-Flyer AI-HPC (SC'24), arXiv 2408.14158 | §VI-A1 (four traffic classes → SL → VL, proportions configured, prevents HOL blocking), §VI-A2 (AR spread storage-incast congestion → static routing), §VIII-A (DCQCN disabled) |
| [AZ] | Azure HPC blog "Achieving Optimal Performance for DeepEP on Azure" (2025-05-16) | NUMA affinity, HCA mapping (`NVSHMEM_HCA_LIST`), PCI atomics; results table (2× ND96isr H100 v5) — no VL/AR content |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 20 of 132 SMs | H800 SMs DeepSeek-V3 allocates to communication in training | [V3] §3.5.1 p.20; [ISCA] §4.4.1 p.7 ("up to 20") |
| 15% | 20 / 132 = 15.2% (derived) | derived |
| 50 GB/s IB vs 160 GB/s NVLink | per-GPU bandwidths (H800, CX7 400 Gb/s) | [V3] §3.2.2 p.13 |
| 8 planes; 64-port 400G switches | MPFT topology | [ISCA] §5.1 p.9 |
| 42.47 / 58.02 / 50.58 / 45.34 GB/s (dispatch), 43.05 / 56.96 / 48.54 / 41.6 GB/s (combine) at 16 / 32 / 64 / 128 GPUs | DeepEP algorithm bandwidth on MPFT, 4096 tokens per GPU | [ISCA] Fig. 7 p.10 (data labels; series order from the PDF text layer) |
| > 40 GB/s per GPU | ISCA's summary of Fig. 7 | [ISCA] §5.1.2 p.10 |
| 1358 → 765 µs (EP16), 3535 → 884 µs (EP32) | FP8 dispatch kernel time, native → hook mode | [PR347] §3.2.1 table |
| 2530 → 1381 µs, 6780 → 2468 µs | BF16 dispatch kernel time (EP16, EP32) | [PR347] §3.2.1 table |
| 2590 → 1425 µs, 6850 → 2250 µs | BF16 combine kernel time (EP16, EP32) | [PR347] §3.2.1 table |
| 1.8–4.0× | kernel-time reduction range (derived: 1358/765 = 1.78, 3535/884 = 4.00, 2530/1381 = 1.83, 6780/2468 = 2.75, 2590/1425 = 1.82, 6850/2250 = 3.04) | derived from [PR347] |
| ≈ 270 MB | hook-mode RDMA buffer per rank in the PR author's setup | [PR347] §3.3.1 |
| 2/4 nodes × 8 H20, 4096 tokens, h 7168, top-8 | PR #347 setup | [PR347] §3.1 |
| +8–10% | decode throughput gain of SBO at batch 20–56 on 16× H20 | [LMSYS-ANT] "Decode" section |
| 24 → 4–6 SMs | DeepEP V2 SM usage for V3-like training "while maintaining equivalent or better performance" | [RD-V2] News |
| up to 4× fewer SMs, up to 1.3× peak | V2 vs V1 | [RD-V2] Performance |

Derived numbers: 20/132 = 15.2%; the 1.8–4.0× kernel-time ratios above. The chart `charts/smfree_kernel_time.py`
plots the six PR #347 before/after pairs as published; `charts/mpft_bandwidth.py` plots the eight Fig. 7 labels.

## Mechanism facts (no numbers) used on slides

- An IB service level (SL) is set per connection/QP; each switch/NIC port maps SL → VL through an SL-to-VL table;
  VLs have separate buffers and credit-based flow control on each link, and a VL arbiter shares the link
  ([FF] §VI-A1 citing the IB QoS white paper: flows in distinct lanes "do not interfere"; [RD] "The fabric's
  SL-to-VL mapping determines which virtual lane carries that traffic"). Per-VL credits are IB architecture
  (InfiniBand spec; [FF] §VIII-A "credit-based flow control ... for IB networks").
- DeepEP V2 picks the SL per buffer (`sl_idx`) or by env (`EP_OVERRIDE_RDMA_SL`, `EP_DEFAULT_RDMA_SL`); it does not
  configure VL mapping or congestion control itself ([RD]).
- Adaptive routing: switches distribute traffic across multiple paths; adds latency; recommended under all loads in
  V2 ([RD], [RD-V2]); V1 recommended it only under heavy load ([RD-V1b]); before 2025-03 it was unsafe for normal
  internode kernels ([RD-V1a], [AR-FIX]). ECMP "struggles to distribute traffic" and DP traffic "lacks randomness"
  ([ISCA] §5.2.2). AR causes out-of-order arrival ([LMSYS-ANT]; [ISCA] §5.1 needs out-of-order placement).
- Fire-Flyer saw AR spread storage-incast congestion and chose static routing for that network ([FF] §VI-A2).
- Normal kernels: IB to the same-index GPU on the target node, then NVLink forwarding, deduplicated per node
  ([V3] §3.2.2; [ISCA] §4.3). Low-latency kernels: direct RDMA per destination, via IBGDA ([V3] §3.4.2).
- Normal-SMFree: SMs stall polling a small, reused ring RDMA buffer; fix = large RDMA buffer, send phase returns a
  recv hook, receive phase runs later; one channel per SM ([PR347] §1-2). Hook-mode kernel time = send + recv kernel
  time with the transfer overlapped by a GEMM ([PR347] §3.2.1). Costs: HBM, NVLink-bound phases ([PR347] §3.3);
  loses NVLink/RDMA pipelining ([PR347] maintainer comment).
- LL-SBO: one signal per block_m tokens per local expert; Down GEMM atomically increments; combine send polls and
  sends when the threshold is reached; Dispatch Recv overlaps the shared expert instead ([PR390] §2; [LMSYS-ANT]).
- LL-Layered: LL dispatch first sends to the same-rail rank on the target node, then NVLink-forwards ([PR500]).
- Current DeepEP: "EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported" ([RD]).

## Discrepancies

1. **Notion: "adaptive routing is enabled" as a settled default.** DeepEP's advice changed three times: unsafe for
   normal kernels (Feb 2025), heavy-load only (Mar 2025 – Feb 2026), all loads (V2, Apr 2026). Fire-Flyer chose
   static routing for its storage-heavy network. Slides show the history.
2. **Notion: "VL isolates EP traffic from everything else".** V1 recommended three classes (normal kernels,
   low-latency kernels, other); only V2 reduced it to EP vs other. DeepEP only sets the SL; the VL mapping and
   arbitration weights are fabric configuration ([RD]).
3. **Notion: "returns RDMA-path SMs to compute" / SM-free RDMA.** Normal-SMFree (PR #347) is still an open PR
   against the `antgroup-opt` branch (not merged anywhere, 2026-10-13). Mainline DeepEP V2.5 states zero-SM RDMA EP
   is not supported; V2 instead cut SMs from 24 to 4–6. The PR's speedups are kernel-execution times with the
   transfer hidden under a GEMM, not end-to-end time.
4. **Notion body: Azure blog cited for "adaptive routing, traffic isolation".** The Azure post covers NUMA/HCA
   affinity and PCI atomics; it does not discuss VLs or AR. Not used for those claims.
5. **Notion: "colliding with DP/checkpoint traffic".** DeepSeek's own V3 fabric also puts storage on a separate
   RoCE plane ([ISCA] §5.1), i.e. physical isolation beyond VLs; Fire-Flyer's four classes were HFReduce, NCCL, 3FS
   storage and other. No public source quantifies the EP slowdown from such collisions; slides keep it qualitative.
6. **Sub-page: LL-SBO "+8–10%" vs PR #390.** The LMSYS blog reports +8–10% at BS 20–56; PR #390's bs-32 serving
   runs show +6.7% and +7.5% output throughput (6667 → 7111, 6671 → 7169 tok/s). Slides quote the blog's range
   and name the batch range.
