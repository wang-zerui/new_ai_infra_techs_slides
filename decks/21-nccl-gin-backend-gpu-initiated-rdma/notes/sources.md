# Sources: NCCL Gin backend (GPU-initiated RDMA) — deck 21

Research copies: `$SCRATCH/research/21/` (GIN paper PDF + text) and `$SCRATCH/research/24/` (shared DeepEP research).
`$SCRATCH` = `$SCRATCH`.

## Sources read

| Tag | Source | What was used |
|---|---|---|
| [GIN] | K. Hamidouche et al. (NVIDIA), "GPU-Initiated Networking for NCCL", arXiv 2511.15076v2 (Nov 2025), https://arxiv.org/abs/2511.15076 ; text at `$SCRATCH/research/21/arxiv-2511.15076.txt` | §I-II (host- vs device-initiated, GPUDirect Async, NVSHMEM IBGDA), §III-A (three layers, one-sided put / put-with-signal, windows, contexts = QPs, signals vs counters, ordering), §III-B (device API, Listing 1), §III-C + Table I (GDAKI vs Proxy), §IV (DeepEP integration: 24 QPs HT, 8-16 QPs LL, 4 contexts per communicator in NCCL 2.28, zero-byte put + SignalAdd), §V (EOS H100 results), Table III |
| [R2] | DeepEP README at V2 release, commit b306af06af / PR #605 | "switched from the NVSHMEM backend to the more lightweight NCCL Gin backend"; "Header-only & lightweight; Able to reuse existing NCCL communicators"; NCCL ≥ 2.30.4; acknowledgement of the NCCL team |
| [R25] | DeepEP README `main` (V2.5), cached `$SCRATCH/research/cache/gh-deepep-README.md` | NVSHMEM "no longer a dependency"; NCCL ≥ 2.32.3; "PyTorch and DeepEP must load the same NCCL shared library"; `EP_REUSE_NCCL_COMM` (default 1), `EP_DISABLE_GIN` ("RDMA operations require Gin; this flag does not select another RDMA backend"), `EP_*_RDMA_SL`; Engram needs `ncclGinOptFlagsWarpGet`; InfiniBand fully tested, RoCE theoretical |
| [PR763] | https://github.com/deepseek-ai/DeepEP/pull/763 | "This release also fully removes V1 and its NVSHMEM dependency" |
| [CTX25] | `csrc/kernels/comm/context.cpp` on `main` (`$SCRATCH/research/24/src_main/`) | reuses the communicator; `ncclDevCommRequirements`: `ginType = NCCL_GIN_TYPE_GDAKI` (asserts GDAKI support), `ginContextCount = num_allocated_qps`, `ginExclusiveContexts = true`, `ginQueueDepth` 1024 default, `ginSignalCount = num_ranks + 4`, `ginConnectionType` RAIL (hybrid) / FULL (direct), `ginTrafficClass` = SL; `ncclCommWindowRegister(..., NCCL_WIN_STRICT_ORDERING)`; `ncclGetLsaDevicePointer` |
| [HDL] | `deep_ep/comm/handle.py` on `main` | `get_nccl_comm_handle`: reuse `backend._comm_ptr()` when available and `EP_REUSE_NCCL_COMM=1`, else all-gather an NCCL unique id and create a DeepEP-managed communicator |
| [DEV2] | `deep_ep/include/deep_ep/common/handle.cuh` and `comm.cuh` at b306af06af (`$SCRATCH/research/24/src_v2/`); `comm/barrier.cuh`, `impls/ep/hybrid_dispatch.cuh`, `impls/engram/engram_fetch.cuh` on `main` | device wrapper `NCCLGin`: `put`, `put_value`, `signal` (`ncclGin_VASignalAdd`), `get`, `flush_async`; teams World / LSA / Rail; LSA pointers via `ncclGetLsaPointer`; `get_qp_mode` (CTA vs GPU resource sharing); Gin barrier via `signal` + polling the signal table; hybrid dispatch: `gin.put<ncclTeamTagRail>` then tail update; forward warps poll signaled tails |
| [V1code] | `deep_ep/buffer.py` at v1.2.1 (`$SCRATCH/research/24/src_v1/buffer.py` lines 85-115) | V1 init: all-gather device ids, all-gather CUDA IPC handles, all-gather NVSHMEM unique id; 8 `NVSHMEM_*` environment variables set before init |
| [V1nvshmem] | `third-party/README.md` at v1.2.1 | NVSHMEM ≥ 3.3.9; IBGDA needs a driver regkey (`NVreg_EnableStreamMemOPs=1`, `PeerMappingOverride=1`) + reboot, or GDRCopy; NVSHMEM SLA |
| [Azure] | Microsoft Azure HPC blog "Achieving optimal performance for DeepSeek Expert Parallelism (DeepEP) on Azure" (2025-05), https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/achieving-optimal-performance-for-deepseek-expert-parallelism-deepep-on-azure/4414699 (listed in the Notion row) | deployment friction with V1/NVSHMEM: extra `NVSHMEM_HCA_LIST` / `NVSHMEM_ENABLE_NIC_PE_MAPPING` settings, QP creation failure on an Ethernet VF (used in notes only) |

## Numbers on slides

Slide numbers are logical. PDF pages: slides 1-4 = p1-4, slide 5 = p5-6 (build), slides 6-16 = p7-17.

| Slide | Number | Source |
|---|---|---|
| 3 pain | **8** `NVSHMEM_*` variables set by V1 before init (`DISABLE_P2P`, `IB_ENABLE_IBGDA`, `IBGDA_NUM_RC_PER_PE`, `QP_DEPTH`, `MAX_TEAMS`, `DISABLE_NVLS`, `CUMEM_GRANULARITY`, `DISABLE_MNNVL`) | [V1code] lines 96-111 (counted) |
| 3 / 4 | **3** all-gather handshakes at init (device ids, IPC handles, NVSHMEM unique id) | [V1code] lines 85-115 (counted) |
| 4 | driver regkey + reboot, or GDRCopy; NVSHMEM ≥ 3.3.9 | [V1nvshmem] §2 |
| 5 insight | NCCL 2.28 Device API: LSA, Multimem, GIN | [GIN] abstract, §I |
| 7 ❶ | `ginContextCount = num_allocated_qps` (up to 129), exclusive contexts, queue depth 1024, signals = ranks + 4, RAIL vs FULL connections, traffic class = SL | [CTX25] lines 121-155 |
| 7 | NCCL 2.28 gave 4 contexts per communicator, so the v1.2.1 port needed ⌈QPs / 4⌉ communicators | [GIN] §IV-B |
| 8 ❷ | GDAKI: GPU threads build WQEs in device memory and write the NIC doorbell; NIC executes RDMA and writes completions to GPU memory | [GIN] §III-C |
| 9 ❷ | signal completing at the destination guarantees all earlier puts to that peer on the same context are visible | [GIN] §III-A "Ordering Semantics" |
| 10 ❸ | GDAKI needs ConnectX-6 Dx or newer + CUDA 12.2+; Proxy uses 64-byte descriptors in lock-free host queues and one CPU proxy thread per communicator | [GIN] §III-C, Table I |
| 10 | DeepEP V2.5 requests GDAKI explicitly and asserts if unavailable | [CTX25] (`reqs.ginType = NCCL_GIN_TYPE_GDAKI`, `props.ginSupport[NCCL_GIN_TYPE_GDAKI]`) |
| 11 chart `p2p_latency` | put-with-signal ping-pong round trip, 4-128 B messages, two H100s: GIN GDAKI **16.7 µs**, GIN Proxy **18.0 µs**, NVSHMEM IBGDA **24.3 µs**, NVSHMEM IBRC **16.0 µs** | [GIN] §V-A |
| 11 | **−31 %** GDAKI vs IBGDA | derived: 1 − 16.7/24.3 = 0.313 |
| 12 chart `deepep_parity` | DeepEP v1.2.1 ported to GIN vs NVSHMEM on EOS (H100, 8×400 Gb/s IB per node, 24 SMs): HT BF16 dispatch, 2 nodes: **84.36 vs 84.97 GB/s** RDMA; LL BF16 dispatch, 1 node with NVLink: **185.28 vs 182.15 GB/s**; LL BF16 dispatch, 1 node pure RDMA: **47.00 vs 46.79 GB/s** | [GIN] §V-B, §V-C; Table III; §V intro ("DeepEP benchmarks allocate 24 SMs per GPU") |
| 12 | "within 1-2 %" across scales (HT) / most LL metrics | [GIN] §V-B, §V-C |
| 12 | HT 8 nodes ≈ 53-54 GB/s both backends | [GIN] §V-B (notes only) |
| 13 chart `ll_latency` | LL BF16 dispatch latency with NVLink: 1 node **40.62 vs 41.43 µs**; 2 nodes **142.51 vs 157.00 µs** (GIN vs NVSHMEM) | [GIN] §V-C |
| 13 | **9 %** lower at 2 nodes | [GIN] §V-C ("9% lower at 2 nodes"); check 1 − 142.51/157.00 = 0.092 |
| 13 notes | pure RDMA 8 nodes: 34-35 GB/s, 219-225 µs both backends | [GIN] §V-C |
| 14 big | **8 → 0** NVSHMEM variables; remaining RDMA knobs are `EP_*` flags (`sl_idx` / `EP_DEFAULT_RDMA_SL` / `EP_OVERRIDE_RDMA_SL`, `EP_REUSE_NCCL_COMM`, `EP_DISABLE_GIN`) | [V1code]; [R25] env-var table; [PR763] |
| 15 limits | NCCL ≥ 2.30.4 (V2) / 2.32.3 (V2.5); same NCCL library in PyTorch and DeepEP; builds with NCCL < 2.31 need exact version match; Engram needs `ncclGinOptFlagsWarpGet`; RDMA requires Gin | [R2], [R25] |

## Discrepancies (Notion row vs primary sources; the primary source wins on slides)

1. **"header-only"** — the V2 release README says "Header-only & lightweight" ([R2]); V2.5 still builds a host C++
   extension against NCCL and JIT-compiles kernels with DeepJIT ([R25]). "Header-only" describes NCCL's device API
   headers that the JIT kernels include, not the whole stack. Slides say "device API headers included by JIT kernels".
2. **"the whole library is JIT-compiled"** — true for GPU kernels in V2/V2.5; the host extension is compiled at install
   time in V2.5 ([R25] "Installation builds the host C++ extension"). Slides: "GPU kernels JIT-compiled".
3. **"put / put-with-signal"** — correct per [GIN] §III-A. DeepEP V2.5 additionally uses one-sided `get` (Engram) and
   `signal` / `flush` ([DEV2]); slides list all.
4. **"basis for 0-SM communication paths"** — the V2 release advertised 0-SM Engram/PP/CP ([R2]); V2.5 states
   "EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported" ([R25] Notes). Gin removes the CPU
   and the proxy from the data path; the SMs still post the work. Slides do not claim 0-SM EP.
5. **"reusing existing NCCL communicators with symmetric memory window registration"** — correct ([HDL], [CTX25]);
   note the fallback: with older PyTorch (no `_comm_ptr`) DeepEP creates its own NCCL communicator.
6. **Performance evidence** — the only published GIN-vs-NVSHMEM numbers ([GIN] §V) are for **DeepEP v1.2.1 ported to
   GIN** on H100, not for DeepEP V2's rewritten kernels. Every result slide says so. The Notion row gives no numbers.
7. **Backend choice** — the GIN paper presents GDAKI and Proxy as interchangeable; DeepEP V2.5 requires GDAKI and
   asserts otherwise ([CTX25]). Slides say so on the backend slide.
