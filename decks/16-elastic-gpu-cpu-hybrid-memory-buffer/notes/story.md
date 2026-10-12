# Story — deck 16: Elastic GPU/CPU hybrid memory buffer

- **Where it lives** (prosperity): DeepEP V2 / V2.5 communication buffers. Every buffer is one pinned,
  NCCL-registered, symmetric address range (same layout on every rank) so that NVLink peers and the RDMA
  NIC can address it by (rank, offset). V2 calls it `ElasticBuffer`; "elastic" refers to the memory behind
  it. HBM is already oversubscribed in MoE training (DeepSeek-V3 needs 199.5 GB per GPU before
  optimization vs 80 GB on H100, [MC]).
- **Pain number** (betrayal): the EP buffer is sized for the worst case: N·T tokens may land on one rank.
  At EP64 with the README's V3-like settings the V2.5 size formula reserves **8.9 GiB per GPU**, about
  **10×** what perfectly balanced routing can touch (≤ k·T tokens, 0.9 GiB) — analytic. And Engram tables
  (5.7B–18.5B parameters in the paper's models, a 100B table in its offload test) outgrow HBM outright.
- **Failed suitors**:
  1. Keep everything in HBM, sized for the worst case: the V2 default; GiBs idle on most steps.
  2. Shrink the buffer and handle overflow (the V2 roadmap's "EP replay"): extra passes, still on the
     roadmap.
  3. A separate host buffer with staging copies: second address space and window, extra copies through
     HBM, and kernels must know which side a byte is on.
- **Villain**: a communication buffer must be one registered, symmetric address range, so *where* bytes
  physically live was tied to *how* they are addressed: all of it in HBM.
- **Turn / insight**: CUDA virtual memory separates the two. One reserved VA range can map GPU pages for the
  hot part and pinned host pages (on the GPU's NUMA node) for the bulk; GPUDirect RDMA reaches both. Offsets
  and symmetry are preserved, only residency changes.
- **Mechanisms**:
  ❶ One VA range, two backings (`cuMemAddressReserve`, `cuMemCreate` DEVICE vs HOST_NUMA, `cuMemMap`,
     `cuMemSetAccess`) — answers: keep one pointer + offset space.
  ❷ Two kinds of windows: GPU segment = normal NCCL window (NVLink + RDMA); each storage shard =
     `NCCL_WIN_GIN_ONLY` window (NIC only) — answers: host pages must not be on the NVLink load/store path.
  ❸ Node-wide shards: POSIX-FD export / `pidfd_getfd` import, every rank maps all node-local shards
     contiguously; one RDMA peer per node serves any shard — answers: one connection per remote node.
- **Release**: shipped for Engram (V2.5 `EngramBuffer(use_cpu_rdma_storage=True)`): in the test config the
  GPU holds a 12 MiB fetch area for 514 MiB of rows per rank (43×, derived). Engram paper: a 100B-parameter
  table in host DRAM costs < 3% inference throughput (not a DeepEP measurement).
- **Limits**: EP overflow into host pages and EP replay are still roadmap in V2.5 (EPBuffer is GPU-only);
  needs GPUDirect RDMA with CUDA VMM support, NCCL elastic-buffer registration, `NCCL_WIN_STRIDE` env
  workaround; host rows cross PCIe, so they must be fetched ahead (deterministic indices); experimental.

## Slide plan (14 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | DeepEP buffers are registered, symmetric ranges | prosperity | fig_symmetric (SVG) |
| 3 | EP buffers reserve 8.9 GiB per GPU at EP64 | pain | `%% big: 8.9 GiB` (analytic) |
| 4 | Worst-case sizing grows with EP size | villain/escalation | chart ep_buffer (analytic) |
| 5 | Engram tables outgrow HBM outright | escalation | two-col text |
| 6 | Today's options cost HBM or add a copy path | failed suitors | two-col text |
| 7 | Insight: an address need not imply HBM | turn | boxed sentence |
| 8 | Hot bytes in HBM, bulk in host pages | overview | fig_layout (SVG) |
| 9 | ❶ Reserve once, map two kinds of pages | mechanism | fig_vmm_1/2 (build) |
| 10 | ❷ Host shards are registered for RDMA only | mechanism | fig_windows (SVG) |
| 11 | ❸ Every rank maps its whole node's shards | mechanism | fig_hybrid (SVG) |
| 12 | 12 MiB of HBM fronts 514 MiB of rows | release | `%% big: 12 MiB` (tone: ours) |
| 13 | A 100B table in host DRAM costs < 3% | release (Engram paper) | chart engram_offload |
| 14 | Engram ships it; EP overflow is roadmap | limits | two-col text |
| 15 | Summary | | QR |
