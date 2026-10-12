# Story — deck 15: 0-SM PP / CP / Engram communication primitives

- **Where it lives** (prosperity): DeepEP V2 (Apr 2026) and V2.5 (Sep 2026), DeepSeek's communication
  library, grows beyond MoE expert parallelism into pipeline parallelism (PP), context-parallel
  all-gather (CP) and Engram (DeepSeek's hashed N-gram conditional memory, arXiv 2601.07372) table
  lookups. All three are experimental.
- **Pain number** (betrayal): DeepSeek-V3 dedicates **20 of 132** H800 SMs to communication [DSV3
  §3.5.1]; Megatron measures **~20%** GEMM-efficiency overhead from DeepEP's 20 SMs per GPU [MC §4.2.3].
- **Failed suitors**:
  1. Use fewer SMs: DeepEP V2's EP kernels go from 24 to 4–6 SMs, but EP cannot reach zero because combine
     reduces and dispatch lays tokens out per expert; both need arithmetic [I178, R25 l.39].
  2. Dedicated communication hardware (co-processor / SHARP) is a wish in the DeepSeek-V3 report, not a
     feature of today's GPUs [DSV3 §3.5.1].
- **Villain**: SMs used as DMA engines. Communication kernels hold CTAs for copying, forwarding,
  reducing and waiting, while tensor cores idle [DSV3 §3.5.1].
- **Turn / insight**: PP send/recv, CP all-gather and Engram lookups are pure data movement to addresses
  known before launch; a GPU node already has two engines that move bytes without SMs (the RDMA NIC via
  GPUDirect, the copy engines over NVLink). So SMs only post descriptors and signals.
- **Mechanisms**:
  ❶ PP over RDMA put: credit-based slot ring, one thread posts the put with a remote signal; the NIC
     carries the payload (answers: point-to-point activations between stages on different nodes).
  ❷ CP all-gather on copy engines: `cudaMemcpyBatchAsync` into peers' symmetric buffers plus stream
     memory-op flags, no kernel; V2.5 hybrid: CTAs post RDMA puts, CEs forward over NVLink (answers:
     every rank needs every shard).
  ❸ Engram fetch over RDMA get: warps turn deterministic indices into one GET per row from sharded
     storage (GPU or host DRAM); one wait hook per layer (answers: many small, scattered reads).
- **Release**: the transfer phase of PP / CP / Engram holds **0 SMs**, versus 20 (DeepSeek-V3) and
  4–6 (V2 EP). No bandwidth or end-to-end numbers are published for the three primitives.
- **Limits**: experimental (V2.5 README drops the "0 SM" label); short TMA staging kernels for PP and a
  spinning receiver; issue warps for Engram; reduce-scatter/all-reduce still on SMs; PP needs an
  RDMA-only group and adjacent ranks; NCCL Gin (GDAKI) ≥ 2.32.3, Engram needs `ncclGinOptFlagsWarpGet`;
  EP itself keeps SMs.

## Slide plan (14 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | DeepEP now moves bytes for PP, CP and Engram | prosperity | fig_paths (SVG, 4 pattern panels) |
| 3 | Communication holds 15% of the SMs | pain | `%% big: 20 / 132` |
| 4 | Communication SMs mostly copy and wait | villain | fig_smgrid (SVG, 132-cell grid + task list) |
| 5 | Fewer SMs is not zero SMs | failed suitors | two-col text |
| 6 | Insight: moving bytes needs no SM | turn | boxed sentence |
| 7 | Each primitive rides a byte mover | overview | fig_overview (SVG, node anatomy + 3 paths) |
| 8 | ❶ PP: a thread posts, the NIC sends | mechanism | fig_pp_1 / fig_pp_2 (build) |
| 9 | ❷ CP all-gather runs on copy engines | mechanism | fig_cp_1 / fig_cp_2 (build) |
| 10 | ❸ Engram lookups become RDMA reads | mechanism | fig_engram (SVG) |
| 11 | Transfers drop from 20 SMs to zero | release | chart sm_budget |
| 12 | Only sums and routing still need SMs | result/boundary | fig_coverage (SVG) |
| 13 | Zero SMs on the wire, not zero cost | limits | two-col text |
| 14 | Summary | | QR |
