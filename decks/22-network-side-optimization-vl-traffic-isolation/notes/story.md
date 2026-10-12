# Story — deck 22: Network-side optimization (VL isolation / adaptive routing / SM-free RDMA)

- **Where it lives** (prosperity): DeepEP's internode EP all-to-all on InfiniBand rail/plane fabrics — DeepSeek-V3's
  2048-H800 multi-plane fat-tree (8 GPU–NIC planes, IB 50 GB/s vs NVLink 160 GB/s); the same fabric carries DP
  all-reduce, PP, storage and checkpoint traffic. Normal kernels (training, prefill) go same-rail then NVLink;
  low-latency kernels (decode) go direct, mostly across the spine.
- **Pain number** (betrayal): DeepSeek-V3 reserves **20 of 132** H800 SMs (15%) just to drive EP communication
  ([V3] §3.5.1). On the network side the pain is the tail: an all-to-all waits for its slowest packet (no public
  number for the collision cost; kept qualitative).
- **Failed suitors**:
  1. One lane for everything: EP packets queue behind bulk DP/storage packets (HOL blocking).
  2. Static/ECMP routing: hashes ignore load, so flows collide on one uplink while another idles; LLM traffic
     "lacks randomness" ([ISCA] §5.2.2).
  3. SM-driven RDMA with a small ring buffer: comm SMs mostly poll for free slots ([PR347]).
- **Villain**: EP all-to-all is treated as ordinary traffic — it shares queues and hash-chosen paths with bulk
  flows, and its NIC is driven by SMs that wait on the wire.
- **Turn / insight**: an all-to-all is as slow as its last packet, and RDMA is one-sided — so the fix is to keep EP
  packets out of other traffic's queues and paths, and to let the NIC move the bytes without SMs watching.
- **Mechanisms**:
  ❶ SL → VL isolation: EP gets its own virtual lane with its own buffers/credits (answers: HOL blocking).
  ❷ Adaptive routing, plus rail-aligned forwarding for LL dispatch (LL-Layered) (answers: hash collisions on
     spine uplinks).
  ❸ SM-free RDMA: Normal-SMFree send phase + recv hook with a large RDMA buffer; LL-SBO signals that let combine
     send stream behind the Down GEMM (answers: SMs idling on the transfer).
- **Release**: on DeepSeek's isolated multi-plane fabric DeepEP holds 42–58 GB/s per GPU from 16 to 128 GPUs
  ([ISCA] Fig. 7); hook mode cuts comm-kernel time 1.8–4.0× (PR #347, H20); SBO +8–10% decode throughput at
  batch 20–56 on 16 H20 ([LMSYS-ANT]).
- **Limits**: fabric-admin work (SL→VL tables, AR on switches); AR adds latency and reorders packets, and spread
  storage incast in Fire-Flyer; SMFree needs ~270 MB RDMA buffer per rank, loses NVLink/RDMA pipelining, and is an
  open PR — mainline V2 keeps 4–6 SMs (down from 24) and says zero-SM RDMA EP is unsupported.

## Slide plan (15 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | EP takes two paths across a rail fabric | prosperity | fig_paths (SVG) |
| 3 | V3 spends 20 of 132 SMs on communication | pain | `%% big:` |
| 4 | Shared lanes and hashed paths stall EP packets | villain (network) | fig_net_before (SVG) |
| 5 | Comm SMs spend the transfer polling | villain (GPU) | fig_sm_1 (SVG timeline) |
| 6 | Insight: an all-to-all is as slow as its last packet | turn | bullets + box |
| 7 | Three fixes at three layers | overview | fig_overview (SVG stack) |
| 8 | ❶ A service level puts EP on its own VL | mechanism | fig_vl (SVG) |
| 9 | ❷ Spread EP over all uplinks, or skip the spine | mechanism | fig_ar (SVG) |
| 10 | ❸ SMs hand bytes to the NIC, then leave | mechanism | fig_sm_2 (same canvas as fig_sm_1) |
| 11 | ❸ SBO sends each GEMM block as it lands | mechanism + result | fig_sbo (SVG) |
| 12 | Isolated planes keep EP above 40 GB/s per GPU | result | chart mpft_bandwidth |
| 13 | Hook mode cuts comm-kernel time 1.8–4× | result | chart smfree_kernel_time |
| 14 | The cost: fabric work, reordering, memory | limits | two-col |
| 15 | Summary | | QR |
