# Story: DeepEP v2 (container deck)

- **Where it lives** (prosperity): DeepSeek's open-source expert-parallel (EP) library under DeepSeek-V3-style MoE
  training and serving (256 routed experts, top-8, EP64 training across 8 nodes; EP320 decode). V2 (Apr 2026)
  rewrote EP on NCCL's Gin backend; V2.5 (Sep 2026) split the buffers and removed V1/NVSHMEM.
- **Pain number** (betrayal): DeepSeek-V3 spends **20 of 132 SMs** per H800 on all-to-all communication
  (V3 report §3.5.1); V1's example config pins 24 SMs. The release reports SMs, so the pain is in SMs.
- **Failed suitors**:
  1. V1 high-throughput kernels: fixed 20-24 SMs plus a per-EP-size tuning table (10 sizes, "TODO: automatically
     tune"); every other EP size asserts out.
  2. V1 low-latency kernels: a second API and buffer mode with QPs pinned to local-expert count, on a second
     runtime (NVSHMEM + CUDA IPC handshakes, driver regkeys).
- **Villain**: SM and QP counts were treated as tuning constants, while they are really a function of
  topology, routing statistics and link bandwidth; and the transport lived outside the framework's NCCL world.
- **Turn / insight**: communication SMs only have to keep the *bottleneck link* busy — one per-SM HBM write stream
  (≈ 45-50 GB/s) is the same order as a whole 400 Gb/s NIC (50 GB/s) — so the SM budget is a small, computable number.
- **Mechanisms** (container = components, each a sibling deck):
  ❶ one `EPBuffer` API for train / prefill / decode (deck 17)
  ❷ hybrid vs direct kernels with warp-role channels (scale-out, forward, notify) (decks 17, 21)
  ❸ analytic SM/QP budget, no auto-tuning (deck 17)
  ❹ NCCL Gin backend: GPU-initiated RDMA on PyTorch's own NCCL communicator (deck 21)
  plus FP8 dispatch / BF16 combine (deck 18), PP / Engram / bucket collectives (deck 15), elastic buffers (deck 16),
  network-side VL isolation and adaptive routing (deck 22); related HybridEP (deck 19), masked grouped GEMM (deck 34).
- **Release**: V3-like training **24 → 4-6 SMs** at equivalent or better speed; up to **1.3×** peak and **4×** fewer SMs
  than V1; EP 8×4 sustains 61 GB/s per GPU with 6 SMs; 24 SMs keep 89-91 % of the 64-SM NVLink bandwidth (derived).
- **Limits**: larger buffers than V1 (fixed max capacity); RDMA low-latency 0-SM EP dropped (EP always needs SMs);
  Hopper+, CUDA 13.1, NCCL 2.32.3; Engram / PP / bucket experimental; SM model assumes balanced gating; IB-tested only;
  no published numbers beyond EP32.

## Storyboard (final: 16 logical slides, 18 PDF pages, 8 minutes)

| # | Headline | Device |
|---|---|---|
| 1 | Title | |
| 2 | Tokens cross two networks twice per MoE layer | fig_hook |
| 3 | All-to-all takes 15% of every GPU's SMs | big 20 of 132 (pain) |
| 4 | V1 grew two kernel families and a tuning table | fig_v1 (problems red) |
| 5 | Insight: SMs only need to feed the slowest link | fig_insight + callout (build) |
| 6 | Four layers share one NCCL communicator | fig_stack (overview, deck refs) |
| 7 | ❶ One EPBuffer serves training, prefill and decode | bullets, deck 17 / 34 refs |
| 8 | ❷ Hybrid dispatch: rail RDMA, then NVLink | fig_hybrid_1 → fig_hybrid_2 (build) |
| 9 | ❸ A formula sets SM and QP counts | fig_budget |
| 10 | ❹ Warps post RDMA writes through NCCL Gin | fig_gin |
| 11 | Gin also carries PP, Engram and collectives | bullets, decks 15/16 |
| 12 | 6–12 SMs drive 61–91 GB/s of RDMA per GPU | chart rdma_bw |
| 13 | V3-style training drops from 24 SMs to 4–6 | big 24 → 4–6 (release) |
| 14 | 24 SMs keep 89–91% of peak NVLink bandwidth | chart nvlink_bw |
| 15 | Adopted widely, but needs bigger buffers | two-col adoption / costs |
| 16 | Summary | QR |
