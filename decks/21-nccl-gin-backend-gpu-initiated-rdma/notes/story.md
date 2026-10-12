# Story: NCCL Gin backend (deck 21)

- **Where it lives** (prosperity): every RDMA byte DeepEP V2 / V2.5 moves (EP dispatch/combine, Engram gets, PP puts,
  bucket collectives) goes through Gin, the GPU-Initiated Networking part of NCCL's Device API (NCCL 2.28+), on the
  NCCL communicator PyTorch already created.
- **Pain number** (betrayal): V1 needed a second communication runtime for its device-initiated RDMA: **8** `NVSHMEM_*`
  environment variables and **3** all-gather handshakes before the first byte, plus a driver regkey + reboot (or
  GDRCopy) and a separately licensed NVSHMEM.
- **Failed suitors**:
  1. Host-initiated NCCL collectives: the CPU orchestrates, one launch per call, host-device sync — cannot send tokens
     whose destinations are known only inside a running kernel.
  2. NVSHMEM IBGDA (DeepEP V1): device-initiated and fast, but a standalone runtime beside NCCL — own bootstrap, own
     symmetric heap, own QP configuration, own failure modes (Azure needed NIC-mapping fixes).
- **Villain**: device-initiated networking lived outside the framework's communication runtime, so every deployment
  carried two runtimes for one NIC.
- **Turn / insight**: a NIC only needs a work-queue entry and a doorbell write; since NCCL 2.28 the communicator itself
  can hand both to GPU threads (GDAKI), with put-with-signal semantics. Measured: GIN's put-with-signal round trip
  (16.7 µs) is no slower than NVSHMEM's (IBGDA 24.3 µs, IBRC 16.0 µs).
- **Mechanisms**:
  ❶ one communicator, one window: reuse PyTorch's communicator, request one Gin context per QP, register a symmetric
     window; NVLink peers through LSA pointers into the same window — answers "second bootstrap and heap".
  ❷ warps post one-sided `put` / `signal` / `get`; a signal on the same context lands after the earlier puts, so the
     receiver polls a counter instead of a fence — answers "fine-grained, in-kernel token routing".
  ❸ contexts = QPs (exclusive per SM when possible), rail vs full connections, traffic class = IB service level,
     GDAKI required (Proxy exists in NCCL but DeepEP does not use it) — answers "QP parallelism and traffic isolation".
- **Release**: same speed as NVSHMEM inside DeepEP (HT dispatch 84.36 vs 84.97 GB/s; LL 185.28 vs 182.15 GB/s; −9 %
  LL latency at 2 nodes) and **8 → 0** NVSHMEM variables; V2.5 deletes NVSHMEM.
- **Limits**: NCCL ≥ 2.30.4 / 2.32.3 and the same NCCL library in PyTorch and DeepEP; GDAKI-capable NIC (ConnectX-6 Dx+)
  and co-located PCIe; no other RDMA backend; published comparison is a port of DeepEP v1.2.1 on H100 (not V2);
  InfiniBand tested, RoCE theoretical.

## Storyboard (16 logical slides, 8 minutes)

| # | Headline | Device |
|---|---|---|
| 1 | Title | |
| 2 | Token routes are known only inside the kernel | fig_init (host- vs device-initiated) |
| 3 | V1 needed 8 NVSHMEM knobs before one byte | big 8 (pain) |
| 4 | V1 ran a second runtime beside NCCL | fig_v1rt (problems red) |
| 5 | Insight: NCCL can hand GPUs the NIC doorbell | fig_merge + callout (build) |
| 6 | Gin has three layers: setup, calls, backend | fig_layers (❶❷❸) |
| 7 | ❶ Reuse one communicator and one window | fig_setup |
| 8 | ❷ A warp's put becomes a WQE and a doorbell | fig_gin |
| 9 | ❷ A signal after the puts replaces a fence | fig_signal |
| 10 | ❸ Contexts are QPs; GDAKI is mandatory | two-col |
| 11 | GDAKI's round trip: 16.7 µs vs IBGDA's 24.3 | chart p2p_latency |
| 12 | Inside DeepEP, GIN matches NVSHMEM within 2% | chart deepep_parity |
| 13 | GIN cuts two-node decode dispatch latency 9% | chart ll_latency |
| 14 | 8 NVSHMEM knobs become zero | big 8 → 0 (release) |
| 15 | Costs: version lockstep and GDAKI-capable NICs | limits |
| 16 | Summary | QR |
