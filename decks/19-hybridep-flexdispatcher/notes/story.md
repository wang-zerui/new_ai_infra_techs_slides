# Story — deck 19: HybridEP + Flex dispatcher

- **Where it lives** (prosperity): Megatron-Core MoE (Flex dispatcher since v0.12 with DeepEP; HybridEP
  backend added in v0.15, 2025/11). Fine-grained MoE pretraining such as DeepSeek-V3 (256 experts, top-8,
  58 MoE layers, 116 dispatch/combine per forward pass) at EP64 on two very different networks: H100
  (EP64 = 8 NVLink nodes joined by InfiniBand) and GB200 NVL72 (EP64 inside one NVLink domain, 1.8 TB/s).
- **Pain number** (betrayal): with the standard all-to-all dispatcher, EP communication takes **40–60%** of
  DeepSeek-V3 training time when EP64 crosses H100 nodes, and about **20%** even inside NVL72.
- **Failed suitors**:
  1. NCCL all-to-all dispatcher: permutation replicates each token top-k times, so duplicates cross the
     network; split sizes need a device-to-host sync; permute/unpermute are separate kernels.
  2. DeepEP (v1): token-based, removes the duplicates, but its copy kernels occupy **20 SMs** per GPU
     (~**20%** GEMM-efficiency overhead in the DeepSeek-V3 benchmark) and its kernels assume at most 8
     NVLink peers, so a 72-GPU NVLink domain is out of reach; each backend also had its own wiring.
- **Villain**: SM-driven copy kernels whose topology assumptions are baked in (8-GPU NVLink islands); every
  byte costs SM time, and the MoE layer is coupled to one backend's API.
- **Turn / insight**: the bytes can be moved by hardware engines (TMA bulk copies over NVLink, IBGDA RDMA
  posted by a GPU warp); SMs only need to orchestrate a pipeline, so a few SMs can fill the links. Hide the
  backend behind the dispatcher's fixed six-phase contract so the MoE layer never changes.
- **Mechanisms**:
  ❶ Flex dispatcher (challenge: one MoE code path for many backends and parallel layouts): a single TP×EP
     communication group, routing map expanded to [tokens, TP×EP ranks, local experts], backend managers
     implement setup_metadata / dispatch / combine; one flag selects deepep, deepepv2, hybridep or ncclep.
  ❷ HybridEP dispatch (challenge: dedup traffic with few SMs): persistent warp-specialized blocks, G2S (TMA
     load) → SMEM FIFO → S2G (TMA store over NVLink), RDMA warp group (IBGDA) to the same-local-index GPU
     of every other NVLink domain, then forwarded inside that domain.
  ❸ HybridEP combine (challenge: reduction + unpermute after communication): reduction fused into the
     kernel, hierarchical (pre-reduce inside the domain, RDMA, final reduce), results written in place.
  ❹ NVLink-domain-sized buffers (challenge: NVL72 > 8 peers, dynamic sizes): ranks per NVLink domain
     detected at init via fabric memory handles; one worst-case registered buffer; optional static token
     budget makes dispatch sync-free and CUDA-graph capturable (deck 38).
- **Release**: dispatch latency **1.3–3.9×** lower than all-to-all (GB200 and H100, EP8–EP64); 16 HybridEP
  SMs match DeepEP at **28–36** SMs (B200, derived); end to end **1.14×** over DeepEP for DeepSeek-V3 on
  Grace Blackwell (943 vs 829 TFLOPS/GPU), 1.05–1.10× over all-to-all elsewhere.
- **Limits**: ~3.75–4 GB worst-case registered buffer per rank (EP64 example); dispatch and combine share
  the buffer and cannot overlap each other; nvcc JIT at first use; FP32 probs, BF16 combine, FP8 dispatch
  not used by MCore yet; multi-node build needs DOCA/rdma-core (NIXL path experimental); backend flag is
  "[Experimental]" and the default backend is still deepep; on H100 the paper's best DeepSeek-V3 config
  still uses DeepEP + 1F1B overlap.

## Slide plan (16 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | EP64 spans eight NVLink islands or one domain | prosperity | fig_topo (SVG) |
| 3 | Token exchange eats up to 60% of training | pain | `%% big: 40–60%` |
| 4 | All-to-all ships one copy per expert, not per node | villain (status quo) | fig_a2a (SVG) |
| 5 | DeepEP drops duplicates but costs SMs | failed suitors | two-col text |
| 6 | Insight: engines move bytes, SMs only orchestrate | turn | text + boxed sentence |
| 7 | One Flex dispatcher fronts four backends | overview | fig_stack (SVG) |
| 8 | ❶ Six phases stay fixed; only the manager changes | mechanism | fig_flex (SVG) |
| 9 | ❷ Dispatch: a TMA pipeline in every SM | mechanism | fig_disp_1 → fig_disp_2 (build) |
| 10 | ❸ Combine reduces inside the kernel, twice | mechanism | fig_combine (SVG) |
| 11 | ❹ Buffers sized up front remove the host sync | mechanism | fig_sync (SVG) |
| 12 | HybridEP dispatch is 1.3–3.9× faster than all-to-all | release | chart lat_dispatch |
| 13 | 16 HybridEP SMs match 28–36 DeepEP SMs | release | chart sm_bw |
| 14 | HybridEP lifts DeepSeek-V3 1.14× over DeepEP | release | chart e2e |
| 15 | HybridEP pays in memory and maturity | limits | two-col text |
| 16 | Summary | | QR arXiv |
