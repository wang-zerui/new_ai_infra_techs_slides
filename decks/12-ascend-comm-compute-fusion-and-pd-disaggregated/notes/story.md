# Story — deck 12: Ascend · comm/compute fusion and PD-disaggregated scheduling

- **Where it lives** (prosperity): GLM-5 (744B MoE, 40B active, DSA, 200K context) served on Ascend Atlas
  nodes through vLLM-Ascend or SGLang; tensor parallelism inside the node, attention data-parallel and MoE
  expert-parallel across it; one tensor-parallel AllReduce after attention and another all-to-all pair around the
  experts, in every one of its 3 dense + 75 MoE layers.
- **Pain number** (betrayal): tensor-parallel communication is 8–11% of step time on an NVLink cluster but
  **40–75%** on a weak-fabric cluster (A100 PCIe, training and prefill) — Flux's measurement, not an Ascend
  one; neither the GLM-5 report nor the Ascend engine publishes its own percentage.
- **Failed suitors**:
  1. Make the collective itself faster — the link is what it is, and the AllReduce still sits between two
     computations that need its result.
  2. Chunked stream overlap (the TransformerEngine-style recipe): splitting one GEMM into one kernel per rank
     gives up device utilization and controllable timing; measured overlap efficiency ranges from **−125% to
     36%** on the weak-fabric cluster, i.e. often slower than not overlapping at all.
  3. Keep prefill and decode on the same nodes: a 200K prefill preempts running decodes and wrecks tail
     latency (the reason GLM-5 gives for splitting them in its own RL rollouts).
- **Villain**: AllReduce is an atomic barrier in the middle of the layer. It serializes compute against the
  fabric, and it leaves every rank holding the full token set, so the norm, the router and the MoE all-to-all
  afterwards do TP-times redundant work.
- **Turn / insight**: an AllReduce is nothing but a ReduceScatter followed by an AllGather. Split it and two
  things fall out: each half now borders a matmul it can be fused and pipelined with tile by tile, and the
  window between the halves is a region where the tokens are sharded 1/TP.
- **Mechanisms**:
  ❶ **FlashComm split** — AllReduce → ReduceScatter + AllGather, with RMSNorm, the router and the all-to-all
    running on sharded tokens in between (answers: the collective is a barrier and forces redundant work).
  ❷ **Fuse the collective into the matmul** — tile the row-parallel projection and send each tile's partial
    as soon as it is produced (`fuse_gemm_comms`; `dispatch_ffn_combine` for the MoE path) (answers: chunked
    overlap loses more in utilization than it wins in hiding).
  ❸ **Split the roles across nodes** — prefill nodes and decode nodes with their own parallelism and their
    own kernels, a Mooncake KV path between them, prefix cache reuse and MTP drafts on the decode side
    (answers: prefill and decode poison each other's latency on shared resources).
- **Release**: the report's claim for the whole stack — one domestic node "comparable to dual-GPU
  international clusters", 50% lower long-sequence deployment cost; the only comparable numbers it publishes
  for this deck are MTP accept lengths (2.76 vs DeepSeek-V3.2's 2.55 at 4 speculative steps).
- **Limits**: FlashComm mainly helps prefill and is incompatible with `o_proj` layer sharding; the fused MoE
  operators are restricted to w8a8 with EP ≤ 32 (or EP ≤ 64) and exclude MTP and dynamic EPLB; the fused
  matmul+reduce-scatter pass covers BF16/FP16 at TP 2/4/8 only; PD costs extra nodes, a KV transport and a
  recompute path; and no per-feature speedup is published anywhere.

## Slide plan (13 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Every layer ends in a tensor-parallel AllReduce | prosperity | fig_layer |
| 3 | On a weaker fabric the collective eats the step | pain | `%% big: 40–75%` (Flux) |
| 4 | Chunked overlap can be slower than none | failed suitors | fig_chunked |
| 5 | Insight: AllReduce is ReduceScatter + AllGather | turn | boxed sentence (build) |
| 6 | Hide it at three scales | overview | fig_overview ❶❷❸ |
| 7 | ❶ The split leaves a sharded window | mechanism | fig_split |
| 8 | ❷ Each matmul tile ships its own partial | mechanism | fig_tiles |
| 9 | ❸ Prefill and decode get their own nodes | mechanism | fig_pd |
| 10 | MTP lands 2.76 tokens per decode step | result | chart accept_length |
| 11 | The report claims one node, half the cost | release | bullets + callout |
| 12 | The fine print on every switch | cost/limits | two-col |
| 13 | Summary | | QR |
