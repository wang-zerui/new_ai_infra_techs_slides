# Story — deck 34: Masked grouped GEMM (CUDA Graph friendly)

- **Where it lives** (prosperity): MoE decode (and RL rollout) with large expert parallelism. DeepSeek-V3
  decodes with EP320, one routed expert per GPU, usually ≤ 256 tokens per expert; each step is a long chain
  of small, memory-bound kernels, which CUDA Graphs replay with one launch. DeepEP's low-latency dispatch
  feeds DeepGEMM's `m_grouped_fp8_gemm_nt_masked`; SGLang uses this pair for decode.
- **Pain number** (betrayal): a grouped GEMM shaped on the host needs every MoE layer's per-expert token
  counts on the CPU: **58** blocking device-to-host reads per DeepSeek-V3 decode step (derived: one per
  MoE layer), and a shape that depends on GPU data cannot be captured in a CUDA Graph at all.
- **Failed suitors**:
  1. Host-shaped grouped GEMM (contiguous layout, normal dispatch): CPU waits for counts every layer; normal
     dispatch "is not compatible with CUDA graph".
  2. Plain batched GEMM over worst-case slots: graph-capturable, but computes every padded row.
- **Villain**: a CUDA Graph freezes kernel arguments (pointers, shapes, grid size), while MoE routing decides
  the per-expert row counts on the GPU, every step.
- **Turn / insight**: a graph freezes arguments, not memory contents: keep every shape at a fixed capacity,
  put the per-expert counts in device memory, and let a persistent kernel skip the invalid rows.
- **Mechanisms**:
  ❶ Static per-expert slots `[G, max_m, K]` → fixed pointers and shapes (answers: graph needs static args).
  ❷ `masked_m` on the device read by the persistent tile scheduler: ceil(masked_m[g]/BLOCK_M) tiles per expert,
     grid = #SMs (answers: work varies, launch must not).
  ❸ Zero-copy hand-off with DeepEP low-latency dispatch/combine (`recv_x`, `recv_count` = `masked_m`) plus a
     host-side `expected_m` hint for the JIT config (answers: no host knowledge of counts at all).
- **Release**: whole MoE layer capturable; masked FP8 GEMM keeps 815–1233 TFLOPS on H800 and 1.1–1.2× over the
  tuned CUTLASS 3.6 baseline, streaming weights at up to 2064 GB/s at decode-like sizes; SGLang runs
  low-latency dispatch + masked GEMM for CUDA-Graph decode (22,282 tokens/s per H100 node, system-level).
- **Limits**: memory sized for the worst case (slot = tokens per rank × ranks; 32× the balanced load for
  256 experts top-8, derived); overflow is an error; `expected_m` mismatch costs speed; partial last tile;
  successor is the psum layout with device-side prefix sums (DeepEP V2, `do_cpu_sync=False`); on SM100 small
  batches should use Mega MoE (deck 35).

## Slide plan (13 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Decode runs tiny MoE layers as CUDA Graphs | prosperity | fig_graph |
| 3 | Host-shaped GEMMs cost 58 syncs per step | pain | `%% big: 58` |
| 4 | Host shapes stall; worst-case shapes waste | failed suitors | fig_sync (a)(b) |
| 5 | Insight: graphs freeze arguments, not memory | turn | bullets + boxed sentence |
| 6 | Masked layout: fixed slots, counts on the GPU | overview | fig_masked |
| 7 | ❶ Fixed slots keep every pointer and shape static | mechanism | fig_slots_1/2 (build) |
| 8 | ❷ The scheduler reads masked_m on the device | mechanism | fig_sched_masked |
| 9 | ❸ DeepEP low-latency buffers plug straight in | mechanism | fig_ll |
| 10 | Masked GEMM keeps 1.1–1.2× at decode sizes | result | chart masked_tflops |
| 11 | Prefill takes contiguous, decode takes masked | result/adoption | two-col |
| 12 | Static slots cost memory: 32× the balanced load | limits | two-col |
| 13 | Summary | | QR |
