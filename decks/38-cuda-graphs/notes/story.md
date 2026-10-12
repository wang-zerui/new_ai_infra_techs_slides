# Story — deck 38: CUDA Graphs in Megatron-Core (full-iteration / per-layer / TE graphed callables)

- **Where it lives** (prosperity): Megatron-Core training on Hopper/Blackwell. `--cuda-graph-impl` selects
  `full_iteration` (FullCudaGraphWrapper), `local` (per-layer CudaGraphManager) or `transformer_engine`
  (TE `make_graphed_callables` via TECudaGraphHelper); `--cuda-graph-modules` narrows per-layer capture.
  Used in MLPerf Llama 3.1 405B (full iteration), MLPerf GPT-3 175B (TE per-layer) and the tuned DeepSeek-V3
  GB200 configuration (partial per-layer graphs).
- **Pain number** (betrayal): Llama 3.1 405B FP4 training on 512 GB300 runs **1.79×** faster once the whole
  iteration is graphed, i.e. eager launching costs that much; the gap grows with faster GPUs and lower
  precision (1.15× B200, 1.54× GB200) and is absent for FP8 on B200/GB200.
- **Failed suitors**:
  1. Kernel fusion (deck 42): fewer launches, but per-op Python/framework cost and per-kernel launch cost
     remain for everything not fused.
  2. Plain `torch.cuda.graph` / PyTorch `make_graphed_callables`: assume static shapes, static addresses,
     no host syncs and a fixed order; training breaks all four (dropless MoE token counts and their D2H
     copy, NaN checks, fresh dataloader tensors, FP8 global scaling state, PP microbatch interleaving with a
     shared memory pool).
- **Villain**: a captured graph freezes kernels, pointers and order; a training iteration is static almost
  everywhere but a few places (MoE routing, host syncs, per-iteration FP8 state, PP interleaving) break that.
- **Turn / insight**: most of an iteration *is* static. In a dropless MoE layer only dispatch → experts →
  combine has dynamic shapes; attention, router and EP preprocessing are fixed-shape. So graph the largest
  static scope, and replay graphs exactly in their capture order.
- **Mechanisms**:
  ❶ `full_iteration` (challenge: static models still pay per-layer boundaries): FullCudaGraphWrapper copies
     every microbatch's inputs into static buffers, captures the whole forward-backward (all microbatches,
     PP/DP communication) once after warmup, replays it; optimizer stays eager.
  ❷ `local` (challenge: PP interleaving + shared pool): CudaGraphManager records the order of every layer's
     first fwd/bwd during an eager pass, then captures in that order into one pool; with PP one graph pair
     per layer per microbatch (L×M×2), without PP reuse (L×2).
  ❸ `transformer_engine` (challenge: FP8 state + custom schedules): TECudaGraphHelper calls
     make_graphed_callables ahead of time with `_order` = the PP schedule; FP8 amax reduction deferred to once
     per iteration, FP8 weight casts cached and refreshed only on the first microbatch.
  ❹ Partial MoE scopes (challenge: dynamic expert shapes): `--cuda-graph-modules attn moe_router
     moe_preprocess` graphs everything before dispatch; pool sharing and buffer reuse keep memory down;
     sync-free HybridEP + paged stash extends full-iteration graphs to dropless MoE (deck 19).
- **Release**: 1.15× / 1.54× / 1.79× (B200 / GB200 / GB300, FP4, Llama 3.1 405B, full iteration);
  **+10%** end to end for DeepSeek-V3 on GB200 with partial graphs at ~7 GB extra memory; per-layer TE
  graphs +2.2% to +3.0% on GPT-3 175B (256 → 11,616 GPUs).
- **Limits**: memory (static buffers, separate pools, ~7 GB on DeepSeek-V3); warmup and capture time
  (default 3 warmup steps); constraints (no NaN check, TE RNG tracker, NCCL env vars); dropless MoE experts
  stay eager unless sync-free kernels are used; no gain when GEMMs are large (FP8 Llama on B200/GB200);
  turned off on H100 in the paper's DeepSeek-V3 config; debugging an opaque graph is harder.

## Slide plan (15 logical slides, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Small kernels outrun the CPU that launches them | prosperity / hook | fig_eager (SVG) |
| 3 | Eager FP4 training on GB300 runs 1.79× slower | pain | `%% big: 1.79×` |
| 4 | Training breaks every rule a graph needs | villain | fig_rules (SVG) |
| 5 | Fusion and plain graphs only go halfway | failed suitors | two-col text |
| 6 | Insight: almost everything in a step is static | turn | text + boxed sentence (build) |
| 7 | Three implementations, three capture scopes | overview | fig_scopes (SVG) |
| 8 | ❶ full_iteration: one graph per training step | mechanism | fig_full (SVG) |
| 9 | ❷ local: record the order, then capture | mechanism | fig_local (SVG) |
| 10 | ❸ transformer_engine: capture ahead in _order | mechanism | fig_te (SVG) |
| 11 | ❹ MoE: graph everything before dispatch | mechanism | fig_moe (SVG) |
| 12 | Full-iteration graphs: up to 1.79× on FP4 | release | chart llama_full |
| 13 | Partial graphs add 10% to DeepSeek-V3 | release | `%% big: +10%` |
| 14 | Graphs cost memory, warmup and flexibility | limits | two-col text |
| 15 | Summary | | QR |
