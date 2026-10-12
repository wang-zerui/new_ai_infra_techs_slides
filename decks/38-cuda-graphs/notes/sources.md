# Sources — deck 38: CUDA Graphs in Megatron-Core

Downloads are in `$SCRATCH/research/38/` (fetched 2026-10-13 with `gh api` / `curl`).

## Sources read

| Tag | Source | Where |
|---|---|---|
| [DOC] | Megatron Core developer guide, "CUDA Graph". https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/cuda_graph.html (`docs_cuda_graph.txt`; identical to repo `docs/user-guide/features/cuda_graph.md` on main) | Overview tables; local / transformer_engine / full_iteration sections; MoE example; Additional Notes (warmup default 3) |
| [MC] | NVIDIA, "Scalable Training of Mixture-of-Experts Models with Megatron Core", arXiv 2603.07685v2 (cached txt) | §1.2 p.7; §4.3.1 p.35; §4.3.6 p.38-42 (Fig. 22-26); §4.3.7 p.42-47; Table 15 p.73; Table 17 p.74; §9.2.2 p.75; App. B.2 p.87 |
| [BG] | NVIDIA, "Transformer Engine and Megatron-LM CUDA Graph Support" (dl-cuda-graph docs, written against TE v2.2 / MCore core_v0.14.0). https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/te-megatron-cuda-graphs.html (`te_megatron_cg.txt`) | FP8 and PP challenges; make_graphed_callables `_order` example; CudaGraphManager two-phase record/capture; FullCudaGraphWrapper phases; comparison table |
| [EX-LLAMA] | NVIDIA dl-cuda-graph docs, "Llama 3.1 405B" example (MLPerf Training v5.1). https://docs.nvidia.com/dl-cuda-graph/examples/llama-31-405b.html (`examples_llama-31-405b.txt`) | Capture scope (126 layers, DP/PP/TP/CP comms in graph); "Performance Impact" table; memory note (up to 19%) |
| [EX-GPT3] | NVIDIA dl-cuda-graph docs, "GPT-3 175B" example (MLPerf Training v4.1). https://docs.nvidia.com/dl-cuda-graph/examples/gpt-3-175b.html (`examples_gpt-3-175b.txt`) | per-layer TE graphs; 1,536 graphs/GPU; speedup 2.2% / 3.0%; FP8 weight caching "up to 87.5%" |
| [QB] | NVIDIA dl-cuda-graph docs, "Quantitative Benefits" (`cuda-graph-basics_quantitative-benefits.txt`) | "~1-5 μs per kernel" launch-overhead estimate |
| [FCG] | Megatron-LM `megatron/core/full_cuda_graph.py` (main) | `StaticBufferLoader`, `FullCudaGraphWrapper.__call__` (capture when `curr_iteration == cuda_graph_warmup_steps`, `torch.cuda.graph(..., capture_error_mode="thread_local")`, replay) |
| [CG] | Megatron-LM `megatron/core/transformer/cuda_graphs.py` (main) | `_CudagraphGlobalRecord` (records first fwd/bwd of every runner, creates graphs in recorded order to share a mempool); `CudaGraphManager` (`reuse_cudagraphs = pp.size() == 1`, one runner per microbatch otherwise); `TECudaGraphHelper` (`make_graphed_callables(..., _order=order, num_warmup_iters, fp8_enabled, fp8_recipe, fp8_weight_caching=True, _reuse_graph_input_output_buffers=True (TE ≥ 2.7))`) |
| [TC] | Megatron-LM `megatron/core/transformer/transformer_config.py` (main) | `cuda_graph_impl`, `cuda_graph_modules`, `cuda_graph_warmup_steps = 3`, `cuda_graph_use_single_mempool` docstrings |
| [TE] | TransformerEngine `transformer_engine/pytorch/graph.py` (main) | `make_graphed_callables` signature/docstring; `_order` comment (1-indexed chunk ids, positive = forward, negative = backward) |
| [RD] | Megatron-LM MoE README (cached) | lines 13 (2026/01 "CUDA Graph support with fine-grained scopes"), 35 (v0.12 attention-only capture for dropless MoE), 262, 502-515 |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| ~1–5 µs per kernel launch | rule-of-thumb driver + hardware launch overhead (estimate, not a measurement) | [QB] "Estimating Kernel Launch Overhead Reduction" |
| ~128 tokens per expert | M dimension of DeepSeek-V3 expert GEMMs (256 experts) | [MC] §4.3.1 p.35 |
| 1.15× / 1.54× / 1.79× | speedup from full-iteration CUDA graph, Llama 3.1 405B, FP4, 512 B200 / GB200 / GB300 (MLPerf Training v5.1) | [EX-LLAMA] "Performance Impact" table |
| FP8: no speedup (B200, GB200) | same benchmark in FP8 | [EX-LLAMA] text under the table |
| up to 19% | memory saving of full-iteration vs per-layer graphing with PP (various LLMs) | [EX-LLAMA] "Memory savings" |
| 126 layers | Llama 3.1 405B decoder layers inside one graph | [EX-LLAMA] "Scale" |
| L × M × 2 vs L × 2 | graph count with / without PP (L = layers per GPU, M = microbatches) | [MC] §4.3.6 p.42 |
| 96 × 8 × 2 = 1,536 graphs/GPU | GPT-3 175B per-layer graphs at 11,616 GPUs | [EX-GPT3] |
| +2.2% (256 GPUs), +3.0% (11,616 GPUs) | GPT-3 175B end-to-end speedup from per-layer TE graphs (MLPerf v4.1) | [EX-GPT3] "Performance Impact" |
| 10% end-to-end speedup, ~7 GB extra memory | partial CUDA graphs with memory optimizations, DeepSeek-V3 GB200 training | [MC] §4.3.6 p.42 |
| 1,048 TFLOPS/GPU, 256 GB200, CUDA Graphs enabled; H100 config: no CUDA graphs | DeepSeek-V3 final configurations | [MC] Table 17 p.74; App. B.2 p.87 |
| 3 | default `--cuda-graph-warmup-steps` | [DOC] Additional Notes; [TC] |
| ≥ 10 warmup iterations per layer | TECudaGraphHelper sizes TE `num_warmup_iters` so MCore warmup + TE warmup ≥ 10 | [CG] `get_make_graphed_callables_kwargs` comment |
| up to 87.5% | redundant FP8 weight-cast work removed by caching across microbatches (GPT-3 example) | [EX-GPT3] Key Lessons |
| `_order` = [1, 2, 1, 2, 1, 2, −2, −1, −2, −1, −2, −1] | example: 2 model chunks, 3 microbatches | [BG] make_graphed_callables example |

## Mechanism facts (no numbers) used on slides

- A CUDA graph records kernels with their arguments once and replays them with one launch; it requires static
  shapes; dynamic shapes or control flow break capture ([MC] p.39; [DOC] intro).
- Host syncs (`.item()` in NaN checks, D2H copies of per-expert token counts) cannot run inside a graph
  ([DOC] full_iteration requirements; [MC] p.40, p.43).
- Graph inputs need fixed addresses → FullCudaGraphWrapper's StaticBufferLoader copies every microbatch's
  dataloader output into static buffers ([FCG]; [EX-LLAMA]).
- Graphs that share a memory pool must replay in capture order; PP interleaves microbatches, so with PP every
  microbatch needs its own graph ([MC] p.41-42 Fig. 26; [BG] PP challenge).
- FP8: per-graph amax all-reduce would run per layer per replay → captured with `_graph=True`, reduction once per
  iteration; weight cast/transpose cached and refreshed on the first microbatch via `is_first_microbatch` /
  `skip_fp8_weight_update` ([BG] FP8 challenge + how it works; [MC] p.42).
- full_iteration: captured at iteration `cuda_graph_warmup_steps`, contains forward and backward of all
  microbatches with PP/DP/TP/CP communication; optimizer, LR schedule and logging stay eager; requires
  `--no-check-for-nan-in-loss-and-grad`; `--cuda-graph-modules` must be empty ([DOC]; [FCG]; [EX-LLAMA]).
- local: graphable modules own CudaGraphManagers; an eager pass records each runner's first fwd/bwd in
  `_CudagraphGlobalRecord`; `create_cudagraphs()` then captures them in that order into one global mempool
  ([CG]; [BG] "two-phase JIT").
- transformer_engine: TECudaGraphHelper discovers layers, builds sample args per layer per microbatch, calls
  `make_graphed_callables` with `_order` from the PP schedule (and the 1F1B-overlap order when EP overlap is
  on), FP8 recipe and weight caching; the training loop must call `create_cudagraphs()` and
  `cuda_graph_set_manual_hooks()` ([CG]; [DOC]).
- MoE partial scopes: `attn` = `_forward_attention()`, `moe_router` = router (+ shared experts if not
  EP-overlapped), `moe_preprocess` = `MoELayer.preprocess()`; dispatch, expert GEMMs and combine stay eager
  ([DOC]; [MC] p.40-41 static/dynamic lists).
- Memory optimizations: graphs captured in execution order share one pool (`_order`); static input/output
  buffers reused across graphs per PP order; `is_first_microbatch` GPU flag lets microbatches share a graph
  without PP ([MC] p.42).
- Full-iteration graphs for dropless MoE need sync-free dispatch (HybridEP with a token budget), device-initiated
  grouped GEMM, ECHO and paged stashing ([MC] §4.3.7); [DOC] example flags `--moe-flex-dispatcher-backend
  hybridep --moe-expert-rank-capacity-factor <float> --moe-paged-stash` (dev branch).

## Discrepancies

1. **Notion "transformer_engine (… FP8- and PP-schedule-aware)" as if only TE handles FP8/PP.** [BG] states
   that the MCore `local` CudaGraphManager also handles FP8 (deferred amax, weight caching) and PP (recorded
   order) automatically; the TE path differs by capturing ahead of time from an explicit `_order` and by
   needing explicit helper calls. Slides describe all three.
2. **Notion "Removes kernel-launch overhead".** The primary docs scope it more precisely: full_iteration
   excludes the optimizer step; per-layer graphs leave everything between graphs (and dropless expert compute)
   eager ([DOC]; [MC] p.41).
3. **Background doc vs current code on memory pools.** [BG] (written for core_v0.14.0) says the default with
   PP uses a separate pool per microbatch with graph reuse (e.g. 160 graphs for 80 layers); the paper and the
   current code create one graph pair per layer per microbatch, all in one global mempool
   (`CudaGraphManager.global_mempool`; `reuse_cudagraphs` only when PP = 1), i.e. L×M×2 graphs
   ([MC] p.42; [CG]). Slides follow the paper/code.
