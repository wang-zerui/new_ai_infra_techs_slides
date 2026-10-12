# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** --cuda-graph-impl offers three capture-and-replay modes: full_iteration (FullCudaGraphWrapper), local (per-layer CudaGraphManager) and transformer_engine (make_graphed_callables, FP8- and PP-schedule-aware).
**After:** --cuda-graph-impl offers three capture-and-replay modes: full_iteration (FullCudaGraphWrapper: forward and backward of all microbatches, optimizer stays eager), local (per-layer CudaGraphManager, captured in recorded order) and transformer_engine (make_graphed_callables, captured ahead of time from the PP schedule's order); both per-layer modes handle FP8 and PP.
**Why:**
- FP8 and PP awareness is not specific to the TE path: the MCore `local` CudaGraphManager also handles FP8 (deferred amax, weight caching) and PP (recorded order); TE differs by capturing ahead of time from an explicit `_order` ([BG] FP8/PP challenges and comparison table; [CG] `_CudagraphGlobalRecord`, `TECudaGraphHelper`).
- full_iteration scope: forward and backward of all microbatches with their communication; optimizer, LR schedule and logging stay eager ([DOC]; [FCG]; [EX-LLAMA]).

## Impact / Measured Results
**Before:** Removes kernel-launch overhead in small-kernel-heavy regimes; standard in tuned DeepSeek-V3 configurations on GB200.
**After:** Cuts kernel-launch overhead in small-kernel-heavy regimes, for the captured regions only; partial graphs gave 10% end-to-end (about 7 GB extra memory) in DeepSeek-V3 GB200 training and are enabled in its final GB200 config.
**Why:**
- Launch overhead is removed only inside graphs: per-layer modes leave dispatch, expert GEMMs, combine and inter-graph code eager for dropless MoE, and full_iteration leaves the optimizer eager ([DOC] MoE scopes; [MC] p.40-41).
- "Standard in tuned DeepSeek-V3 configurations" replaced by what the paper reports: 10% end-to-end at about 7 GB extra memory from partial graphs with memory optimizations ([MC] §4.3.6 p.42), and CUDA graphs enabled in the final 256-GB200 configuration but not in the H100 one ([MC] Table 17 p.74, App. B.2 p.87).

## Other properties
none
