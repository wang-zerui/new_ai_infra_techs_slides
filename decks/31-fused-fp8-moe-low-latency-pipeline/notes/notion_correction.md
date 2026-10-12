# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Fuses routing, Gate-Up GEMM, activation quantization, Down GEMM and top-k weighted reduction into a single pipeline; Gate-Up reads original tokens by routing index to drop the gather stage; shared-memory counting relieves global atomics; the low-latency mode trades warp specialization for CTA residency and chains stages with PDL to cut launch bubbles.
**After:** Chains routing count/index, Gate-Up GEMM, activation + FP8 quantization, Down GEMM and top-k weighted reduction as five PDL-linked stages (6 to 7 kernels, HBM intermediates); in the low-latency cp.async path (per-rank intermediate at most 512), Gate-Up reads original tokens by routing index instead of a gathered copy, shared-memory counting replaces per-pair global atomics, and one warpgroup per CTA (no warp specialization) raises CTA residency.
**Why:**
- "Single pipeline" read as one kernel: it is five PDL-chained stages, the count/index stage itself 2 or 3 kernels, with intermediates in HBM ([CPF], [CNT], [EN] allocations).
- The gather-free, no-warp-specialization path is only dispatched when the per-rank intermediate is at most 512 and a multiple of 64; otherwise the TMA warp-specialized path with a gather runs ([EN], [DEF]).
- Shared-memory counting removes the default path's per-(token, expert) global atomic, leaving one per expert per block ([CNT], [DEF]).

## Impact / Measured Results
**Before:** Up to 1.6x (TP) and 1.5x (EP) over vLLM CUTLASS, Triton and SGLang on DeepSeek-V3 / Hunyuan-V3 / Qwen3-235B.
**After:** H20: up to 1.6x (TP8) and 1.5x (EP8) vs the median of vLLM CUTLASS, vLLM Triton and SGLang; vs the best baseline per batch (Hy3): mean 1.08x / 1.21x, max 1.18x / 1.60x; EP8 uses the TMA path, not the low-latency one.
**Why:**
- PR #49 states the 1.6x / 1.5x relative to the median of the compared backends, not each of them ([PR49]); hardware H20 ([PR49]).
- Against the best baseline per batch, LMSYS Tables 4-5 (Hy3, H20) give mean 1.08x (TP8/EP1) and 1.21x (TP1/EP8), maxima 1.18x and 1.60x ([LMSYS]).
- EP8 mode keeps the full intermediate (1536 for Hy3) per rank, above the 512 cp.async limit, so those numbers come from the TMA path ([BR] + [EN]; derived item 3 in sources.md).
- Model list dropped to fit the caveats; the three models remain in the README row ([RD]).

## Other properties
none
