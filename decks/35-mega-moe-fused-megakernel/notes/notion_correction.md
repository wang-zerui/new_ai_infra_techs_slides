# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Fuses EP dispatch, linear1 (FP8 x FP4 / FP8 x FP8), SwiGLU, linear2 and EP combine into one mega-kernel, using symmetric memory so NVLink communication overlaps Tensor-Core compute inside a single kernel.
**After:** unchanged
**Why:**
- Matches the current README and PR #304 ([RD] lines 112-114, [PR304]). FP8 x FP8 was added after the first FP8 x FP4-only release, but the current README lists both, so the row is right as of now ([RD], [API]).

## Impact / Measured Results
**Before:** Eliminates kernel boundaries and intermediate buffer write-backs; single-stream overlap is friendlier to numerical reproducibility than dual-stream or micro-batch splitting.
**After:** Removes kernel boundaries and the BF16 linear1 output round trip (the FP8 post-SwiGLU activation still goes through an HBM ring buffer); 1.50x to 1.96x over the five-kernel path (V4-Flash and V4-Pro, EP8, 1 to 32768 tokens per rank); in deterministic mode the test asserts bitwise-equal output to the non-fused path.
**Why:**
- Not all intermediate write-backs go away: the L1 epilogue TMA-stores FP8 activations into `l2_token_buffer` in global memory, and the test's HBM model counts "L1 output write" and "L2 acts read" ([K] lines 1066-1255, [TEST]). What disappears is the BF16 [n, 2I] linear-1 output and its re-read by a separate SwiGLU kernel ([TEST] `run_baseline`).
- The reproducibility comparison with dual-stream / micro-batch overlap appears in no primary source; replaced with what the sources show: `torch.equal(fused_y, baseline_y)` under `use_deterministic_algorithms(True)` ([TEST], [PR462]; determinism is opt-in since [PR432]).
- Added the published speedup range: eight points of PR #316, 1.50x (V4-Pro, 8192 tokens/rank) to 1.96x (V4-Flash, 1 token/rank) vs the five-kernel DeepEP + DeepGEMM + TileLang path ([PR316], [TEST] at 891d57b4).

## Other properties
none
