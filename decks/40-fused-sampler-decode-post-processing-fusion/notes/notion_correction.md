# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Folds repetition penalty, temperature scaling, Top-K, Top-P, softmax, random sampling and penalty-mask updates into two kernels (temperature-only cases take an even lighter path); small top-k uses local heap/merge selection, and top-k reduction is merged with softmax statistics to cut vocabulary passes.
**After:** Folds repetition penalty, temperature scaling, Top-K, Top-P, softmax, random sampling and penalty-mask updates into two kernels (the temperature-only path is a single kernel); a block radix sort keeps a rolling top-K (bounded by max_topk 32 or 64) per block, and softmax statistics are gathered in the same pass to cut vocabulary passes.
**Why:**
- The temperature-only path is one kernel whose last-arriving block reduces the partials ([KT] header, l.84+), not just a "lighter path".
- Top-k selection is not a heap/merge: keeper threads hold a rolling top-K that each chunk merges into with `cub::BlockRadixSort` ([K] l.91-113, l.226-263; PR #51 "BlockRadixSort top-k path"). The README wording "local heap/merge" does not match the code.
- The candidate count is bounded by the compile-time `max_topk` (32 or 64) ([API], [H]).
- Online max/sum in K1 lets K2 form full-vocab probabilities without a second vocabulary pass ([K] l.265-288, l.329-579); kept, reworded.

## Impact / Measured Results
**Before:** Up to 8.5x over vLLM-style PyTorch / FlashInfer at vocab 120832 and batch 1-512 (largest win in small-batch RL rollout).
**After:** 4.6x to 8.5x over vLLM-style PyTorch and 2.1x to 5.2x over FlashInfer, at vocab 120832 (BF16 logits) and batch 1-512.
**Why:**
- 8.5x is the maximum over the vLLM-style PyTorch path only; over FlashInfer PR #51 reports 2.1x to 5.2x ([PR51] Benchmark line; [RM] Performance row).
- "Largest win in small-batch RL rollout" has no source: nothing mentions RL rollout or the batch size at which the maximum occurs ([RM] only says fragmented kernels hurt small-batch serving); dropped.
- BF16 logits from the benchmark config ([BEN]); the GPU is not stated in PR #51, so none is named.

## Other properties
none (the page body's LMSYS link is annotated "sampling benchmark results", but the blog has no sampler content, and the FlashInfer paper covers attention only; the body is out of scope)
