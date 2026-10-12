# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** A CUTLASS-based chunkwise kernel that overlaps intra-chunk computation with cross-chunk recurrent state propagation instead of stalling between two alternating phases; shared by training and prefill and auto-dispatched as a flash-linear-attention backend.
**After:** A CUTLASS-based chunkwise kernel: a token-parallel first kernel does all state-independent intra-chunk work (16-token chunks), then a warp-specialized second kernel runs the serial cross-chunk state chain while TMA warps overlap memory traffic; K3 uses it for training and prefill, but the public v1 is forward-only (FLA backend, inference mode).
**Why:**
- In the public v1 the two kernels run one after the other on the same stream (SRC fwd_launch.cu); the overlap comes from hoisting state-independent work into the token-parallel K1 and from warp specialization in K2 (load/store warps vs MMA warps) (SRC fwd_kernel2.cuh; BLOG §2) (sources.md discrepancy 3).
- 16-token chunks vs FLA's 64 (BLOG §1).
- K3 §5.1.1 says training and prefill, but FlashKDA v1 exposes only flash_kda.fwd and FLA dispatches it under torch.inference_mode(); no backward kernel is public (README) (discrepancy 1).

## Impact / Measured Results
**Before:** Lets linear attention actually saturate the hardware on long training sequences - a prerequisite for using KDA in a 2.8T model.
**After:** Forward pass at T = 8K, 96 heads: 1.85x (H20) / 2.31x (GB200) faster than FLA's Triton chunk_kda at fixed length, up to 3.27x on varlen batches (GB200); no hardware-utilization or training numbers are published.
**Why:**
- No utilization or long-sequence training numbers exist; the only public measurements are forward timings at T = 8K (BENCH-H20, BENCH-GB200: 2.6220 vs 4.8388 ms; 1.0087 vs 2.3271 ms; varlen 1024 x 8 = 3.27x) (discrepancy 4).
- "Prerequisite for using KDA in a 2.8T model" is not stated by any source; K3 says only that FlashKDA "substantially outperforms the Triton reference" (discrepancy 4).

## Other properties
none. Not changed (page body, out of scope): the body links the Kimi-Linear repo as "FlashKDA kernels"; FlashKDA lives in github.com/MoonshotAI/FlashKDA (sources.md discrepancy 2).
