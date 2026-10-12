# Story — deck 39, DeepJIT runtime compilation + PDL + determinism switches

- **Where it lives**: under every DeepGEMM kernel (deck 25). The shapes come from the model at run
  time — tokens per expert, context length, batch size — and DeepGEMM compiles a kernel per shape
  through DeepJIT, with no CUDA build at install time.
- **Pain number**: a power-of-two tile wastes the machine. At M = 256, N = 7168 a 128x128 tile gives
  `(256/128) * (7168/128) = 112` thread blocks on a **132-SM** H800: 20 SMs get nothing. Choosing
  BLOCK_N = 112 instead fills 128 of 132 (S4.2).
- **Failed suitors**:
  1. Ahead-of-time template libraries: the tile set is frozen at build time, so a shape outside it
     falls back to a generic kernel; compiling every variant at install is what DeepGEMM explicitly
     avoids (S4.4).
  2. Runtime autotuning: measure the candidates and keep the fastest — pays a benchmarking cost at
     startup and makes the chosen kernel (and thus the numerics) depend on the machine's mood.
     DeepGEMM selects deterministically instead, "without auto-tuning" (S4.1).
- **Villain**: the gap between when a kernel must be compiled and when its shape is known. Everything
  the scheduler wants as a constant — tile, stages, SM budget, cluster, even the determinism policy —
  is only decided once the model is running.
- **Turn / insight**: compile at the first call, when the shape is known, and the shape stops being a
  variable: N and K are baked in (`compiled_dims = "nk"`), the block scheduler's bounds are constants,
  pipelines unroll fully, and runtime knobs become template arguments. The compile is paid once per
  (shape, config) and cached on disk under a content digest that several processes and nodes share.
- **Mechanisms**:
  ❶ **Deterministic per-shape configuration**: enumerate the legal layouts (swap-AB x BLOCK_M x
     BLOCK_N x cluster) and rank them — single wave first, then multicast, then fewer waves, then
     last-wave utilisation (S3.4, S3.5); emit a `.cu` that instantiates the kernel template with ~25
     constants (S3.2); DeepJIT hashes source + includes + NVCC version + flags + post hook into a
     32-hex digest, publishes the cubin by atomic rename, and reuses it from a shared cache (S1.3-S1.5).
  ❷ **PDL**: launches carry `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`, and every kernel
     calls `cudaGridDependencySynchronize()` *after* its prologue, so the next GEMM's barrier
     initialisation, tensor-memory allocation and register reconfiguration overlap the previous
     kernel's tail (S2.1, S3.8).
  ❸ **Determinism and budget switches**: `use_deterministic_algorithms(True)` keeps the library off
     cuBLASLt, forbids split-K in Mega Gate and fixes mHC's split count, which is what makes a token's
     output independent of its batch position (S3.9, S5.1); `set_num_sms` caps the SM budget so GEMMs
     coexist with communication kernels; `set_tc_util` compiles in idle cycles after each UMMA to keep
     the clock from dropping (S3.6, S3.7).
- **Release**: the knobs are free at run time because they are constants at compile time; the right
  tile fills 128 of 132 SMs instead of 112; V4 could replace cuBLAS end-to-end and keep bitwise
  batch invariance while matching split-K performance (S5.2); DeepGEMM's tests re-run each GEMM 20
  times and require bitwise equality (S3.10).
- **Limits**: the first call to a new shape compiles — SGLang ships a precompile pass with a 3600 s
  default timeout and vLLM a warm-up module (S7); kernels must be compiled and loaded before CUDA
  Graph capture (S1.7); the include parser tracks only literal `#include <prefix/...>` lines and
  untracked dependencies need `extra_signature` (S1.6); header and hook digests are cached per
  process (S1.11); determinism is off by default since the 2026-09 release (S6.1).

## Beat → slide map

| Beat | Slide |
|---|---|
| Prosperity | 2 — a kernel per shape, decided at run time |
| Betrayal (pain number) | 3 — 112 of 132 SMs |
| Failed suitors / villain | 4 — AOT tile sets and autotuning |
| Turn | 5 — compile when the shape is known |
| Overview | 6 — the DeepJIT path |
| Mechanisms ❶❷❸ | 7, 8, 9, 10 |
| Release | 11 — determinism and batch invariance in V4 |
| Humility | 12 — cold start and cache rules |
| Close | 13 — summary + QR |
