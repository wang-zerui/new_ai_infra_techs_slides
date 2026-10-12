# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Moves from NVSHMEM to NCCL Gin: GPU threads issue one-sided put / put-with-signal directly, header-only, reusing existing NCCL communicators with symmetric memory window registration; the whole library is JIT-compiled.
**After:** Moves from NVSHMEM to NCCL Gin (GDAKI backend required): GPU threads post one-sided put / put-with-signal (plus get, signal, flush) straight to the NIC through NCCL's header-only device API, reusing existing NCCL communicators (or creating one) with symmetric memory window registration; GPU kernels are JIT-compiled, the host extension is built at install.
**Why:**
- "Header-only" applies to NCCL's device API headers that the JIT kernels include, not to DeepEP as a whole ([R2] vs [R25]).
- Only GPU kernels are JIT-compiled; V2.5 "Installation builds the host C++ extension" ([R25]).
- DeepEP V2.5 also uses `get` (Engram), `signal` and `flush_async` ([DEV2]).
- V2.5 requests `NCCL_GIN_TYPE_GDAKI` and asserts if unavailable; the Proxy backend is not used ([CTX25]).
- Communicator reuse falls back to a DeepEP-created communicator when PyTorch lacks `_comm_ptr` ([HDL]).

## Impact / Measured Results
**Before:** Lighter dependency chain and simpler deployment, and the basis for 0-SM communication paths.
**After:** Lighter dependency chain and simpler deployment (NVSHMEM removed); GPU-posted RDMA carries the 0-SM-payload PP and Engram paths, but SMs still post the work and EP dispatch/combine still need SMs.
**Why:**
- V2.5: "EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported" ([R25] Notes); Gin removes the CPU proxy, not the SM that posts the work ([GIN] §III-C).
- NVSHMEM "no longer a dependency" ([R25]; [PR763]).

## Other properties
none
