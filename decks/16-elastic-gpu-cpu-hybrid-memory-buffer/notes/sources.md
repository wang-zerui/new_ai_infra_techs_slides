# Sources — deck 16: Elastic GPU/CPU hybrid memory buffer

Local copies (downloaded 2026-10-13): `$SCRATCH/research/15/` (READMEs, `v2/` source at b306af06af,
`main/` source at 93eb6eb238, `mid/` = commits a422b5a369 (2026-05-21) and a56d6156fe (2026-09-16),
`engram/` = Engram paper) and `$SCRATCH/research/16/` (`dispatch.hpp`, `combine.hpp`, `compiled.cuh`,
`test_ep_main.py`, `scripts/ep_buffer_size.py`).

## Sources read

| Tag | Source | Where |
|---|---|---|
| [R2] | DeepEP README at the V2 release commit b306af06af (2026-04-29) | l.29 ("Buffer size consumption is larger than V1"); l.33-37 (still on-going: "Elastic GPU & CPU buffers: A contiguous virtual address space that maps to a hybrid of GPU and CPU physical memory under the hood, enabling fully automatic and transparent Engram or imbalanced EP"; "Reducing intermediate buffer sizes by leveraging EP replay to handle load imbalance"); l.43 (V3-like benchmark setting: 8K tokens per batch, hidden 7168, top-8, FP8 dispatch, BF16 combine) |
| [EB2] | V2 `deep_ep/buffers/elastic.py` l.88-98 | docstring: "'Elastic' refers to the flexibility of underlying memory: currently GPU-only, with CPU and mixed (GPU+CPU) backends on the roadmap" |
| [EB-mid] | `deep_ep/buffers/elastic.py` at a422b5a369 (2026-05-21, PR #637 "Merge some updates") and a56d6156fe | new `num_cpu_bytes` argument ("the number of CPU buffer bytes (e.g. for Engram storage)"), POSIX-FD exchange `create_cpu_handle` + `all_gather_object`, `NCCL_WIN_STRIDE` enlargement; the docstring still says "currently GPU-only" |
| [R25] | DeepEP README at main 93eb6eb238 (V2.5) = `$CACHE/gh-deepep-README.md` | l.12 (ElasticBuffer split into EPBuffer / EngramBuffer / PPBuffer / BucketBuffer); l.17 ("multi-layer Engram storage on GPU or CPU"); l.33, l.40-41 (Engram experimental, needs `ncclGinOptFlagsWarpGet`); l.384-386 (EngramBuffer "supports GPU/CPU storage"; CPU storage example in tests); l.407 (`EP_NUM_MAX_LOCAL_RANKS` = 16 sizing estimate for `NCCL_WIN_STRIDE`) |
| [PR763] | PR #763 "DeepEP V2.5", merged 2026-09-29 | "Engram: Multi-layer storage on GPU or CPU" (tianr22, sphish) |
| [PR605] | PR #605 V2 release body | same on-going list as [R2]; contributors "Networking backends & elastic buffers: sphish, tianr22" |
| [SYM] | main `csrc/kernels/comm/symmetric.hpp` | l.21-76 `DeviceContext` (`CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID`; `CU_MEM_LOCATION_TYPE_DEVICE` vs `CU_MEM_LOCATION_TYPE_HOST_NUMA`; pinned; `gpuDirectRDMACapable`; assert "GPUDirect RDMA with CUDA VMM is not supported"); l.95-109 `set_access` (GPU, plus host NUMA node for the CPU part); l.119-135 `GPUSymmetricMemory` (`ncclMemAlloc`, the default); l.137-190 `ElasticSymmetricMemory` ("The two physical handles share one contiguous VA range": `cuMemAddressReserve`, `cuMemMap` GPU at offset 0, storage at offset `num_sym_bytes`); l.192-300 `HybridElasticSymmetricMemory` (import every node-local storage shard via POSIX FD + `pidfd_getfd`, map them back-to-back after the GPU segment); l.302-330 `alloc` (`NCCL_ELASTIC_BUFFER_REGISTER=1` when CPU-backed) |
| [CTX] | main `csrc/kernels/comm/context.cpp` | l.75-99 (constructor args incl. `num_rdma_storage_bytes`, `use_cpu_rdma_storage`); l.139-143 (`NCCL_GIN_CONNECTION_CUSTOM_STRIDE` when hybrid + storage); l.185-191 (`symmetric::alloc` with `gin_min_stride`); l.196-200 (`ncclCommWindowRegister(..., num_sym_bytes, ..., NCCL_WIN_STRICT_ORDERING)` for the GPU segment); l.207-222 (each storage shard registered with `NCCL_WIN_GIN_ONLY`) |
| [BUF] | main `csrc/buffers/{ep,engram,pp,bucket}.hpp` constructors | only `EngramBuffer` passes `num_rdma_storage_bytes > 0` and `use_cpu_rdma_storage`; EP / PP / Bucket pass 0 (GPU-only, `ncclMemAlloc`) |
| [ENG] | main `deep_ep/buffers/engram.py` l.31-104, `csrc/buffers/engram.hpp` l.251-271 (`calculate_storage_size`), `impls/engram/engram_fetch.cuh` l.91, l.124-129 | "GPU fetch buffer with separate CPU or GPU storage shards registered for RDMA"; GPU part = receive area (layers × tokens × entries × row bytes); storage = all entries × row bytes; GET targets `rdma_peer_rank` (one rank per RDMA peer group) with `storage_windows[intra_peer_rank_idx]` |
| [TEN] | main `tests/engram/test_engram.py` argparse defaults | `--num-entries-per-layer 524288,524309`, `--hidden 256`, `--num-tokens 512`, `--num-entries-per-token 24`, BF16, `--use-cpu-rdma-storage` flag |
| [EPSZ] | main `csrc/buffers/ep.hpp` l.183-282 (`get_dispatch_buffer_size`, `get_combine_buffer_size`, `calculate_buffer_size`), `deep_ep/include/deep_ep/layout/ep/token.cuh` (`TokenLayout`, `BufferLayout`), `csrc/kernels/ep/{dispatch,combine}.hpp` (token layouts), `compiled.cuh` (alignments), `ep.hpp` l.41-43 (`kNumMaxChannels` = 8 × 160) | receive regions sized for `num_scaleup_ranks × num_scaleout_ranks × num_max_tokens_per_rank` tokens; result aligned to 2 MiB |
| [TEP] | main `tests/ep/test_ep.py` l.64, l.582-585 | `num_max_tokens_per_rank = args.num_tokens`, i.e. the token count is per rank |
| [EG] | Engram paper, arXiv 2601.07372 (PDF from github.com/deepseek-ai/Engram) | §2.5 + Fig. 2 p.5-6 (training: tables sharded across GPUs, all-to-all; inference: deterministic prefetch from host memory; multi-level cache hierarchy, Zipfian); Table 1 p.9 (Engram-27B: 5.7B Engram params; Engram-40B: 18.5B); §6.4 + Table 4 p.17-18 (100B-parameter table in host DRAM, H800, nano-vLLM, 512 sequences: 4B-Dense 9,031.62 → 8,858.28 tok/s; 8B-Dense 6,315.52 → 6,140.02 tok/s; "peaking at only 2.8%") |
| [MC] | Megatron-Core MoE report, arXiv 2603.07685 | p.20 (DeepSeek-V3 needs 199.5 GB per GPU without optimization vs the 80 GB of an H100) |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 8.9 GiB | EP buffer per GPU at EP64 from the V2.5 size formula (hybrid mode, 8 GPUs/node, T = 8,192 tokens/rank, hidden 7,168, top-8, FP8 dispatch, BF16 combine) — **analytic** | derived: `scripts/ep_buffer_size.py` = replica of [EPSZ]; settings from [R2] l.43 read as per-rank per [TEP] |
| 0.9 GiB | upper bound on bytes a rank receives under perfectly balanced routing: k·T tokens × 14,400 B (BF16 combine token) — **analytic** | derived: each of N ranks sends each of T tokens to ≤ k ranks → ≤ N·T·k copies spread over N ranks |
| ~10× | 8.93 / 0.88 GiB at EP64 | derived from the two rows above |
| 0.88 / 2.23 / 4.46 / 8.93 / 16.97 / 33.07 GiB | formula output for EP 8 / 16 / 32 / 64 / 128 / 256 (chart) | derived, same script |
| 8,192 B / 14,400 B | bytes per token in the dispatch (FP8 + scales + metadata) and combine (BF16 + top-k weights) layouts | derived from [EPSZ] `TokenLayout` |
| 199.5 GB vs 80 GB | DeepSeek-V3 per-GPU memory need before optimization vs H100 capacity | [MC] p.20 |
| 5.7B / 18.5B | Engram parameters in Engram-27B / Engram-40B | [EG] Table 1 p.9 |
| 200 GB | a 100B-parameter table in BF16 (100e9 × 2 B) — derived | derived from [EG] §6.4 (table size) and BF16 storage |
| 12 MiB vs 514 MiB | GPU fetch area vs host storage per rank for the Engram test defaults (2 layers × 512 tokens × 24 entries × 512 B; (524,288 + 524,309) rows × 512 B, 2 MiB-aligned) — derived | [ENG] `calculate_storage_size` + [TEN] defaults |
| 43× | 514 / 12 | derived |
| < 3% (1.9%, 2.8%) | throughput cost of a 100B-parameter Engram table in host DRAM, Engram paper inference harness (not DeepEP) | [EG] p.2 and Table 4 p.18; 1.9% = 1 − 8858.28/9031.62 (derived); 2.8% quoted |
| 2 MiB | allocation alignment of every segment | [EPSZ] `compiled.cuh` `kNumAllocationAlignmentBytes` |
| 16 | `EP_NUM_MAX_LOCAL_RANKS` default used to size `NCCL_WIN_STRIDE` | [R25] l.407; [ENG] engram.py l.61 |
| Apr / May / Sep 2026 | V2 roadmap item (2026-04-29); first `num_cpu_bytes` in ElasticBuffer (a422b5a369, 2026-05-21); V2.5 EngramBuffer `use_cpu_rdma_storage` (2026-09-29) | [R2], [EB-mid], [PR763] |

Chart `charts/ep_buffer.py` recomputes the formula itself (same code as `scripts/ep_buffer_size.py`).
Chart `charts/engram_offload.py` plots [EG] Table 4.

## Mechanism facts (no numbers) used on slides

- One VA range, two physical backings ([SYM] l.137-179): `cuMemAddressReserve(num_sym + num_storage)`;
  `cuMemCreate` a pinned DEVICE allocation (GPUDirect-RDMA capable) and `cuMemMap` it at offset 0;
  `cuMemCreate` the storage on `CU_MEM_LOCATION_TYPE_HOST_NUMA` (the GPU's host NUMA node) or on the device,
  `cuMemMap` it right after; `cuMemSetAccess`: GPU read/write on the whole range, plus the host NUMA node on
  the storage part.
- Registration split ([CTX] l.196-222): the GPU segment (workspace + fetch/receive buffer) becomes one
  NCCL window with strict ordering (NVLink LSA pointers + Gin); each storage shard is a separate
  `NCCL_WIN_GIN_ONLY` window, i.e. reachable by the RDMA NIC only.
- Node-wide storage ([SYM] l.192-300, [CTX] l.185-191, [ENG] engram.py l.73-77): each rank creates its
  storage shard, exports it as a POSIX FD, `all_gather_object`s `(pid, fd)`; every rank imports the shards
  of its node-local group with `pidfd_open` / `pidfd_getfd` and maps them back-to-back after its own GPU
  segment. A remote Engram GET targets one rank per peer group and names the shard by window
  ([ENG] engram_fetch.cuh l.91, l.124-129).
- Window sizing: if GPU bytes + storage × `EP_NUM_MAX_LOCAL_RANKS` + 4 GiB exceeds GPU memory, DeepEP sets
  `NCCL_WIN_STRIDE` (a TODO calls this env-var route "fragile") ([ENG] engram.py l.61-65; [EB-mid]).
- What uses it in V2.5: only `EngramBuffer`; `EPBuffer`, `PPBuffer`, `BucketBuffer` allocate GPU-only via
  `ncclMemAlloc` ([BUF]).

## Discrepancies (Notion row vs primary sources; primary sources win on the slides)

1. **"transparently supporting Engram and imbalanced EP"**: in V2 this was an *on-going* feature ([R2]
   l.35) and the V2 docstring says "currently GPU-only" ([EB2]). Host-backed storage shipped for Engram only
   (ElasticBuffer `num_cpu_bytes` in May 2026, `EngramBuffer(use_cpu_rdma_storage=True)` in V2.5). No
   released code backs EP buffers with host memory: `EPBuffer` passes zero storage bytes ([BUF]). It is
   also not "fully automatic": the caller sizes the storage and passes `use_cpu_rdma_storage`.
2. **"EP replay compresses intermediate buffers"**: listed as on-going in V2 ([R2] l.36); no replay
   mechanism appears in the V2.5 sources (grep for "replay" finds only cached-layout reuse in the README,
   l.281). The deck reports it as roadmap.
3. **Impact "worst-case load no longer dictates the memory budget"**: not yet true for EP in V2.5; the
   EP size formula still reserves room for N × T tokens ([EPSZ]); the deck shows this analytically.
4. The README describes one "contiguous virtual address space" mapping GPU and CPU memory; the code
   implements it, but registers the two parts as separate NCCL windows (one per storage shard): locally
   the GPU can address the whole range by pointer, while remote access to the storage goes through
   Gin-only (RDMA) windows, not NVLink ([CTX]).
5. Stale docstring: the ElasticBuffer docstring kept "currently GPU-only" after `num_cpu_bytes` landed
   ([EB-mid]).
