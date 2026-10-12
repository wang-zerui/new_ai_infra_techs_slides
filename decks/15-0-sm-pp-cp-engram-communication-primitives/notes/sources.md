# Sources — deck 15: 0-SM PP / CP / Engram communication primitives

Local copies (downloaded 2026-10-13) live in
`$SCRATCH/research/15/` (READMEs, PR JSON, `v2/` = source at commit b306af06af, `main/` = source at
main 93eb6eb238, `mid/` = intermediate commits, `engram/` = Engram paper + README).

## Sources read

| Tag | Source | Where |
|---|---|---|
| [R2] | DeepEP README at the V2 release commit b306af06af (2026-04-29), https://github.com/deepseek-ai/DeepEP/blob/b306af06af/README.md | l.3 (scope, "zero or minimal SM occupation"), l.9 (V2 news), l.13-25 (new features incl. "0 SM Engram/PP/CP"), l.27-31 (notes: experimental, 0-SM LL EP dropped), l.33-37 (on-going features), l.43-55 (EP perf table, "up to 1.3x / 4x SM") |
| [PR605] | PR #605 "Introducing EPv2 ... Engram/PP/CP supports", opened 2026-04-23, merged 2026-04-29 | body: "experimental 0 SM Engram, 0 SM Pipeline Parallelism, and 0 SM Context Parallelism all-gather"; contributors (Engram: tianr22, sphish, LyricZhao; PP & CP: skip2004, LyricZhao) |
| [R25] | DeepEP README at main 93eb6eb238 (V2.5, 2026-09-30) = `$CACHE/gh-deepep-README.md` | l.3 (scope), l.11-18 (V2.5 news), l.33-41 (features + notes), l.293-328 (BucketBuffer; "All-gather is driven by copy engines (num_sms=0); reduce-scatter and all-reduce use GPU SMs"), l.382-390 (EngramBuffer, PPBuffer) |
| [PR763] | PR #763 "DeepEP V2.5", merged 2026-09-29 | body: dedicated buffers; "Bucket collectives, Engram, and PP remain experimental"; requirements |
| [R1] | DeepEP README at 2be0d4f8e9 (V1, 2025-10-21) | l.7 and l.290: V1 low-latency "hook-based ... does not occupy any SM resource" |
| [I178] | Issue #178 reply by LyricZhao, 2025-05-29 | "combine must be done by GPU" (why EP keeps SMs) |
| [I694] | Issue #694 (2026-07-22) and reply (2026-10-01) | CP was listed in the V2 README but shipped publicly only as V2.5 `BucketBuffer` |
| [C-PP] | `deep_ep/include/deep_ep/impls/pp/pp_send_recv.cuh` (main); V2: `deep_ep/include/deep_ep/impls/pp_send_recv.cuh`, `csrc/kernels/elastic/pp_send_recv.hpp`, `csrc/elastic/buffer.hpp` `pp_send/pp_recv` | send: wait release → TMA stage into send slot → grid sync → one elected thread per CTA issues `gin.put` with remote signal-add; recv: wait arrival → TMA copy-out → `gin.signal` release; `__launch_bounds__(32,1)`; `num_sms` 0/None → all SMs |
| [C-PPpy] | `deep_ep/buffers/pp.py` (main), `ElasticBuffer.get_pp_buffer_size_hint` (V2 `deep_ep/buffers/elastic.py` l.290-307) | adjacent ranks only; slots = send/recv × prev/next × `num_max_inflight_tensors` |
| [C-AG2] | V2 `csrc/elastic/buffer.hpp` `all_gather` (l.375-458) and `csrc/kernels/backend/cuda_driver.cu` | `cudaMemcpyBatchAsync(..., cudaMemcpyFlagPreferOverlapWithCompute)` into NVLink peers' symmetric buffers; flags via `cuStreamBatchMemOp` write/wait value; asserts all ranks in one NVLink domain |
| [C-AG25] | main `csrc/kernels/bucket/all_gather.hpp`, `csrc/kernels/driver/driver.cpp`, `csrc/buffers/bucket.hpp` l.244-257, `impls/bucket/all_gather/{rdma,hybrid}.cuh` | pure NVLink: CE sync → CE push → CE sync, no kernel; pure RDMA: warps post `gin.put`; hybrid: one CTA per RDMA peer posts puts, CE forwards arrived chunks over NVLink; chunk count from a CE/RDMA timeline model; `num_sms == 0` asserted for all-gather; "0-SM reduce-scatter / all-reduce is not implemented" |
| [C-EN2] | V2 `deep_ep/include/deep_ep/impls/engram_fetch.cuh`, `csrc/elastic/buffer.hpp` `engram_fetch` | one CTA per QP; each warp issues one `gin.get` per token row with `ncclGinOptFlagsAggregateRequests`; `flush_async` per peer; separate `engram_fetch_wait_impl` |
| [C-EN25] | main `deep_ep/include/deep_ep/impls/engram/engram_fetch.cuh`, `deep_ep/buffers/engram.py`, `csrc/buffers/engram.hpp` l.251-299, `tests/engram/test_engram.py` | multi-layer fetch, issue warps per (RDMA peer, GIN context), doorbell batching, storage on GPU or CPU (`use_cpu_rdma_storage`), one wait hook per layer; QP model `nic_read_mpps` 45/90 × 10 µs RTT; test defaults |
| [DSV3] | DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv 2412.19437 (`$CACHE/arxiv-2412.19437.txt`) | §3.2.2 p.14 (20 SMs, 10 channels, warp specialization); §3.5.1 p.20 (20 of 132 H800 SMs; SM task list; "tensor cores remain entirely under-utilized"; wish for co-processor / SHARP) |
| [MC] | NVIDIA, "Scalable Training of MoE Models with Megatron Core", arXiv 2603.07685 (`$CACHE/arxiv-2603.07685.txt`) | §4.2.3 p.34 (DeepEP 20 SMs/GPU ≈ 20% GEMM-efficiency overhead); p.24 ("Copy Engine and Compute Engine operate independently") |
| [EG] | DeepSeek-AI & PKU, "Conditional Memory via Scalable Lookup" (Engram), arXiv 2601.07372; PDF from github.com/deepseek-ai/Engram | §2.2 p.4 (hashed N-gram lookup); §2.5 + Fig. 2 p.5-6 (deterministic indices; training shards tables across GPUs with all-to-all; inference prefetch from host); abstract/§1 p.1-2 and §6.4 + Table 4 p.17-18 (< 3% overhead, 100B table in host DRAM) |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 20 of 132 SMs | SMs DeepSeek-V3 dedicates to communication on H800 | [DSV3] §3.5.1 p.20 ("we allocate 20 out of the 132 SMs available in the H800 GPU") |
| 15% | derived: 20 / 132 = 15.2% (headline of slide 3) | derived from the row above |
| ~20% | GEMM-efficiency overhead from DeepEP's 20 SMs per GPU (DeepSeek-V3 benchmark) | [MC] §4.2.3 p.34 |
| 24 → 4–6 SMs | DeepEP V1 vs V2 SMs for "V3-like legacy training" | [R2] l.22; [PR605] |
| up to 4× fewer SMs, up to 1.3× peak | V2 vs V1 EP | [R2] l.55 |
| 0 SMs | V2 "0 SM Engram (RDMA)", "0 SM PP (RDMA)", "0 SM CP (Copy Engine)" | [R2] l.23-25; [PR605] |
| Apr 2026 / Sep 2026 | V2 (PR #605 merged 2026-04-29) / V2.5 (PR #763 merged 2026-09-29) | [PR605], [PR763] |
| 1 thread per CTA posts the put; 1-2 QPs | PP send issue path | [C-PP] (`kNumQPs > 0 and <= 2`) |
| 512 tokens × 24 entries × 2 layers, 512-B rows (hidden 256, BF16) | Engram test defaults (not a measurement) | [C-EN25] `tests/engram/test_engram.py` argparse defaults |
| 24,576 GETs per fetch | derived: 512 × 24 × 2 | derived from the row above |
| 45 / 90 Mpps, 10 µs | NIC read message-rate assumption (without / with fast RDMA atomics) and read RTT used to size QPs | [C-EN25] `engram.py` `get_theoretical_config`, `engram.hpp` l.279 |
| < 3% (1.9% / 2.8%) | Engram paper: throughput cost of a 100B-parameter table in host DRAM (inference, nano-vLLM, H800, PCIe prefetch) — NOT a DeepEP measurement | [EG] p.2 and §6.4 Table 4 p.18 (4B: 9,031.62 → 8,858.28 tok/s; 8B: 6,315.52 → 6,140.02 tok/s); 1.9% is derived (1 − 8858.28/9031.62) |

Chart `charts/sm_budget.py` plots: DeepSeek-V3 report 20 SMs [DSV3]; DeepEP V1 (V3-like) 24 [R2];
DeepEP V2 EP 4–6 (bar drawn to 6, labelled "4–6") [R2]; DeepEP V2 PP / CP / Engram 0 [R2].
The DeepSeek-V3 (20) and V1 (24) bars come from two documents describing related but not identical
setups; the slide says so in the notes.

Derived numbers: 15% = 20/132; 24,576 = 512 × 24 × 2 (test defaults); 1.9% = 1 − 8858.28/9031.62; 2.8% is quoted by [EG].

## Mechanism facts (no numbers) used on slides

- PP send (main): `pp_send_impl` — wait `pp.release[peer] >= send_count − num_max_inflight_tensors + 1`
  (credit), TMA-copy `x` into the local send slot, `cooperative_groups::this_grid().sync()`, then one
  elected lane per CTA posts `gin.put` (split across ≤ 2 QPs) into the peer's recv slot with a
  `ncclGin_StrongVASignalAdd` on `pp.arrival[...]`. The kernel then exits; the NIC moves the payload.
- PP recv: wait `pp.arrival >= recv_count + 1`, TMA-copy recv slot → `x`, grid sync, `gin.signal`
  `pp.release` back to the sender. Launch: `__launch_bounds__(32, 1)` (one warp per CTA), grid = all SMs
  by default ([C-PP], [C-PPpy] "None selects all SMs").
- CP all-gather V2: host-side only — `cudaMemcpyBatchAsync` with `cudaMemcpyFlagPreferOverlapWithCompute`
  copies the local tensor into every NVLink peer's symmetric buffer (peer pointers from
  `ncclGetLsaDevicePointer`), then `cuStreamBatchMemOp` writes a session flag to each peer and waits for
  each peer's flag ([C-AG2]). V2.5 `BucketBuffer.all_gather` keeps this for pure NVLink and adds RDMA and
  hybrid paths in which warps only post `gin.put` and the CE forwards arrived chunks ([C-AG25]).
- Engram fetch: indices `[num_layers, num_tokens, num_entries_per_token]`; each issue warp owns one
  (RDMA peer, GIN context) pair, ballots the indices that belong to its peer, and posts
  `gin.get(storage_window[peer-local rank], row offset → local fetch area)`; only every n-th batch rings
  the doorbell (`ncclGinOptFlagsAggregateRequests`); `flush_async` per peer; `fetch()` returns one
  wait hook per layer ([C-EN25], [R25] l.384-386).
- Engram storage can live in GPU memory or host memory (`use_cpu_rdma_storage`), registered as a
  GIN-only NCCL window ([C-EN25]; mechanism detail in deck 16).
- Engram indices are deterministic functions of input token IDs (hashed N-grams), so they are known
  before the layer runs ([EG] §2.2, §2.5).
- V1 precedent: low-latency EP kernels returned a receive hook so RDMA traffic ran "in the background"
  without SMs ([R1] l.7, l.290); V2 dropped 0-SM low-latency EP ([R2] l.30).

## Discrepancies (Notion row vs primary sources; primary sources win on the slides)

1. **"DP/TP all-gather and reduce-scatter implementations are included."** At V2 these were listed under
   *still on-going* ("All-gather updates and reduce-scatter implementations for DP & TP", [R2] l.37).
   V2.5 ships `BucketBuffer` all-gather / reduce-scatter / all-reduce for **CP and DP** (not TP) ([R25]
   l.14, l.35, l.293), and only all-gather is 0-SM: "reduce-scatter and all-reduce use GPU SMs" ([R25]
   l.328; code asserts "0-SM reduce-scatter is not implemented").
2. **"0 SMs" is about the payload path, not literally no kernels.** PP send/recv launch short
   one-warp-per-CTA TMA staging kernels (grid = all SMs by default) and the receiver spins on the arrival
   signal; Engram fetch launches issue warps that post GETs; hybrid/RDMA all-gather launches CTAs that post
   puts ([C-PP], [C-EN25], [C-AG25]). Only the pure-NVLink all-gather launches no kernel at all. The V2.5
   README no longer uses the "0 SM" label: "Engram (experimental, with RDMA)", "PP (experimental, with
   RDMA)", "Bucket collectives (experimental)" ([R25] l.33-35).
3. **"CP uses the Copy Engine."** True for the V2 NVLink-only all-gather and the V2.5 NVLink path; the
   V2.5 RDMA path uses the NIC and the hybrid path uses both ([C-AG25]). The CP all-gather was not
   publicly usable as a separate API until V2.5 ([I694]).
4. **Impact: "Communication no longer competes with compute for SMs."** Holds for these three
   primitives' transfers; EP dispatch/combine still need SMs ([R25] l.39: "zero-SM RDMA EP is not
   supported"). No source publishes bandwidth or end-to-end numbers for the PP / CP / Engram primitives.
5. Minor: the V2 `ElasticBuffer` docstring says PP and AGRS use NVLink ([C-AG2] `elastic.py` l.93-96)
   while the README says PP uses RDMA; the code issues GIN puts and V2.5 requires an RDMA-only PP group
   ([R25] l.390, l.101). Slides follow the README + V2.5 code.
