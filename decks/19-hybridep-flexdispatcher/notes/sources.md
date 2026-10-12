# Sources — deck 19: HybridEP + Flex dispatcher

All downloads are in `$SCRATCH/research/19/` (fetched 2026-10-13 with `gh api` / `curl`).

## Sources read

| Tag | Source | Where |
|---|---|---|
| [MC] | NVIDIA, "Scalable Training of Mixture-of-Experts Models with Megatron Core", arXiv 2603.07685v2. https://arxiv.org/abs/2603.07685 (cached `arxiv-2603.07685.txt`) | §2.1.1-2.1.3 (p.9-10), §4.2 (p.29), §4.2.1 (p.30), §4.2.2 + Fig. 14-15 + Table 7 (p.30-32), §4.2.3 (p.34), §4.3.7 (p.43-44), §8.1 (p.68), Table 17 + §9.2.2 (p.74-75) |
| [RD] | Megatron-LM `megatron/core/transformer/moe/README.md` (cached `megatron-moe-README.md`) | "What's New" lines 16, 33; perf flags lines 116-118; Token Dispatching table lines 306-313 |
| [BLOG] | F. Yu, T. Liu, K. Sun, "Optimizing Communication for Mixture-of-Experts Training with Hybrid Expert Parallel", NVIDIA Technical Blog, 2 Feb 2026. https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/ (`research/19/blog_hybrid_ep.txt`) | design section (TMA, IBGDA, warp groups, per-block pipeline); test section (8 SMs, ~4 SMs, 16 SMs); Table 1 (end-to-end TFLOPS) |
| [HEP-RD] | DeepEP `hybrid-ep` branch, `docs/README_Hybrid-EP.md` (commit 10d4dd7, 2026-09-11) | design goals; CUDA Graph integration; B200 performance table (DeepEP vs HybridEP vs SM count); GB200 tables |
| [HEP-IMPL] | DeepEP `hybrid-ep` branch, `docs/Hybrid-EP_Implementation.md` | §2 interface (non_blocking, num_permuted_tokens), §4.2 buffer allocation (EP64 example), §4.3 shared buffer, §5 executor, §6 JIT, §7.2 permutation, §8 kernels (G2S/S2G/RDMA/Reduce warp groups, peers) |
| [HEP-CODE] | DeepEP `hybrid-ep` branch: `deep_ep/hybrid_ep_buffer.py` (`HybridEPBuffer.__init__`: `detect_accessible_ranks`, ranks per NVLink domain, `num_of_nodes`); `csrc/hybrid_ep/allocator/allocator.cu` (`CU_MEM_HANDLE_TYPE_FABRIC` allocation, falls back when fabric unsupported); `csrc/hybrid_ep/config.cuh` (`Configurer`: default 24 dispatch/combine SMs for one NVLink domain, 8 when multi-domain); `csrc/hybrid_ep/backend/hybrid_ep_backend.cuh` (TMA-aligned SMEM FIFOs with mbarriers; `doca_gpunetio_device.h` for GPU-initiated RDMA); `csrc/kernels/configs.cuh` (DeepEP v1 kernels: `#define NUM_MAX_NVL_PEERS 8`) | |
| [MLM-TD] | Megatron-LM `megatron/core/transformer/moe/token_dispatcher.py` (main, fetched 2026-10-13) | `MoETokenDispatcher` six abstract phases; `MoEAlltoAllTokenDispatcher` docstring (7-step workflow, DtoH sync points); `_DispatchManager`; `_HybridEPManager` (setup_metadata/dispatch/combine, `num_permuted_tokens` None → CPU sync); `_DeepepManager` (default 20 SMs); `MoEFlexTokenDispatcher` (backend selection, `_initialize_metadata` TP×EP expansion) |
| [MLM-A2A] | Megatron-LM `megatron/core/transformer/moe/fused_a2a.py` | `HybridEPDispatch` / `HybridEPCombine` autograd functions (backward of dispatch = combine and vice versa; `non_blocking = num_permuted_tokens is not None`; "we do not support fp8 dispatch") |
| [MLM-CFG] | Megatron-LM `megatron/core/transformer/transformer_config.py` | `moe_flex_dispatcher_backend` docstring ("[Experimental]", default "deepep", "Currently only hybridep backend supports the MNNVL case"); `moe_flex_dispatcher_num_sms`; `moe_permute_fusion_into_hybridep`; `moe_hybridep_num_sms_preprocessing = 108`; `moe_expert_rank_capacity_factor` (static budget; tokens above it dropped with hybridep) |
| [DSV3] | DeepSeek-V3 technical report, arXiv 2412.19437 | background only (node-limited routing); no number used |

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 256 experts, top-8; 58 MoE layers; 116 dispatch/combine per forward pass | DeepSeek-V3 MoE configuration and op count | [MC] §9.2 p.73; §4.2.1 p.30 |
| EP64 = 8 nodes on H100 (NVL8) | topology | [MC] §9.2.2 p.75 ("EP64 spans 8 nodes") |
| EP64 inside one NVLink domain on GB200 NVL72; 1.8 TB/s bidirectional NVLink | topology | [MC] §9.2.2 p.75 |
| ~50 GB/s | inter-node bandwidth used in the paper's example (InfiniBand NDR); label in fig_topo | [MC] §4.2.1 p.30 |
| 40–60% (cross-node, H100 EP64) and ~20% (inside NVL72, GB200 EP64) | EP all-to-all share of training time before optimization | [MC] §4.2 p.29 |
| top-k copies per token | permutation before all-to-all replicates each token top-k times | [MC] §4.2.2 p.30-31 |
| 20 SMs per GPU, ~20% GEMM-efficiency overhead | DeepEP SM carve-out in the DeepSeek-V3 benchmark | [MC] §4.2.3 p.34; MCore default 20 in [MLM-TD] `_DeepepManager.__init__` |
| ≤ 8 NVLink peers | DeepEP v1 kernels: `NUM_MAX_NVL_PEERS 8` | [HEP-CODE] `csrc/kernels/configs.cuh` |
| six phases | dispatch_preprocess, token_dispatch, dispatch_postprocess, combine_preprocess, token_combine, combine_postprocess | [MLM-TD] `MoETokenDispatcher`; [MC] §2.1.3 p.9-10 |
| four backends: deepep, deepepv2, hybridep, ncclep | Flex backends on main | [MLM-CFG] `moe_flex_dispatcher_backend`; [MLM-TD] `MoEFlexTokenDispatcher.__init__` |
| v0.12 (2025/05) Flex+DeepEP; v0.15 (2025/11) HybridEP backend | release history | [RD] lines 33, 16 |
| 24 / 8 default SMs | HybridEP default dispatch/combine SM count with one NVLink domain / several | [HEP-CODE] `config.cuh` `Configurer` |
| 108 | default SMs for HybridEP preprocessing (metadata scan) kernel | [MLM-CFG] `moe_hybridep_num_sms_preprocessing` |
| 64 tokens | default chunk size of dispatch/combine/preprocessing APIs | [HEP-RD] "Chunk Size Tuning" |
| ~3.75 GB (MNNVL, 64 ranks × 1 domain) and ~4 GB (RDMA, 8 ranks × 8 domains) | worst-case registered buffer per rank, EP64, 256 experts, BF16, hidden 7168, 4096 tokens/rank | [HEP-IMPL] §4.2 table |
| ≤ 1 GB | per RDMA buffer cap | [HEP-IMPL] §4.2 |
| Table 7 latencies (µs), hidden 7168, seq 4096, 256 experts: GB200 dispatch HybridEP/all-to-all EP8 391/735, EP16 578/743, EP32 612/769, EP64 675/930; H100 dispatch EP8 661/1265, EP16 1485/5774, EP32 3064/8059, EP64 4626/9164; GB200 combine EP8 353/741, EP16 527/765, EP32 646/758, EP64 744/827; H100 combine EP8 624/1277, EP16 1688/5628, EP32 3088/7815, EP64 4398/8727 | communication latency only | [MC] Table 7 p.32 |
| 1.3–3.9× (dispatch), 1.1–3.3× (combine) | all-to-all latency ÷ HybridEP latency over the 8 Table 7 settings per op | **derived** from [MC] Table 7: dispatch ratios GB200 1.88/1.29/1.26/1.38, H100 1.91/3.89/2.63/1.98; combine ratios GB200 2.10/1.45/1.17/1.11, H100 2.05/3.33/2.53/1.98 |
| B200 EP8 bandwidth (GB/s, Torch API, 4096 tokens, hidden 7168, top-8, 8 local experts): DeepEP BF16 dispatch at 16/24/28/32/36/40/44/48 SMs = 348/494/560/619/635/629/640/646; DeepEP combine = 302/420/477/524/549/570/577/586; DeepEP FP8 dispatch = 246/349/397/443/482/519/544/554; HybridEP at 16 SMs: BF16 dispatch 535.94, combine 530.86, FP8 dispatch 409.71 | SM-scaling comparison | [HEP-RD] "B200 Platform" table |
| 1.54× / 1.76× / 1.67× | HybridEP ÷ DeepEP at 16 SMs (BF16 dispatch / combine / FP8 dispatch) | **derived**: 535.94/348, 530.86/302, 409.71/246 |
| 28–36 SMs | smallest DeepEP SM count whose bandwidth ≥ HybridEP's 16-SM bandwidth: BF16 dispatch 28 (560 ≥ 535.94), FP8 dispatch 32 (443 ≥ 409.71), combine 36 (549 ≥ 530.86) | **derived** from [HEP-RD] B200 table |
| 8 SMs fill NVLink (DGX H100, 8 GPUs); ~4 SMs approach NIC max (4× DGX H100, CX7 400 Gb/s); 16 SMs fill NVLink across 36 GPUs (GB200 NVL36) | blog test results (hidden 8192, BF16, 4096 tokens/rank, top-8) | [BLOG] test section |
| End-to-end TFLOPS/GPU on Grace Blackwell: DeepSeek-V3 MXFP8 DeepEP 829 → HybridEP 943 (1.14×); DeepSeek-V3 FSDP MXFP8 A2A 597 → 645 (1.08×); Qwen3-235B BF16 A2A 665 → 698 (1.05×); Qwen3-235B MXFP8 A2A 728 → 800 (1.10×) | Megatron Core integration | [BLOG] Table 1 |
| 1,048 TFLOPS/GPU with HybridEP (GB200, 256 GPUs) vs 368 with DeepEP (1,024 H100) | final DeepSeek-V3 configurations (speaker notes only) | [MC] Table 17 p.74 |

## Mechanism facts (no numbers) used on slides

- Flex dispatcher uses one communication group over all TP and EP ranks; bool routing maps are expanded from
  [tokens, experts] to [tokens, TP×EP ranks, local experts] so the backend is agnostic of TP/EP sizes
  ([MLM-TD] `MoEFlexTokenDispatcher` docstring and `_initialize_metadata`). Each backend manager implements
  `setup_metadata`, `dispatch`, `combine` (+ permuted/restored views) ([MLM-TD] `_DispatchManager`).
- HybridEP dispatch = permute + dispatch all-to-all + permute for experts in one step; combine is the reverse;
  the backward of dispatch is a combine with the same handle and vice versa ([MLM-TD] `_HybridEPManager`
  docstring; [MLM-A2A]).
- Kernels: persistent (one block per SM for its lifetime), warp-specialized, independent blocks each owning
  data chunks ([HEP-IMPL] §8; [BLOG]). Dispatch: G2S warp group loads tokens/probs into an SMEM cyclic FIFO,
  S2G warp group writes them to remote ranks' buffers over NVLink, RDMA warp group (multi-node) sends to the
  same-local-index GPU on other nodes using IBGDA and also receives ([HEP-IMPL] §8.1; [BLOG]); routing map
  decides which ranks receive a token, so unwanted token data is not transmitted ([BLOG]). TMA: [HEP-RD]
  "Intra-node communication: NVLink-optimized data transfer using TMA instructions"; [BLOG] "TMA commands for
  data communication on NVLink scale-up networks"; IBGDA: [BLOG] "IBGDA network technology for RDMA networks";
  [MC] p.31.
- Combine: reduction fused into the communication kernel; inter-node case reduces across nodes and then
  within each node ([MC] p.31); blog/impl describe intra-node pre-reduction before the RDMA send, then the
  global reduction; accumulation on CUDA cores; results written by TMA ([BLOG]; [HEP-IMPL] §8.2).
- Fused permute: `dispatch_with_permute` writes expert-major, padded rows directly into PyTorch memory;
  optional permute blocks inside the same grid (`moe_permute_fusion_into_hybridep`) ([HEP-IMPL] §7.2; [MLM-CFG]).
- NVLink domain: `HybridEPBuffer` detects ranks reachable over NVLink and sets ranks-per-domain and
  `num_of_nodes` = number of NVLink domains ([HEP-CODE]); buffers allocated with fabric handles when
  supported (MNNVL), otherwise IPC ([HEP-CODE] allocator; [HEP-IMPL] §4.1). One worst-case registered buffer
  per rank, allocated and exchanged once at init ([HEP-IMPL] §4.2, §4.4).
- Sync-free: without a token bound, `_HybridEPManager` reads `tokens_per_expert.sum()` (CPU sync); with
  `num_permuted_tokens` (static budget from `moe_expert_rank_capacity_factor`), `non_blocking=True` skips the
  sync, excess tokens are dropped and flagged ([MLM-TD]; [MLM-A2A]; [HEP-IMPL] §7.2); [MC] p.44 "Sync-Free
  Dispatch with HybridEP"; [HEP-RD] "Full CUDA Graph compatibility ... Zero CPU-GPU synchronization".
- Limits: dispatch and combine share one buffer by default, so two HybridEP ops on the same buffer cannot
  overlap ([HEP-IMPL] §4.3); buffer reallocation is slow ([HEP-IMPL] §4.5); kernels JIT-compiled with nvcc and
  cached ([HEP-IMPL] §6); combine supports BF16 only ([HEP-IMPL] §8); HybridEP needs FP32 probs ([MLM-TD]);
  MCore passes `fp8_dispatch = False` ([MLM-A2A]); multi-node build needs rdma-core/DOCA or the experimental
  NIXL path ([HEP-RD]); backend option marked "[Experimental]", default `deepep` ([MLM-CFG]).
- Platform choice in [MC]: HybridEP on GB200/GB300, DeepEP on H100 (§8.1 p.68; Table 17 p.74).

## Discrepancies

1. **Notion "available by default in MCore v0.15".** The README lists HybridEP as *added* to the Flex
   dispatcher in v0.15 (2025/11), but it is not the default: `moe_token_dispatcher_type` defaults to
   `allgather`, `moe_flex_dispatcher_backend` defaults to `deepep` and is marked "[Experimental]"; HybridEP
   must be selected with `--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep`
   ([RD] line 16, 312; [MLM-CFG]). Slides say "added in v0.15, opt-in".
2. **Notion "abstracts the DeepEP and HybridEP backends".** On main the Flex dispatcher has four backends:
   deepep, deepepv2 (DeepEP v2 ElasticBuffer, deck 24), hybridep and ncclep (TransformerEngine NCCL EP)
   ([MLM-CFG]). Slides show all four.
3. **Blog text vs blog table.** The blog prose says "5.5% improvement in the BF16 scene" for Qwen3-235B, but
   its Table 1 gives 665 → 698 TFLOPS (1.05×, i.e. ~5.0%). Slides use the table values (and the table's
   "BF18" typo is read as BF16).
