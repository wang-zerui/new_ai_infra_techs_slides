# Sources — deck 12: Ascend · comm/compute fusion and PD-disaggregated scheduling

## Sources read

| Tag | Source | Where |
|---|---|---|
| [GLM5] | GLM-5 Team, "GLM-5: from Vibe Coding to Agentic Engineering", arXiv 2602.15763 (cached text) | §2.1 p.5 + Table 2 (MTP with shared parameters, accept lengths); §3.6.2 p.14-15 (PD disaggregation, MTP and FP8 for RL rollouts); §4.1 p.17 (DP-aware routing / prefix reuse); §5 p.21-22 (Ascend adaptation: FlashComm, DP attention + EP, RadixCache/Prefix Cache, async scheduling, MTP, result claim); App. A Table 10 p.36 (architecture) |
| [VA-GLM5] | vLLM Ascend, `docs/source/tutorials/models/GLM5.md` (repo `vllm-project/vllm-ascend`, commit 8194ece, fetched 2026-10-13 via `gh api`) | §5 deployment flags; §5.3 Prefill-Decode disaggregation (Mooncake connector, roles, MTP per role); §9.1 Table 1 (reference configurations) and Table 2 (optimizations requiring explicit enablement) |
| [VA-SP] | vLLM Ascend, `docs/source/user_guide/feature_guide/sequence_parallelism.md` (same commit) | SP MoE dataflow, `enable_flashcomm1`, `fuse_gemm_comms` (matmul + reduce-scatter fusion) |
| [FLUX] | Chang et al., "FLUX: Fast Software-based Communication Overlap on GPUs via Kernel Fusion", arXiv 2406.06858 | §1 p.2 + Fig. 1 (TP communication share); §2.2 p.4 (why chunked overlap underperforms); §5.1 p.13 (operation-level overlap efficiency); §6 p.15 (model-level communication portions) |

Downloaded copies live in `$SCRATCH/research/12/` (`flux.pdf`, `flux.txt`, the two vLLM Ascend docs).

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| **40–75%** of step time | tensor-parallel communication share on a weak-fabric cluster (A100 PCIe, 128 GPUs, training and prefill) | [FLUX] §6 p.15 ("much higher communication portions, such as 40% to 75% on A100 PCIe training and prefill"); the workload mix is defined in §1 Fig. 1 |
| **8–11%** of step time | the same measurement on A100 NVLink training | [FLUX] §6 p.15 |
| **−125% to 36%** overlap efficiency | chunked ("medium-grained") overlap with TransformerEngine 1.4 + UserBuffer, GEMM+ReduceScatter / AllGather+GEMM, GPT-3 175B shapes, A100 PCIe; negative means slower than not overlapping | [FLUX] §5.1 p.13 |
| **41–57%** overlap efficiency | Flux's tile-fused overlap on the same A100 PCIe setup | [FLUX] §5.1 p.13 |
| **2.76** vs **2.55** accept length | GLM-5 vs DeepSeek-V3.2 MTP accept length at 4 speculative steps on the authors' private prompt set | [GLM5] Table 2 p.5 |
| **3** shared MTP layers (training) ; **3** speculative tokens on decode nodes, **1** on prefill nodes | MTP configuration | [GLM5] §2.1 p.5; [VA-GLM5] §5.3 "MTP in PD scenarios" |
| 744B / 40B active, 256 experts, 200K context | model under discussion | [GLM5] §2.1 p.4, Table 10 p.36 |
| One Atlas 800 A3 node = 8 × 128 GB; TP16 per node; reference serving length **202,752** tokens ("198K") | deployment scale of the reference configs | [VA-GLM5] §3, §5, §9.1 Table 1 |
| PD reference split: prefill **PP2 TP16**, decode **DP8 TP4** per node (A3, 198K high throughput) | upstream reference configuration | [VA-GLM5] §9.1 Table 1 and §5.3.2 |
| TP **2, 4 or 8** for the fused matmul + reduce-scatter pass | constraint of `fuse_gemm_comms` | [VA-SP] "Matmul reduce-scatter fusion" |
| Fused MC2 limits: `dispatch_ffn_combine` w8a8 and EP ≤ 32; `mega_moe` EP ≤ 64; both incompatible with MTP and dynamic EPLB | constraint table | [VA-GLM5] §9.1 Table 2 and §5.3 |
| **50%** lower long-sequence deployment cost; one domestic node "comparable to dual-GPU international clusters" | the report's claim for the whole Ascend stack | [GLM5] §5 p.22 |

Derived numbers: none. No percentage on any slide is presented as an Ascend measurement; the 40–75% /
8–11% / overlap-efficiency numbers are labelled on the slide as Flux's GPU-cluster measurements.

## Mechanism facts (no numbers) used on slides

- GLM-5 §5 lists, for the Ascend inference engines: asynchronous scheduling (overlapping the device-to-host
  sampling copy with preparation of the next decode step), context management (RadixCache prefix sharing and
  a Prefix Cache that extends KV storage into system RAM), a hybrid parallel strategy (attention data
  parallelism + MoE expert parallelism) "alongside FlashComm, which splits AllReduce operations to hide
  communication latency behind computation", and MTP. ([GLM5] §5 p.22)
- FlashComm_v1 in vLLM Ascend: "Splits AllReduce into Reduce-Scatter and All-Gather, improving prefill
  throughput and reducing communication latency"; switched on with
  `--additional-config '{"enable_flashcomm1": true}'`; unavailable when `layer_sharding` includes `o_proj`.
  ([VA-GLM5] §9.1 Table 2)
- Why the split pays: the attention `o_proj` ends in a TP all-reduce, so its output is replicated on every TP
  rank; feeding replicated tokens into the experts duplicates compute and communication under expert
  parallelism. Sequence-parallel MoE keeps the tokens sharded between the reduce-scatter and the all-gather,
  so the norm, the router and the all-to-all see 1/TP of the tokens. Dataflow given upstream:
  shard → all-gather → attention → reduce-scatter → RMSNorm → router → all-to-all → MoE. ([VA-SP])
- Fusing the collective into the matmul: `fuse_gemm_comms` fuses a sequence-parallel, bias-free row-parallel
  projection with its token-dimension reduce-scatter (BF16/FP16, TP 2/4/8); `enable_fused_mc2` replaces
  ALLTOALL+MC2 with `dispatch_ffn_combine` / `dispatch_gmm_combine_decode` for the MoE path.
  ([VA-SP]; [VA-GLM5] §9.1 Table 2)
- Why tiles rather than chunks: chunked overlap splits one GEMM into as many kernels as there are ranks;
  execution timing is not controllable, the dependent element-wise work between chunks blocks concurrency,
  and the small GEMMs underutilize the device. Flux instead decomposes computation and communication into
  tiles and fuses them into one kernel, where each dependent compute/communication tile pair maps to a thread
  block. ([FLUX] §1 p.2, §2.2 p.4). Flux is a GPU work from a different vendor; it is cited here only as the
  published explanation of the technique class, never as an Ascend measurement.
- PD disaggregation upstream: Mooncake `MooncakeConnectorV1` moves the KV cache, prefill nodes are
  `kv_producer` and decode nodes `kv_consumer`, `use_ascend_direct` enables RDMA-like transfer; prefill nodes
  run DSA context parallelism and `enforce_eager`, decode nodes run MLAPO, `FULL_DECODE_ONLY` graph capture
  and the recompute scheduler (requests bounce back to prefill when decode KV cache runs out).
  ([VA-GLM5] §5.3, §9.1 Table 2)
- In GLM-5's own RL rollouts, the reason given for PD disaggregation is interference: under DP attention a
  heavy prefill preempts ongoing decodes and sharply worsens tail latency, so prefills and decodes get
  dedicated resources. ([GLM5] §3.6.2 p.15)
- MTP: GLM-5 shares the parameters of 3 MTP layers during training, keeping the draft-model memory cost of a
  single-layer MTP while raising the acceptance rate. ([GLM5] §2.1 p.5)

## Discrepancies

1. **"pipelines it with tiled matrix computation" (Notion Core Mechanism).** The GLM-5 report says only that
   FlashComm "splits AllReduce operations to hide communication latency behind computation" ([GLM5] §5 p.22).
   Tile-level fusion of the matmul with the collective is documented in the Ascend engine
   (`fuse_gemm_comms`, [VA-SP]) and in the Flux paper ([FLUX]), not in the GLM-5 report. The deck attributes
   it that way: the split is the report's, the tiling is the engine's and the literature's.
2. **"One domestic node matches a dual-GPU node" (Notion Impact).** The report writes "comparable to
   dual-GPU international clusters" and gives no baseline, benchmark, metric or configuration ([GLM5] §5
   p.22). The slide quotes the report and says the evidence is not published.
3. **Attribution of the 50%.** The Notion row attaches the 50% cost reduction to this technique; in the
   report it is the result of the whole Ascend adaptation stack (quantization + fused kernels + scheduling),
   with no per-pillar breakdown. The deck states this.
4. **No Ascend percentage exists.** Neither the report nor the upstream docs publish how much of a GLM-5 step
   the collectives take on Ascend. The pain number on slide 3 is therefore Flux's GPU-cluster measurement,
   labelled as such on the slide, and is used to show how the share grows when the fabric is weaker — not as
   a GLM-5 or Ascend result.
