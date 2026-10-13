# Notion row (new entry, created 2026-10-13)

Database: Industrial Pretraining Architecture & Infra Tech Landscape (2026)
Status: created in Notion on 2026-10-13 as https://app.notion.com/p/3f895647fb2a81aa8f07fc1ab10670bb (body without the Related rows section, plus an Explainer slides section). Every claim is traced in `notes/sources.md`.

## Properties

- **Technique:** MagiAttention (distributed attention for ultra-long, heterogeneous-mask training)
- **Category:** Frameworks & Platform Stacks
- **Core Mechanism:** Container entry: Sand AI's Apache-2.0 context-parallel attention library, built to train MAGI-1 (24B video model, up to 4M-token contexts) on packed, block-causal masks. The mask is written as AttnSlices (Q range, K range, mask type), which drive four parts: the Flex-Flash-Attention kernel (FA3-based; FFA_FA4 on Blackwell), an area-balancing dispatch solver, zero-redundant GroupCast/GroupReduce (native DeepEP-based kernels), and multi-stage overlap.
- **Impact / Measured Results:** Vendor CP benchmark, varlen causal mask, 64 H100s: about 238 fwd / 265 bwd TFLOPs/s per GPU, 8-9x ring P2P and 1.06-1.24x the best of six baselines (incl. Ulysses, USP, LoongTrain); 64 B200s: 1.4-1.9x. FFA reaches about 90% of FA3 on standard masks and 3.6x FlashMask on varlen block-causal (H100, 32K). Values read from published charts.
- **Reference:** https://github.com/SandAI-org/MagiAttention
- **Source:** MagiAttention, MAGI-1

(Core Mechanism 453 characters, Impact 338 characters; neither contains any of the characters the database
text should avoid.)

## Page body

## Papers & slides (PDF)
- [MAGI-1: Autoregressive Video Generation at Scale (PDF)](https://arxiv.org/pdf/2505.13211) — Sand AI report; §4.1.2 and App. B introduce MagiAttention and its first benchmarks
- [FlashAttention-3 (PDF)](https://arxiv.org/pdf/2407.08608) — the Hopper kernel that Flex-Flash-Attention extends
- [Ring Attention with blockwise transformers (PDF)](https://arxiv.org/pdf/2310.01889) — the ring P2P context-parallel baseline
- [DeepSpeed Ulysses (PDF)](https://arxiv.org/pdf/2309.14509) — the head-sharded all-to-all baseline

## Public resources
- [MagiAttention repo](https://github.com/SandAI-org/MagiAttention) — Apache-2.0 source, docs, Megatron/FSDP/Transformers examples
- [MagiAttention technical blog](https://sandai-org.github.io/MagiAttention/docs/main/blog/magi_attn.html) — AttnSlice, dispatch solver, GroupCast/GroupReduce, multi-stage overlap
- [Long-context attention benchmark](https://sandai-org.github.io/MagiAttention/docs/main/blog/cp_benchmark.html) — kernel and CP benchmarks on H100 and B200 against six CP baselines
- [Native group collective blog](https://sandai-org.github.io/MagiAttention/docs/main/blog/native_grpcoll.html) — DeepEP-based GroupCast/GroupReduce with RDMA de-duplication
- [FFA_FA4 Blackwell backend blog](https://sandai-org.github.io/MagiAttention/docs/main/blog/blackwell_ffa_fa4.html) — flexible masks on a forked FlashAttention-4
- [Learnable attention sink blog](https://sandai-org.github.io/MagiAttention/docs/main/blog/attn_sink.html) — sink tokens in FFA and distributed MagiAttention
- [Release notes](https://github.com/SandAI-org/MagiAttention/releases) — v1.0.0 (Apr 2025) to v1.1.6 (Sep 2026)
- [Megatron-LM-MagiAttention](https://github.com/SandAI-org/Megatron-LM-MagiAttention) — Megatron-LM fork with a Llama-3 1B convergence example
- [MAGI-1 repo](https://github.com/SandAI-org/MAGI-1) — the video model MagiAttention was built to train

## Related rows
- Deck 68 Context Parallelism / Dynamic CP · deck 69 KDA Context Parallelism · deck 51 Dynamic batching / sequence packing ·
  deck 24 DeepEP v2 and deck 15 0-SM PP/CP/Engram primitives · decks 07 and 09 attention kernels
