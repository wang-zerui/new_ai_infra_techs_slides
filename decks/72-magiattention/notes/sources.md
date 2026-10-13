# Sources — deck 72: MagiAttention

All downloads are in the session scratch folder `research/72/` (README, blog sources, release notes, repo
assets, MAGI-1 PDF). Fetched 2026-10-13 with curl / `gh api`; no WebSearch/WebFetch.

## Sources read

| Tag | Source | Where |
|---|---|---|
| [README] | SandAI-org/MagiAttention `README.md` (main, 2026-10-13). https://github.com/SandAI-org/MagiAttention | Latest News (v1.0.0-v1.1.1), About, Key Designs, Benchmark, Citation, Acknowledgement |
| [BLOG] | "MagiAttention" technical blog, 21 Apr 2025 (updated with v1.1.0). https://sandai-org.github.io/MagiAttention/docs/main/blog/magi_attn.html (HTML) and its source `docs/source/blog/magi_attn.md` | Overview (Fig. 1), Introduction, Related Work (Fig. 2), Methodology: AttnSlice (Fig. 3-9), Dispatch Solver (Eq. 1, Fig. 10), Static/Dynamic Attn Solver, Ring P2P Redundancy (Fig. 11), Group Collective (Fig. 12), AlltoAll-v / Native Implementation, Multi-Stage Overlap (Fig. 13, 14), Experiments (Fig. 15-18), FFA_FA4, Future Work |
| [CPB] | "Long-Context Attention Benchmark" blog. https://sandai-org.github.io/MagiAttention/docs/main/blog/cp_benchmark.html and `docs/source/blog/cp_benchmark.md` | Benchmark Settings (common configs, metrics Eq. 2-3, data sampling), Kernel/Distributed Baselines, Kernel Level Fig. 20-43, Distributed Level Fig. 44-59 and the "experimental features" warning |
| [GRP] | "Support Native Group Collective" blog, 24 Jan 2026, `docs/source/blog/native_grpcoll.md` | Introduction (3 benefits), Enabling (`MAGI_ATTENTION_NATIVE_GRPCOLL=1`, IBGDA for cp_size > 8), Similarity to DeepEP, kernel designs, RDMA Transfer De-duplication, Other Features (cp size limits) |
| [SINK] | "Support Learnable Attention Sink" blog, 17 Nov 2025, `docs/source/blog/attn_sink.md` | Introduction, Overview (v1.0.5), MagiAttn Impl (sink applied once, at the host stage) |
| [FA4] | "Support Blackwell with FFA_FA4 Backend" blog, 7 Feb 2026, `docs/source/blog/blackwell_ffa_fa4.md` | Introduction (temporary backend, with MiniMax and NVIDIA), HSTU Function Representation, FA4 modifications, magi_to_hstu |
| [OVL] | "How to Ensure Kernels Actually Overlap" blog, 15 Feb 2026, `docs/source/blog/kernel_overlap.md` | SM starvation, sm_margin, high-priority stream, KernelBarrier |
| [QS] | User guide `quickstart.md` | `magi_attn_flex_key` / `dispatch` / `calc_attn` / `undispatch` / `roll`; chunk_size; note that dispatch may shuffle tokens (care for RoPE) |
| [REL] | GitHub releases v1.0.0 (2025-04-21) ... v1.1.1 (2026-05-21) ... v1.1.6 (2026-09-18), via `gh api repos/SandAI-org/MagiAttention/releases` | per-release feature lists |
| [MEG] | `examples/megatron/README.md` | Megatron-LM v0.11.0 fork, Llama-3 1B convergence vs TE Ring Attention, CP 1/2/4/8 |
| [MAGI1] | Sand AI, "MAGI-1: Autoregressive Video Generation at Scale", arXiv 2505.13211v1 (61 pp.) | Abstract and §1 (24B, up to 4M tokens), Fig. 1 (24-frame chunks, block-causal mask), §4.1 intro, §4.1.1 (online PnP, 99% capacity), §4.1.2 (MagiAttention), App. B.2 (benchmarks, Table 11) |
| [GH] | `gh api repos/SandAI-org/MagiAttention` (2026-10-13) | created 2025-04-20, 952 stars, 71 forks, Apache-2.0 |
| [EXP] | `exps/dist_attn/benchmark_conf.py`, `exps/dist_attn/baselines/*.py` | baselines are the MagiAttention team's own harness (ring_attn.py uses TE + zigzag dispatch; hybrid_dcp.py uses Megatron's BalancedCPScheduler) |

No standalone MagiAttention paper exists on arXiv (arXiv API search for "MagiAttention" returns only MAGI-1;
the blog's Future Work still lists "Prepare a standalone technical report/paper"). The citation is a
`@misc` GitHub entry [README].

## Chart readings (method)

The benchmark blogs publish bar charts without printed values. Values below were pixel-read from the
original PNGs in the repo (`assets/magi_attn/exp/{kernel,distributed}/{h100,b200}/<mask>/{fwd,bwd}/flops_report.png`,
the same images as [CPB] Fig. 20-59) with `notes/measure_bars.py`: y = 0 at the x-axis frame, scale from
the gridline spacing measured in the gaps between bar groups, bar top = median over four columns of each
bar. Full readings: `notes/chart_readings.txt`. Precision about ±2 TFLOPs/s at these scales; slides
round and mark them "≈" / "read from chart". Spot-checked by eye against the rendered charts.

## Numbers on slides

| Number on slide | Meaning | Source |
|---|---|---|
| 24B parameters, up to 4M-token contexts | largest MAGI-1 variant | [MAGI1] abstract; §1 p.2 |
| 24-frame chunks; each chunk attends to all earlier chunks; block-causal mask | MAGI-1 generation scheme | [MAGI1] Fig. 1 caption |
| online Patch-and-Pack (PnP), 99% capacity | packing of variable-length videos | [MAGI1] §4.1.1 |
| ≈513 vs ≈29 TFLOPs/s/GPU (513 / 29 read); ≈18× (derived 513/29 = 17.7) | Ring P2P forward, 64 H100, full mask vs varlen causal mask | [CPB] Fig. 44 and Fig. 50 (pixel-read) |
| 25% / 33% redundant ring KV traffic | 4 ranks, 8 chunks, zigzag; causal mask / varlen block-causal mask with a last global block | [BLOG] Fig. 11 caption; re-derived below |
| Ulysses: heads must be divisible by CP size | head-sharded CP limit | [BLOG] Related Work |
| AttnSlice = (QRange, KRange, MaskType); 4 types FULL, CAUSAL, INV-CAUSAL, BI-CAUSAL | mask representation | [BLOG] AttnSlice Representation; Basic Mask Types (Fig. 6) |
| FFA built on FA3, Hopper TMA, slice-level parallelism with atomic reduction | kernel | [BLOG] AttnSlice-level Parallelism (Fig. 5) |
| FFA_FA4: HSTU function (per-row KV intervals) on forked FA4, temporary; `MAGI_ATTENTION_FA4_BACKEND=1` | Blackwell backend | [FA4] Introduction, HSTU Function Representation; [REL] v1.1.0 |
| chunk_size; seqlen divisible by cp_size × chunk_size; equal chunk count per rank; greedy min-heap O(n log n) | dispatch solver | [BLOG] Dispatch Solver, Eq. (1), Fig. 10 |
| 2, 6, 10, 2 vs 6, 5, 5, 4 (area units), max 10 vs 6, even share 5 | running example: varlen causal mask, 8 chunks, 4 ranks, contiguous vs min-heap | analytic, derived below |
| KV5 on rank2 needed only by Q6, Q7 → rank0, rank1; dKV5 reduced back to rank2 | GroupCast / GroupReduce example | [BLOG] Group Collective Primitives (Fig. 12 text) |
| a2a-v prototype: extra Range-Gather / Range-Scatter-Reduce copies; m separate send buffers for m peers | prototype limits | [BLOG] AlltoAll-v Implementation; [GRP] Limitations |
| native: DeepEP dispatch/combine-based; no D2D copies; one RDMA send to the peer with the same local rank, then NVLink fan-out (k× fewer RDMA transfers); lse reduce op in-kernel | native group collectives | [GRP] Introduction, Similarity to DeepEP, RDMA Transfer De-duplication, Other Features |
| IBGDA required for native kernels when cp_size > 8; internode: intranode size 8 only, internode sizes 2-32 | prerequisites | [GRP] Installation, Other Features |
| forward 4-stage example; backward exposes only the last dKV reduce; overlap_degree, manual tuning in {1,2,3,4} recommended | overlap | [BLOG] Multi-Stage Overlap (Fig. 13), Dynamic Overlap Stage Search |
| KernelBarrier, sm_margin | overlap enforcement | [OVL] |
| FFA vs FA3 vs FlashMask fwd at 32K, H100: full 634 / 701 / 360; causal 622 / 680 / 352; varlen causal 548 / 584 / 241; varlen block-causal 575 / n.a. / 159 TFLOPs/s | kernel benchmark | [CPB] Fig. 20, 22, 26, 30 (pixel-read); FA2/FA3/cuDNN/Flex marked unsupported on varlen block-causal |
| ≈90% (634/701 = 0.90), 0.91, 0.94 of FA3; 3.6× FlashMask (575/159 = 3.62) | derived ratios | derived from the row above |
| backward 32K: full FFA 576 vs FA3 601 (0.96); causal 560 vs 585 (0.96); varlen block-causal FFA 363 vs FlashMask 205 (1.77×) | kernel bwd (notes only) | [CPB] Fig. 21, 23, 31 (pixel-read) |
| B200 full mask fwd 32K: FFA_FA4 1401 vs FA4 1299, cuDNN 1329 | Blackwell kernel (notes only) | [CPB] Fig. 32 (pixel-read) |
| kernel setup: b = 1, GQA 64:8:8, head dim 128, BF16, sequence 1K-32K | kernel benchmark config | [CPB] Common Configurations; chart x-axes |
| distributed setup: cp_size 8-64, 8K tokens per GPU (H100), 16K (B200), so 64K-512K total on H100; varlen samples ≤ 1/4 of total; baselines on FA3 (H100) or cuDNN (B200, Magi on FFA_FA4); HybridCP omitted on B200 | distributed benchmark config | [CPB] Distributed Level, Distributed Baselines, Data Distribution |
| H100 varlen causal fwd, Ring P2P / Ulysses / USP / MagiAttention-native: 8 GPUs 89 / 278 / 261 / 459; 16: 69 / 200 / 251 / 427; 32: 48 / 186 / 255 / 316; 64: 29 / 163 / 192 / 238 | scaling chart | [CPB] Fig. 50 (pixel-read; also README H100 figure) |
| H100 varlen causal bwd, same order: 8: 97 / 332 / 325 / 361; 16: 86 / 295 / 309 / 371; 32: 57 / 277 / 285 / 334; 64: 29 / 249 / 196 / 265 | scaling chart | [CPB] Fig. 51 (pixel-read) |
| other H100 baselines at 64 GPUs: fwd Ring AllGather 181, LoongTrain 193, HybridCP 162; bwd Ring AllGather 55, LoongTrain 172, HybridCP 164 | notes only | [CPB] Fig. 50, 51 (pixel-read) |
| best baseline at 64 H100: fwd 193 (USP / LoongTrain), bwd 249 (Ulysses); MagiAttention 238 / 265 → 1.24× / 1.06×; vs Ring P2P 238/29 = 8.2×, 265/29 = 9.1× | derived ratios | derived |
| 64 GPUs, best baseline / a2av / native: H100 fwd 193 / 208 / 238; H100 bwd 249 / 243 / 265; B200 fwd 295 (Ring AllGather) / 398 / 563; B200 bwd 404 (Ulysses) / 449 / 554 | native-gain chart | [CPB] Fig. 50, 51, 58, 59 (pixel-read) |
| native over a2av: +14% (238/208), +9% (265/243), +41% (563/398), +23% (554/449); B200 vs best: 1.91× (563/295), 1.37× (554/404) | derived ratios | derived |
| at 8 GPUs a2av ≥ native on B200: fwd 913 vs 870, bwd 694 vs 648 | notes only | [CPB] Fig. 58, 59 (pixel-read) |
| per-GPU throughput 8 → 64 H100 fwd (varlen causal): 459 → 238 (52% kept); bwd 361 → 265 (73%) | limits | [CPB] Fig. 50, 51 (pixel-read); ratios derived |
| full mask 64 H100 bwd: MagiAttention-native 549 vs USP 581, Ulysses 578 (≈5% behind); fwd 629 vs Ulysses 620 | limits | [CPB] Fig. 44, 45 (pixel-read) |
| "experimental features" enabled for MagiAttention in the CP benchmark | limits | [CPB] Distributed Level warning |
| static attn solver: mask known at data-processing time and same for all layers; KV-only comm; dynamic solver + QO comm experimental | limits | [BLOG] Static / Dynamic Attn Solver; [REL] v1.0.4 |
| dispatch may shuffle tokens; sample-wise ops like RoPE need care | limits | [QS] comment on `dispatch` |
| v1.0.0 2025-04-21; v1.0.5 2025-11-19 (attention sink, experimental intranode native kernels); v1.1.0 2026-02-28 (FFA_FA4, full native group collectives, Muon QK-clip); v1.1.1 2026-05-21 (roll API for MTP, FFA_FA4 on sm80); v1.1.6 2026-09-18 (native CuTeDSL FFA on SM100) | maturity | [REL] |
| Megatron-LM v0.11.0 fork, Llama-3 1B, CP 1-8, loss matches TE Ring Attention | adoption | [MEG] |
| 952 stars, 71 forks, Apache-2.0 | adoption | [GH] |

### Derived numbers

**Ring redundancy, causal mask, 4 ranks, zigzag (re-derivation of [BLOG] Fig. 11a).** Rank r holds chunks
{r, 7−r} (queries and KV). Under causal masking rank j needs key chunk c iff c ≤ 7−j. Ring P2P delivers all
6 remote chunks to every rank (24 transfers). Unused: rank0 none; rank1 {7}; rank2 {6, 7}; rank3 {5, 6, 7}
→ 6 / 24 = **25%**.

**Varlen block-causal mask with a last global block ([BLOG] Fig. 11b).** Rows Q0-Q1 see K0-K1, Q2-Q3 see
K0-K3, Q4-Q5 see K4-K5, Q6-Q7 see all keys; holdings rank0 {0,2}, rank1 {1,3}, rank2 {4,6}, rank3 {5,7}.
Rank0 and rank1 need only chunks 0-3, so 4 of their 6 received chunks each are unused; rank2/rank3 use all
→ 8 / 24 = **33%**.

**Dispatch running example (analytic).** Varlen causal mask of 8 equal chunks: document A = C0-C5,
document B = C6-C7. Area of a chunk row in chunk-block units (diagonal block counts ½): A: 0.5, 1.5, 2.5,
3.5, 4.5, 5.5; B: 0.5, 1.5; total 20, even share 5 per rank. Contiguous split (rank r gets C2r, C2r+1):
**2, 6, 10, 2** (max 10). Greedy min-heap ([BLOG] Fig. 10: sort by area desc, give each chunk to the
least-loaded rank with fewer than 2 chunks, ties by rank id): rank0 {C5, C6} = 6, rank1 {C4, C0} = 5,
rank2 {C3, C7} = 5, rank3 {C2, C1} = 4 → **6, 5, 5, 4** (max 6). Slowest rank: 10 → 6.

**Pain ratio.** Ring P2P forward on 64 H100: full mask 513 / varlen causal 29 = 17.7 → "≈18×". Both bars
read from [CPB] Fig. 44 and Fig. 50; the varlen causal bar has ±1 reading error (range 17.1-18.3×).

## Notion row draft (notes/notion_row.md): provenance of the body links and property claims

- MAGI-1 PDF: [MAGI1]; FlashAttention-3 (arXiv 2407.08608), Ring Attention (2310.01889) and DeepSpeed Ulysses
  (2309.14509) are cited with these arXiv ids in the [BLOG] / [CPB] reference lists (FFA is "an extension of
  FlashAttention-3"; Ring P2P and Ulysses are CP baselines).
- Blog URLs: [BLOG], [CPB], [GRP], [FA4], [SINK] (all returned HTTP 200 on 2026-10-13); release page [REL];
  Megatron-LM-MagiAttention fork [MEG] / [REL] v1.0.2; MAGI-1 repo linked from [README] ("magi-source" in [MAGI1]).
- "six baselines": Ulysses, Ring P2P, Ring AllGather, USP, LoongTrain, Megatron HybridCP ([CPB] Distributed Baselines).
- "8-9x ring P2P": 238/29 = 8.2 (fwd), 265/29 = 9.1 (bwd), derived above.
- "Megatron/FSDP/Transformers examples": `examples/megatron`, `examples/torch_native` (FSDP), `examples/transformers` [README].

## Discrepancies

- **README "Latest News" stops at v1.1.1 (2026/05)**, but GitHub releases continue to v1.1.6 (2026-09-18).
  Slides cite the release list.
- **"Linear scalability" vs the published charts.** README and blog describe near-linear scalability; on the
  varlen causal mask the H100 per-GPU throughput of MagiAttention falls from ≈459 to ≈238 TFLOPs/s/GPU
  (fwd) between 8 and 64 GPUs. It stays above every baseline, so the slides say "leads at every scale" and
  list the decline under limits instead of claiming linear scaling.
- **"Performance comparable to FA3".** On FA3-supported masks FFA reads at 90-94% of FA3 forward and ≈96%
  backward at 32K on H100 ([CPB] charts). Slides say "≈90% of FA3".
- **MAGI-1 benchmark vs blog benchmark.** The MAGI-1 paper's module-level benchmark (App. B.2.2) compares
  only Ring-Attention and Ulysses on full / varlen-full masks on an undisclosed cluster up to cp 48 and 3M
  tokens; the slides use the newer [CPB] benchmark (H100 / B200, 6 baselines), which has disclosed hardware.
- **Kernel benchmark range.** [CPB] text says sequence lengths go "up to 64K"; the published kernel charts
  stop at 32K. Slides use 32K.
- No Notion row exists yet (new entry); see `notes/notion_row.md` for the proposed text.
