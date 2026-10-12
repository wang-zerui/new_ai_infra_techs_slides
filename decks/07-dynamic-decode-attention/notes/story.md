# Story: Dynamic Decode Attention (HPC-Ops)

- **Where it lives** (prosperity): decode attention in Tencent's Hunyuan (Hy3) serving on H20 and in SGLang's
  `hpc_ops` attention backend; continuous batching plus long-context, multi-turn and agentic traffic put 128K and
  4K contexts in the same decode launch.
- **Pain number** (betrayal): with a static split-KV schedule, 1×128K + 31×4K takes **4.3×** longer (0.186 vs
  0.043 ms) than 64×4K although it reads ~2% *less* KV (derived from LMSYS Table 1).
- **Failed suitors**: (1) fixed split count per request (e.g. 4 or 16 chunks): long requests make heavy chunks and
  a few CTAs set the kernel tail; (2) fixed chunk size: the grid must reserve splits for the longest request, so
  short requests occupy slots with empty chunks. Library baselines (FlashInfer, FlashAttention) also degrade.
- **Villain**: the launch grid maps work by *request* (head × request × chunk) with one policy for the whole batch,
  decided without the live length distribution.
- **Turn / insight**: KV work is countable in 64-token tiles and all lengths are on the device before each step,
  so total tiles ÷ CTAs is an exact balanced budget: balance tiles, not requests. Evidence: on a uniform batch the
  scheduled kernel ties the static one (1.00×), so the scheduling costs nothing where it is not needed.
- **Mechanisms**:
  ❶ assign kernel: tile every request, compute the per-CTA tile budget (with a 512-token floor) ↔ unknown, skewed
  lengths;
  ❷ bin fill + task descriptors: each persistent CTA gets an equal bin; long requests span CTAs, short ones share
  ↔ heavy chunks / empty chunks;
  ❸ persistent CTAs drain bins writing partial O + LSE; a combine kernel merges each request's chunks by
  log-sum-exp; PDL chains assign → attention → combine; the task map is reused by every layer ↔ correctness and
  overhead of splitting.
- **Release**: up to **2.95×** vs static (0.186 → 0.063 ms), parity on uniform; on average **2.25×** faster than
  the best of FlashInfer/FlashAttention across six H20 cases. End to end: small alone on H200 (+2–3% throughput),
  part of Hy3 TPOT −15.1–48.8% with MoE on H20.
- **Limits**: SM90 only; head_dim 128, GQA groups of 4/8, MTP ≤ 2; workspace + assign kernel per step (benchmark
  excludes it by default); combine work grows with chunk count; skewed batch still 1.9× the uniform one.

## Slide plan (14 logical slides)

| # | Headline | Device |
|---|---|---|
| 1 | Title | |
| 2 | Decode batches mix 128K and 4K contexts | bullets |
| 3 | Same KV work, 4.3× the latency | big 4.3× (derived) |
| 4 | Static split-KV either stalls or wastes CTAs | fig_static (two panels) |
| 5 | Insight: balance KV tiles, not requests | bullets + boxed takeaway (build) |
| 6 | One task map per step drives every layer | fig_overview |
| 7 | ❶ The assign kernel sets a per-CTA tile budget | fig_bins_1 |
| 8 | ❷ Equal bins: long requests span CTAs | fig_bins_2 |
| 9 | ❸ CTAs drain bins; combine merges by LSE | fig_exec |
| 10 | Gains grow with skew, up to 2.95× | chart_static |
| 11 | 2.25× faster than the faster library on average | chart_libs |
| 12 | End to end: modest alone, large with fused MoE | bullets + callout |
| 13 | Costs: one more kernel, SM90 only | bullets |
| 14 | Summary | bullets + QR |
