# Story — deck 13: Fused AllReduce + residual + RMSNorm (HPC-Ops)

- **Where it lives** (prosperity): Tencent Hunyuan production inference with HPC-Ops; tensor parallelism over one
  8×H20 NVLink/NVSwitch node, BF16, hidden 4096/5120/7168; two TP AllReduces per layer, each followed by residual
  add + RMSNorm.
- **Pain number** (betrayal): vLLM lists an indicative **5–20%** end-to-end speedup from fusing just this
  AllReduce + RMSNorm chain (small token counts) — the layer boundary is a real share of decode latency.
- **Failed suitors**:
  1. NCCL AllReduce + a separate fused_add_rmsnorm: two kernels, the reduced tensor round-trips HBM, and all 8
     GPUs normalize all N tokens.
  2. FlashInfer / TRT-LLM fused kernels: vLLM enables them only for small num_tokens; in HPC-Ops' benchmark the
     FlashInfer mnnvl path is capped at 8,192 tokens on H20, trtllm clamps its Lamport buffer.
- **Villain**: message size spans 112 KiB (8 decode tokens) to 448 MiB (32K prefill tokens) at hidden 7168: small
  messages are latency/synchronization-bound, large ones bandwidth-bound, and the norm sits behind a kernel
  boundary in both.
- **Turn / insight**: RMSNorm needs only one complete reduced token; a two-shot all-reduce hands each GPU complete
  reduced tokens after its first shot → normalize inside the collective, and choose the transport by token count.
- **Mechanisms**:
  ❶ Two-shot schedule with token ownership (answers: where can the norm run before the full all-reduce completes).
  ❷ HT: NVSwitch multicast `ld_reduce` → add + RMSNorm in registers → multicast store, one kernel (answers: bytes
     at large N; the reduced tensor never lands in HBM, norm work per GPU is 1/8).
  ❸ LL: Lamport P2P scatter (data is its own ready flag, −0 sentinel, 3 rotating buffers) + multicast broadcast;
     RMSNorm kernel launched with PDL, prefetching residual/γ and polling the broadcast buffer (answers: sync and
     launch latency at small N).
- **Release**: up to **1.76×** lower latency than the best of NCCL + norm and FlashInfer fused AllReduce, BF16 TP
  on one 8×H20 node (README). Analytic support: HT touches 2.25 vs ≥ 6 hidden vectors of HBM per token per GPU.
- **Limits**: single NVSwitch node with multicast (SM90); BF16 and three hidden sizes only; LL needs power-of-two
  TP 2–64; inputs in symmetric multicast buffers; LL workspace 6× activation; caller chooses HT vs LL (no
  automatic switch); per-token-count results not published (max only); not in the SGLang upstream integration.

## Slide plan (15 logical slides, 18 pages, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Every TP block ends in AllReduce, add, RMSNorm | prosperity | fig_tp_layer |
| 3 | This boundary is worth 5–20% end-to-end | pain | big number |
| 4 | Unfused, the sum round-trips through HBM | status quo | fig_unfused |
| 5 | Prior fused kernels stop at decode sizes | suitors + villain | two-col + callout |
| 6 | Insight: two-shot hands each GPU whole tokens | turn | text + callout (build) |
| 7 | One op, two transports, one two-shot schedule | overview | fig_overview |
| 8 | ❶ Two shots give every token one owner | mechanism 1 | fig_twoshot |
| 9 | ❷ HT: the switch sums, the owner normalizes | mechanism 2 | fig_ht_1 → fig_ht_2 |
| 10 | ❸ LL: the data is its own ready flag | mechanism 3 | fig_lamport_1 → fig_lamport_2 |
| 11 | ❸ PDL launches the norm kernel early | mechanism 3 | fig_pdl |
| 12 | HT cuts HBM traffic per token ≥ 2.7× | why two modes (analytic) | chart hbm_traffic |
| 13 | Up to 1.76× over NCCL and FlashInfer | release | big number (tone ours) |
| 14 | The price: one node, three shapes, two modes | limits | two-col |
| 15 | Summary | | QR |
