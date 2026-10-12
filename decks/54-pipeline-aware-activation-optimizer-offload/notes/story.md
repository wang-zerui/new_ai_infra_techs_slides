# Story — deck 54: Pipeline-aware activation / optimizer offload

Family: decks 52 / 53 / 54 attack the MoE activation memory wall (recompute, reorder, offload). Shared visuals:
the DeepSeek-V3 per-GPU memory stack (here both offload targets, activations and main weights + optimizer
state, are highlighted), the saved-tensor chip language of deck 52's overview (fig_policy reuses it with an
"offloaded" chip), and the "memory toolbox" ledger with the optimizer-state-offload bar highlighted.
Cross-references: 52, 53, 46, 49 (BF16 moments shrink what must be moved), 70 (state offload excludes
Megatron-FSDP), 11 (offload is compatible with 1F1B EP overlap).

- **Where it lives** (prosperity): Megatron-Core (fine-grained activation offloading, "pipeline-aware" in the
  2026/01 dev release; optimizer-state offload on the dev branch); Kimi K2 (1T, H800, 2 TB RAM per node) and
  Kimi K3 (2.8T) in their own stacks.
- **Pain number** (betrayal): 15% — Qwen3-235B is held at TP2/EP16 (800 TF/s) because TP1/EP64 (920 TF/s) does
  not fit without offload (Table 5).
- **Failed suitors**: full recompute keeps L × layer input on GPU and costs ~33% compute; Transformer Engine's
  layer-level offload is all-or-nothing per layer (legacy cpu_offloading rejects PP > 1 and recompute).
- **Villain**: in 1F1B the first pipeline stage holds the activations of PP micro-batches; EP and PP shrink
  parameter memory, not activation memory.
- **Turn / insight**: copy engines run beside the SMs — a module whose compute outlasts its activation copy
  offloads for free — and the pipeline schedule fixes when each activation is needed again.
- **Mechanisms**:
  ❶ Module-level offload + layer-staggered reload: D2H right after each offloaded module's forward; in backward,
     the same module of the previous layer reloads while the current one computes; the last layer stays.
  ❷ Pipeline-aware handlers: one ChunkOffloadHandler per (micro-batch, VPP chunk), queued to follow interleaved
     1F1B; cross-chunk pre-reload; later PP ranks offload less; K2: offload previous forward, prefetch next
     backward.
  ❸ Optimizer state offload: Adam moments + FP32 master weights to host after the step, back before the next;
     step stays on GPU (vs the CPU optimizer, which runs the step on the host).
- **Release**: DeepSeek-V3 −10.7% memory for −1.6% throughput; Qwen3-235B +15.0% throughput via TP1/EP64
  (Table 5); 10–18% memory for 1.6–2% throughput overall; optimizer-state offload 15–20 GB for 0.1–0.2 s/iter;
  long-context worst case ~2%; Kimi K2/K3 offload all remaining activations at trillion scale with EP still
  fully overlapped (K2).
- **Limits**: copy bandwidth (C2C vs PCIe), PCIe contention with EP, pinned host memory, last layer kept,
  TE-only (NVTE_CPU_OFFLOAD_V1=1 on TE ≥ 2.10), CUDA-graph restrictions, state offload excludes Megatron-FSDP.

## Slide plan (15 logical slides, 16 PDF pages, 8 minutes)

| # | Headline | Beat | Visual |
|---|---|---|---|
| 1 | Title | | |
| 2 | Activations and optimizer state overflow HBM | prosperity | chart mem_stack (family) |
| 3 | Memory, not speed, picks the parallel layout | pain | `%% big: 15%` |
| 4 | Activations pile up on the first pipeline stage | villain | fig_inflight (1F1B) |
| 5 | Full recompute and layer offload both fall short | failed suitors | two-col |
| 6 | Insight: copies are free while a module computes | turn | fig_hide |
| 7 | Offload heavy inputs, recompute cheap outputs | overview | fig_policy |
| 8 | ❶ Copy out after forward, back one layer early | mechanism (build ×2) | fig_streams_1/2 |
| 9 | ❷ The 1F1B schedule decides what moves when | mechanism | fig_1f1b |
| 10 | ❸ Optimizer state leaves HBM between steps | mechanism | fig_optimizer |
| 11 | Offload costs 1.6% speed, or buys back 15% | release | chart offload_table5 |
| 12 | Optimizer offload frees 15–20 GB per GPU | result (family) | chart toolbox |
| 13 | Kimi K2 and K3 offload the rest at trillion scale | adoption | two-col |
| 14 | The cost: bandwidth, host memory, constraints | limits | two-col |
| 15 | Summary | | QR arxiv.org/abs/2603.07685 |
