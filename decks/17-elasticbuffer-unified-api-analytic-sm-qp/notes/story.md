# Story: ElasticBuffer unified API + analytic SM/QP budgeting (deck 17)

- **Where it lives** (prosperity): the buffer object every MoE layer calls in DeepEP v2 (`ElasticBuffer` in V2.0,
  `EPBuffer` since V2.5): dispatch before the expert GEMMs, combine after, and both again in backward; used for
  DeepSeek-V3-style training, prefill and decode, and exposed by Megatron-LM as the `deepepv2` dispatcher backend.
- **Pain number** (betrayal): **24 SMs** per GPU spent on V1's communication kernels for V3-like training
  (V2 release notes), chosen by a global constant plus a hand-tuned chunk table that covers only 10 EP sizes.
- **Failed suitors**:
  1. Fixed SM count + per-EP-size lookup table (V1 normal kernels): re-tune per cluster ("run all the tests and use
     the best auto-tuned configuration"), any other EP size asserts out.
  2. A second API for decoding (V1 low-latency kernels) with QPs pinned to the number of local experts.
- **Villain**: treating SM and QP counts as tuning constants, although they are determined by routing statistics,
  topology and link bandwidth.
- **Turn / insight**: communication SMs only have to feed the slowest link. One SM writes ≈ 45-50 GB/s to HBM, the same
  order as a 400 Gb/s NIC (50 GB/s), so the budget is a small number that can be computed. Check: for EP 8×4 the
  V2.0 model yields 6 SMs, exactly the SM count of the release-table row that reaches 61 GB/s.
- **Mechanisms**:
  ❶ one `EPBuffer` API for training / prefill / decode (expanded layout aligned to grouped GEMM; CPU-sync vs
     GPU-side counts; cached handle for backward; deferred epilogue) — answers "two APIs, two buffer modes".
  ❷ analytic SM count (routing statistics → bytes per token → bottleneck link → SMs, +30 %) — answers "tuning table".
  ❸ QP budget derived from the SM count and mode (direct ≤ 9, hybrid one per channel; CTA-exclusive mapping) —
     answers "QPs pinned to local experts".
- **Release**: **24 → 4-6 SMs** in V3-like training at equivalent or better speed; 6 SMs → 61 GB/s per GPU at EP 8×4;
  analytic budget never exceeds 8 SMs from 32 to 256 GPUs; 24 SMs keep 89-91 % of NVLink peak.
- **Limits**: model assumes a balanced gate (not V3's group-limited gate), ignores expand / multi-reduction (TODO),
  uses fixed per-SM constants (TODO per architecture), depends on `nvidia-smi` / `ibstat` detection; fixed
  max-capacity buffers are larger than V1's; overlap-off mode forces ≥ 64 SMs; users can still override `num_sms`.

## Storyboard (16 logical slides, 8 minutes)

| # | Headline | Device |
|---|---|---|
| 1 | Title | |
| 2 | Every MoE layer calls dispatch and combine twice | fig_layer |
| 3 | V1 spent 24 SMs per GPU on communication | big 24 (pain) |
| 4 | V1 looked its chunk sizes up in a table | fig_v1table (red assert row) |
| 5 | Insight: SMs only need to feed the slowest link | fig_insight |
| 6 | One buffer, one computed budget | fig_overview (❶❷❸) |
| 7 | ❶ One dispatch layout serves training and decode | fig_layout_1 → fig_layout_2 (build) |
| 8 | ❷ The SM count follows the bottleneck link | fig_budget |
| 9 | ❷ Worked example: EP 8×4 needs 6 SMs | fig_worked_1→2→3 (build) |
| 10 | ❸ QPs follow SMs and channels | fig_qp |
| 11 | V3-style training drops from 24 to 4–6 SMs | big 24 → 4–6 (release) |
| 12 | 6 SMs sustain 61 GB/s at EP 8×4 | chart rdma_bw |
| 13 | Larger EP needs fewer SMs, never 24 | chart sm_model (analytic) |
| 14 | 24 SMs keep 89–91% of NVLink peak | chart nvlink_bw |
| 15 | The model trades generality for zero tuning | limits |
| 16 | Summary | QR |
