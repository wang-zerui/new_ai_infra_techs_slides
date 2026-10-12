# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** Algebraic reordering: moves the router-probability multiply out of unpermute and into the GroupedMLP activation function, avoiding an extra large buffer allocation.
**After:** Algebraic reordering: since fc2 is linear and bias-free, the router-probability multiply moves out of unpermute into each expert's activation function (fused weighted SwiGLU), so the router-weight gradient needs only the SwiGLU input already saved, not the expert output.
**Why:**
- The equivalence (Eq. 1 = Eq. 2) holds for bias-free experts; with Eq. 2 the probability gradient needs phi(z), recomputed from z, which SwiGLU saves anyway ([MC] §4.1.2 p.21; [SW]).
- Current code applies the probs inside the activation of every expert backend, not only GroupedMLP ([EXP]: TEGroupedMLP, SequentialMLP via `per_token_scale`).
- The saving is a tensor no longer retained for backward, not an avoided buffer allocation ([MC] §4.1.2).

## Impact / Measured Results
**Before:** Removes one full activation copy - a classic low-cost fix for the MoE memory wall.
**After:** Expert outputs are no longer retained for backward: about 26.3 GB per GPU on DeepSeek-V3 (about 20% of activations, derived), with no trade-off listed.
**Why:**
- What is removed is the expert output kept for the router-weight gradient (one hidden-size row per routed token copy per MoE layer), not a copy of all activations; measured at about 26.3 GB per GPU ([MC] §4.1.2 p.21); 26.3 / 131.0 GB = 20% is derived from [MC] Table 3.
- [MC] Table 6 lists no trade-off for the technique. (The code adds a small separate all-to-all for the probs, [TD]; not stated in the row.)

## Other properties
none (the discrepancy notes that the Reference, the MoE README, carries no numbers while all numbers come from arXiv 2603.07685; the README does describe the technique, so the Reference is not plainly wrong and was left unchanged for the user to decide)
