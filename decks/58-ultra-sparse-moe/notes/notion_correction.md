# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** DeepSeekMoE-style fine-grained expert pool plus one shared expert; total and activated parameters fully decoupled, with sparsity chosen by scaling laws.
**After:** DeepSeekMoE-style pool of hundreds of narrow routed experts (6 or 8 active) plus one always-on shared expert; total parameters grow apart from per-token FLOPs, bounded by memory and EP all-to-all cost; only Kimi K2 publishes a sparsity scaling law, and stops at sparsity 48 for infra cost.
**Why:**
- "Fully decoupled" overstates: K2 stops at sparsity 48 because of "increased infrastructure complexity" ([K2] p.7), K2 needs about 6 TB for weights + grads and the smallest EP is 16 ([K2] §2.4.2), and GLM-5 cuts its layer count to 80 to limit EP communication ([GLM5] §2.1 p.4).
- "Sparsity chosen by scaling laws": only K2 publishes a sparsity law; V4 and GLM-5 give configs without a sweep ([K2] p.7 + Fig. 5; [V4] §4.2.1; [GLM5] Table 10).
- Pool shape from configs: 384 or 256 routed, top-6 (V4) or top-8 (K2, GLM-5), 1 shared ([V4] §4.2.1; [K2] Table 2; [GLM5] Table 10).

## Impact / Measured Results
**Before:** V4-Pro 1.6T/49B active, V4-Flash 284B/13B; K2 1.04T/32B; GLM-5 744B/40B.
**After:** unchanged
**Why:** all sizes match the sources ([V4] Abstract, §4.2.1; [GLM5] Table 10). K2's 32B is the abstract's rounding of Table 2's 32.6B ([K2] abstract, Table 2), so not an error.

## Other properties
none
