# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** MCore's Flex dispatcher abstracts the DeepEP and HybridEP backends uniformly; HybridEP uses TMA/IBGDA and fewer SMs, with native MNNVL support.
**After:** MCore's Flex dispatcher abstracts four backends (deepep, deepepv2, hybridep, ncclep on main) behind one interface; HybridEP uses TMA over NVLink and IBGDA over RDMA in persistent warp-specialized kernels with fewer SMs, and is the only Flex backend with MNNVL support.
**Why:**
- On main the Flex dispatcher has four backends, not two ([MLM-CFG] `moe_flex_dispatcher_backend`; [MLM-TD] `MoEFlexTokenDispatcher`).
- TMA for NVLink, IBGDA for RDMA, persistent warp-specialized kernels ([BLOG] design section; [HEP-IMPL] §8; [HEP-RD]).
- "Currently only hybridep backend supports the MNNVL case" ([MLM-CFG]).

## Impact / Measured Results
**Before:** Higher MoE all-to-all bandwidth utilization on GB200 NVL72 / B200 / H100 (available by default in MCore v0.15).
**After:** Higher MoE all-to-all bandwidth utilization on GB200 NVL72 / B200 / H100; added in MCore v0.15 as an opt-in Flex backend marked experimental (the Flex default backend is deepep).
**Why:**
- Not a default: v0.15 added HybridEP to the Flex dispatcher ([RD] l.16), but `moe_token_dispatcher_type` defaults to allgather and `moe_flex_dispatcher_backend` defaults to deepep, marked "[Experimental]" ([RD] l.312; [MLM-CFG]).

## Other properties
none
