# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** InfiniBand virtual lanes (VL) isolate EP traffic from everything else and adaptive routing is enabled; community work (AntGroup) adds Normal-SMFree (decoupling comm-kernel execution from NIC transfer), LL-SBO (overlapping Down GEMM with combine sends) and LL-Layered (rail-optimized forwarding and merging).
**After:** DeepEP sets an InfiniBand service level per buffer so the fabric's SL-to-VL mapping can put EP traffic on its own virtual lane (V2: EP vs other; V1 used three classes), and V2 recommends adaptive routing under all loads (V1: heavy load only); community work (AntGroup, experimental branch) adds Normal-SMFree (open PR: decoupling comm-kernel execution from NIC transfer), LL-SBO (overlapping Down GEMM with combine sends) and LL-Layered (same-rail RDMA then NVLink forwarding).
**Why:**
- DeepEP only sets the SL (`sl_idx`, `EP_OVERRIDE_RDMA_SL`, `EP_DEFAULT_RDMA_SL`); the SL-to-VL mapping is fabric configuration ([RD] "Network configurations").
- V1 recommended three classes (normal, low-latency, other); V2 reduced it to EP vs other ([RD-V1a], [RD-V2]).
- AR advice changed: unsafe for normal kernels (Feb 2025), heavy load only (Mar 2025 to Feb 2026), all loads (V2) ([RD-V1a], [AR-FIX], [RD-V1b], [RD-V2]).
- Normal-SMFree is PR #347, still open against `antgroup-opt` ([PR347]); LL-Layered sends to the same-rail rank, then NVLink-forwards; "merging" is not described ([PR500]).

## Impact / Measured Results
**Before:** Stops EP all-to-all from colliding with DP/checkpoint traffic and returns RDMA-path SMs to compute.
**After:** Separates EP all-to-all from other traffic (e.g. DP, storage) on its own virtual lane; returning RDMA-path SMs to compute is only an open AntGroup PR, as mainline DeepEP V2.5 still needs SMs for RDMA EP.
**Why:**
- Mainline: "EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported" ([RD] Notes); SM-free normal kernels exist only in unmerged PR #347 ([PR347]).
- "Stops colliding" softened: VLs share each link through an arbiter, and no source quantifies the collision cost (discrepancy item 5; [FF] §VI-A1).

## Other properties
none

Left for the user (page body, not touched): the body cites the Azure DeepEP blog for "adaptive routing, traffic isolation", but that post covers NUMA/HCA affinity and PCI atomics, not VLs or AR ([AZ]); the discussion sub-page's LL-SBO "+8-10%" is the LMSYS figure for BS 20-56, while PR #390's bs-32 runs show +6.7% / +7.5% ([LMSYS-ANT], [PR390]).
