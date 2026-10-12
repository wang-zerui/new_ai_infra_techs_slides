# Notion row correction (2026-10-13)

## Core Mechanism
**Before:** One contiguous virtual address range backed by a mix of GPU and CPU physical memory, transparently supporting Engram and imbalanced EP; EP replay compresses intermediate buffers.
**After:** One contiguous virtual address range: GPU memory first, then pinned host (or GPU) storage shards registered for RDMA only; shipped for Engram tables (opt-in CPU storage in V2.5 EngramBuffer, experimental), while host-backed EP buffers and EP replay remain V2 roadmap items.
**Why:**
- Host-backed storage shipped for Engram only; `EPBuffer` passes zero storage bytes, so "imbalanced EP" is not supported ([BUF]; [R2] l.35 lists it as on-going; [EB2] "currently GPU-only").
- Not "transparent": the caller sizes the storage and sets `use_cpu_rdma_storage` ([ENG], [CTX] l.75-99).
- EP replay is a V2 on-going item with no code in V2.5 ([R2] l.36).
- Mechanism detail: VA layout from [SYM] l.137-179; storage shards registered as `NCCL_WIN_GIN_ONLY` windows ([CTX] l.207-222).

## Impact / Measured Results
**Before:** Mitigates V2's larger buffer footprint so worst-case load no longer dictates the memory budget.
**After:** Not yet for EP: V2.5 EP buffers are GPU-only and still sized for worst-case load; host-backed storage has shipped only for Engram tables, letting them exceed HBM (experimental).
**Why:**
- The V2.5 EP size formula still reserves room for N x T tokens per rank and EP passes no storage bytes ([EPSZ], [BUF]).
- The only shipped user is EngramBuffer ([R25] l.17, l.384-386; [PR763]).

## Other properties
none
