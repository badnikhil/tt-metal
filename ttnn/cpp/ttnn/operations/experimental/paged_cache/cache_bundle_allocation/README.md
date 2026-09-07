# Cache bundle allocation metadata

`ttnn.experimental.update_cache_bundle_allocation` manages one pool, independently on
all participating devices. It does not access KV payloads or use CCL.

```python
ttnn.experimental.update_cache_bundle_allocation(
    page_table, allocated_pages, free_list, free_count,
    slot_id=slot, actual_start=start, actual_end=end, page_size=32,
)
```

`actual_end` is exclusive; any token count is accepted and rounded up to pages.
A nonzero `actual_start` grows capacity without changing existing mappings.
A zero `actual_start` starts a new lifetime: release the old allocation, then
allocate for the new end. `(actual_start=0, actual_end=0)` releases only.
Repeating a nonzero-start call is idempotent; repeating a zero-start call resets.

## Host initialization

Create these rank-2, row-major, interleaved DRAM tensors once. The logical shapes
below are local shapes: replicate every tensor in full across the device mesh.

| Tensor | Dtype | Shape | Initial values |
|---|---|---|---|
| `page_table` | UINT32 | `[slots, max_pages]` | Zero |
| `allocated_pages` | UINT32 | `[1, slots]` | Zero |
| `free_list` | UINT32 | `[SP, bundles_per_sp]` | Each row: `N-1, ..., 0` |
| `free_count` | UINT32 | `[1, SP]` | `N` |

```python
host_metadata = [
    torch.zeros((slots, max_pages), dtype=torch.int32),
    torch.zeros((1, slots), dtype=torch.int32),
    torch.arange(N - 1, -1, -1, dtype=torch.int32).repeat(SP, 1),
    torch.full((1, SP), N, dtype=torch.int32),
]
page_table, allocated_pages, free_list, free_count = [
    ttnn.from_torch(
        value,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    for value in host_metadata
]
```

Logical page `p` belongs to SP `p % SP`. Its bundle ID addresses the local KV
shard on each TP device at that SP. IDs are unique within each SP's pool; the
same numerical ID on different SPs denotes different bundles. IDs range from
`0` to `N-1`; no ID is reserved as a sentinel. Validity comes from counts.
Entries outside valid prefixes must not be consumed.

Size each SP's pool for the combined live allocations across slots. To fill
all slots to their maximum context, provision at least
`slots * ceil(max_pages / SP)` bundles per SP. For 500 slots, 1,048,576 tokens
per slot, 32 tokens per page, and SP=8, this is 2,048,000 bundles per SP.
UINT32 IDs allow this pool to exceed the 65,536-bundle limit of UINT16 IDs.
This describes metadata capacity; KV payload storage must be provisioned separately.

## Caller responsibilities

The op updates all four metadata tensors in place and returns the input
`page_table`. It allocates no output buffer and returns no status.
Host-side shape, dtype, placement, and scalar errors raise normally.

The inference server must reserve sufficient capacity before calling. With equal
capacity on every SP and prefix allocations, SP 0 has the highest usage:
`sum(ceil(pages_in_slot / SP))` across all live slots. Evaluate this total after
the proposed update, including reclaiming the old slot on reset. It must not
exceed `bundles_per_sp`. Serialize admission and updates so concurrent requests
cannot reserve the same capacity. The kernel does not detect or report OOM.

The caller must maintain valid counts and nonoverlapping ownership, and pass a
nonzero start only within the slot's allocated capacity. These are preconditions:
the kernel trusts tensor contents, and host validation does not read them back.
Finish old slot accesses before a reset, and order consumers after the update.
All replicas must start identically and execute the same requests in the same order.

## Scope

The implementation uses one data-movement core and stages one page-table row,
the two counter rows, and at most 4 KiB of one free-list row. Free-list windows
are aligned and flushed before reuse, including when reset changes stack direction.
Total scratch is capped at 512 KiB. Free-list rows may
exceed L1 capacity: the 500-full-slot example above has 7.8 MiB per free-list row
but needs approximately 134 KiB of scratch. Metadata row byte sizes must fit
32-bit NoC offsets, and tensor allocation remains subject to available DRAM.
One op call manages one slot and one pool. Multiple independent pools require
separate calls; this does not provide a transaction across pools.

Request scalars are runtime arguments on normal program-cache hits. A trace
captures those values; replaying a trace does not read new host scalar values.
This op introduces the allocator contract only. Existing attention/cache-write
ops are not converted to this UINT32/SP-interleaved table format by this change.
