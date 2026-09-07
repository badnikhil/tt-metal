// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "paged_cache_nanobind.hpp"

#include <cstdint>
#include <optional>
#include <set>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/operations/experimental/paged_cache/paged_cache.hpp"

namespace ttnn::operations::experimental::paged_cache::detail {

void bind_experimental_paged_cache_operations(nb::module_& mod) {
    ttnn::bind_function<"update_cache_bundle_allocation", "ttnn.experimental.">(
        mod,
        R"doc(Update cache bundle allocator metadata in place, independently on each device.

Metadata is rank-2, row-major, interleaved DRAM: page_table UINT32 [slots, max_pages],
allocated_pages UINT32 [1, slots], free_list UINT32 [SP, bundles_per_sp], and
free_count UINT32 [1, SP]. Logical page p belongs to SP p % SP; the bundle ID
is shared by the TP devices at that SP. Only the first allocated_pages[slot]
table entries and the first free_count[sp] stack entries are valid.

actual_end is an exclusive token position, rounded up to page_size. When
actual_start is zero, release the old slot lifetime before allocating the new
one. (0, 0) releases only. Otherwise grow without moving existing mappings;
actual_start must lie within allocated capacity. Nonzero-start calls are safe
to repeat; zero-start calls always reset. This operation does not write KV data.

Returns the input page_table, updated in place; no output buffer is allocated.
The caller must reserve sufficient capacity before calling. The kernel assumes
valid counts, unique bundle ownership, and a nonzero start within allocated
capacity; it does not detect or report OOM. Host-side shape, dtype, placement,
and scalar validation errors raise. Tensor contents are not copied to the host.

Initialize metadata on the host and replicate all four tensors across the mesh.
Every replica must execute identical requests in identical order; no CCL is used.
Serialize calls sharing a pool, finish prior slot accesses before reset, and
order consumers after this update before accessing new pages. Scalars are captured
by traces; changing a request requires a fresh invocation outside that trace.

The kernel stages one table row, the counter rows, and at most 4 KiB of a
free-list row, with a 512 KiB total scratch limit. Pools may
exceed 65,536 bundles per SP; size them for all simultaneously live slots.
IDs are in [0, bundles_per_sp); no ID is reserved as a sentinel. Metadata row
byte sizes must fit 32-bit NoC offsets; tensor storage is limited by available DRAM.
)doc",
        &ttnn::experimental::update_cache_bundle_allocation,
        nb::arg("page_table").noconvert(),
        nb::arg("allocated_pages").noconvert(),
        nb::arg("free_list").noconvert(),
        nb::arg("free_count").noconvert(),
        nb::kw_only(),
        nb::arg("slot_id"),
        nb::arg("actual_start"),
        nb::arg("actual_end"),
        nb::arg("page_size") = 32);

    const auto* paged_update_cache_doc =
        R"doc(
         Paged update cache operation. This operation expects the following inputs: cache_tensor of shape [B, 1, kv_len, head_dim] and input_tensor of shape [1, B, 1[32], head_dim] where input_tensor is height sharded on B cores. update_idxs will specify for each batch element which token to update in the cache.

         ``head_dim`` is read from ``input_tensor.padded_shape[-1]``. ``block_size``
         defaults to ``cache_tensor.padded_shape[2]``; pass the kwarg to override it for
         callers that reinterpret one physical buffer with different
         ``(block_size, head_dim)`` tile layouts (e.g. vLLM's hybrid kv-cache-groups
         path). ``num_kv_heads`` defaults to ``cache_tensor.padded_shape[1]``; pass the
         kwarg when the input view has a different kv-head count from the cache (e.g.
         Gemma4 sliding kv=8 / full kv=2 sharing one HMA buffer) — the decode-time
         input is height-sharded with the kv-heads dim padded to TILE_HEIGHT so the
         logical count can't be inferred from the tensor.
         ``num_kv_heads * block_size * head_dim`` must be preserved across views.
         Input tensors must be FLOAT32 or BFLOAT16 even when the destination cache is
         BFLOAT8_B/BFLOAT4_B. This op updates one token row inside an existing cache
         tile and owns the final repack into the cache dtype; do not pre-cast decode
         K/V updates to the low-precision cache dtype. In contrast,
         ``paged_fill_cache`` performs a raw tile copy and requires matching input
         and cache dtypes.
         ``cache_position_modulo`` (optional, paged mode only) makes the kernel
         compute ``update_idx %= cache_position_modulo`` before resolving the
         page_table entry — i.e. treats the cache as a circular buffer of that
         many tokens. Required when the cache is sized smaller than the model's
         max sequence length (vLLM's ``SlidingWindowSpec`` allocation pattern);
         without it, positions past the bounded capacity collapse onto block 0
         and silently corrupt the cache. Must be a multiple of the effective
         ``block_size`` and ≤ ``page_table.shape[1] * block_size``.
        )doc";

    ttnn::bind_function<"paged_update_cache", "ttnn.experimental.">(
        mod,
        paged_update_cache_doc,
        &ttnn::experimental::paged_update_cache,
        nb::arg("cache_tensor").noconvert(),
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("update_idxs").noconvert() = nb::cast(std::vector<uint32_t>()),
        nb::arg("update_idxs_tensor").noconvert() = nb::none(),
        nb::arg("share_cache").noconvert() = nb::none(),
        nb::arg("page_table").noconvert() = nb::none(),
        nb::arg("batch_offset") = 0,
        nb::arg("compute_kernel_config").noconvert() = nb::none(),
        nb::arg("mesh_coords").noconvert() = nb::none(),
        nb::arg("block_size") = nb::none(),
        nb::arg("num_kv_heads") = nb::none(),
        nb::arg("cache_position_modulo") = nb::none());

    const auto* paged_fused_update_cache_doc =
        R"doc(
            Updates the cache tensors `cache_tensor1` and `cache_tensor2` in parallel with values derived from the corresponding input tensors. This function supports fine-grained updates using specified index lists or tensors.

            Positional Arguments:
                cache_tensor1 (ttnn.Tensor): The first cache tensor to update.
                input_tensor1 (ttnn.Tensor): The input tensor corresponding to `cache_tensor1`.
                cache_tensor2 (ttnn.Tensor): The second cache tensor to update.
                input_tensor2 (ttnn.Tensor): The input tensor corresponding to `cache_tensor2`.

            Keyword Args:
                update_idxs (List[int]): A list of indices specifying the cache update positions. Defaults to an empty list.
                update_idxs_tensor (ttnn.Tensor, optional): A tensor specifying update indices. Defaults to None.
                share_cache (bool, optional): Whether the cache tensors share memory regions. Defaults to None.
                page_table (ttnn.Tensor, optional): The page table for managing memory regions during updates. Defaults to None.
                batch_offset (int): Offset for batching updates. Defaults to 0.
                compute_kernel_config (DeviceComputeKernelConfig, Optional): Optional configuration for the device compute kernel. Defaults to None.
                mesh_coords (Set[MeshCoordinate], optional): Set of mesh coordinates to execute on.

            Input tensors must be FLOAT32 or BFLOAT16 even when the destination
            caches are BFLOAT8_B/BFLOAT4_B; this op repacks into each cache dtype.
            In contrast, ``paged_fill_cache`` performs a raw tile copy and requires
            matching input and cache dtypes.

            Returns:
                ttnn.Tensor, ttnn.Tensor: Tensors representing the updated cache states.
        )doc";

    ttnn::bind_function<"paged_fused_update_cache", "ttnn.experimental.">(
        mod,
        paged_fused_update_cache_doc,
        &ttnn::experimental::paged_fused_update_cache,
        nb::arg("cache_tensor1").noconvert(),
        nb::arg("input_tensor1").noconvert(),
        nb::arg("cache_tensor2").noconvert(),
        nb::arg("input_tensor2").noconvert(),
        nb::kw_only(),
        nb::arg("update_idxs").noconvert() = nb::cast(std::vector<uint32_t>()),
        nb::arg("update_idxs_tensor").noconvert() = nb::none(),
        nb::arg("share_cache").noconvert() = nb::none(),
        nb::arg("page_table").noconvert() = nb::none(),
        nb::arg("batch_offset") = 0,
        nb::arg("compute_kernel_config").noconvert() = nb::none(),
        nb::arg("mesh_coords").noconvert() = nb::none());

    const auto* paged_fill_cache_doc =
        R"doc(
        Paged fill cache operation. This operation expects the following inputs: cache_tensor, input_tensor, and page_table.
        It uses either batch_idx_tensor (if provided, kwarg batch_idx_tensor) or batch_idx (kwarg batch_idx) as a fallback to determine the batch index for updating the cache.
        cache_tensor shape: [max_num_blocks, 1, block_size, head_dim]
        input_tensor shape: [input_batch, num_heads, input_seq_len, head_dim]
        page_table shape: [batch_size, max_num_blocks_per_seq]
        batch_idx_tensor (optional) shape: [input_batch], dtype int32 or uint32, ROW_MAJOR layout, INTERLEAVED DRAM — one batch_idx per input batch row.
        batch_idx (scalar, defaults to 0) is used if batch_idx_tensor is not provided; in that case input_batch must be 1.
        mesh_coords (optional) is a set of MeshCoordinate objects that specify the mesh coordinates to execute on.

        ``head_dim`` is read from ``input_tensor.padded_shape[-1]``. ``block_size``
        defaults to ``cache_tensor.padded_shape[2]``; pass the kwarg to override it (see
        ``paged_update_cache`` for details). Per-block byte count must be preserved.
        ``paged_fill_cache`` does not perform dtype conversion; it copies prefill K/V
        tiles into the cache. ``input_tensor.dtype`` must match ``cache_tensor.dtype``.
        Cast prefill K/V to the cache dtype before calling this op. In contrast,
        ``paged_update_cache`` and ``paged_fused_update_cache`` accept FLOAT32 or
        BFLOAT16 decode inputs and repack them into the cache dtype.
        ``cache_position_modulo`` (optional) treats the cache as a circular buffer of
        that many tokens: each tile write computes ``seq_tile_id %=
        cache_position_modulo / TILE_HEIGHT`` before the page_table lookup. Lets
        prefill writes longer than the bounded sliding-window capacity land correctly
        (only the last ``cache_position_modulo`` tokens survive). Must be a multiple
        of the effective ``block_size`` and ≤ ``page_table.shape[1] * block_size``.
        ``valid_seq_len_tensor`` (optional, bounded mode only) is a 1-element int
        device tensor giving the block-aligned real fill length (in tokens). The
        writer restricts the surviving ring window to end there instead of the padded
        input end, so a captured prefill trace (which can't slice the padded input on
        the host per request) still avoids wrapping the prompt's padding tail over the
        real recent window. Refresh its contents per request outside the trace.
        )doc";

    ttnn::bind_function<"paged_fill_cache", "ttnn.experimental.">(
        mod,
        paged_fill_cache_doc,
        &ttnn::experimental::paged_fill_cache,
        nb::arg("cache_tensor").noconvert(),
        nb::arg("input_tensor").noconvert(),
        nb::arg("page_table").noconvert(),
        nb::kw_only(),
        nb::arg("batch_idx_tensor").noconvert() = nb::none(),
        nb::arg("batch_idx") = 0,
        nb::arg("compute_kernel_config").noconvert() = nb::none(),
        nb::arg("mesh_coords").noconvert() = nb::none(),
        nb::arg("block_size") = nb::none(),
        nb::arg("cache_position_modulo") = nb::none(),
        nb::arg("valid_seq_len_tensor").noconvert() = nb::none());
}

}  // namespace ttnn::operations::experimental::paged_cache::detail
