// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t sp_size = get_compile_time_arg_val(0);
constexpr uint32_t page_size = get_compile_time_arg_val(1);
constexpr uint32_t table_bytes = get_compile_time_arg_val(2);
constexpr uint32_t allocated_bytes = get_compile_time_arg_val(3);
constexpr uint32_t free_window_bytes = get_compile_time_arg_val(4);
constexpr uint32_t counts_bytes = get_compile_time_arg_val(5);
constexpr uint32_t free_row_bytes = get_compile_time_arg_val(6);
constexpr uint32_t free_window_entries = free_window_bytes / sizeof(uint32_t);

constexpr auto table_args = TensorAccessorArgs<7>();
constexpr auto allocated_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
constexpr auto free_args = TensorAccessorArgs<allocated_args.next_compile_time_args_offset()>();
constexpr auto count_args = TensorAccessorArgs<free_args.next_compile_time_args_offset()>();

struct MetadataBuffers {
    decltype(TensorAccessor(table_args, 0)) table_acc;
    decltype(TensorAccessor(allocated_args, 0)) allocated_acc;
    decltype(TensorAccessor(free_args, 0)) free_acc;
    decltype(TensorAccessor(count_args, 0)) count_acc;
    CoreLocalMem<volatile uint32_t> table;
    CoreLocalMem<volatile uint32_t> allocated;
    CoreLocalMem<volatile uint32_t> counts;
    uint32_t free_l1_address;
};

// Bind the four DRAM tensors and their L1 scratch buffers in host argument order.
MetadataBuffers make_metadata_buffers(uint32_t& rt_args_idx) {
    CircularBuffer cb_table(0);
    CircularBuffer cb_allocated(1);
    CircularBuffer cb_free(2);
    CircularBuffer cb_counts(3);
    return {
        TensorAccessor(table_args, get_arg_val<uint32_t>(rt_args_idx++)),
        TensorAccessor(allocated_args, get_arg_val<uint32_t>(rt_args_idx++)),
        TensorAccessor(free_args, get_arg_val<uint32_t>(rt_args_idx++)),
        TensorAccessor(count_args, get_arg_val<uint32_t>(rt_args_idx++)),
        CoreLocalMem<volatile uint32_t>(cb_table.get_write_ptr()),
        CoreLocalMem<volatile uint32_t>(cb_allocated.get_write_ptr()),
        CoreLocalMem<volatile uint32_t>(cb_counts.get_write_ptr()),
        cb_free.get_write_ptr()};
}

// Count this SP's pages in a slot's interleaved logical-page prefix.
uint32_t pages_on_sp(uint32_t pages, uint32_t sp) { return pages / sp_size + (sp < pages % sp_size); }

// Load allocation counters first so unchanged decode requests can return immediately.
uint32_t read_allocated_pages(const Noc& noc, const MetadataBuffers& buffers, uint32_t slot) {
    noc.async_read(buffers.allocated_acc, buffers.allocated, allocated_bytes, {.page_id = 0}, {});
    noc.async_read_barrier();
    return buffers.allocated[slot];
}

// Load the selected page-table row and all SP free counts before updating IDs.
void read_slot_metadata(const Noc& noc, const MetadataBuffers& buffers, uint32_t slot) {
    noc.async_read(buffers.table_acc, buffers.table, table_bytes, {.page_id = slot}, {});
    noc.async_read(buffers.count_acc, buffers.counts, counts_bytes, {.page_id = 0}, {});
    noc.async_read_barrier();
}

// Publish the page table and counter rows, and wait for all DRAM writes to finish.
void write_slot_metadata(const Noc& noc, const MetadataBuffers& buffers, uint32_t slot, uint32_t next_pages) {
    buffers.allocated[slot] = next_pages;
    noc.async_write(buffers.table, buffers.table_acc, table_bytes, {}, {.page_id = slot});
    noc.async_write(buffers.allocated, buffers.allocated_acc, allocated_bytes, {}, {.page_id = 0});
    noc.async_write(buffers.counts, buffers.count_acc, counts_bytes, {}, {.page_id = 0});
    noc.async_write_barrier();
}

// Preserve untouched entries and padding. Flush before reusing the buffer,
// including when a reset switches from returning IDs to allocating them.
template <typename Accessor>
class FreeListWindow {
public:
    // Attach the reusable L1 window to one SP's free-list row.
    FreeListWindow(const Noc& noc, const Accessor& accessor, uint32_t sp, uint32_t l1_address) :
        noc_(noc), accessor_(accessor), sp_(sp), data_(l1_address) {}

    // Load the containing window on demand; accesses within it stay in L1.
    volatile uint32_t& operator[](uint32_t index) {
        const uint32_t offset = (index / free_window_entries) * free_window_bytes;
        if (bytes_ == 0 || offset != offset_) {
            flush();
            offset_ = offset;
            const uint32_t remaining = free_row_bytes - offset;
            bytes_ = remaining < free_window_bytes ? remaining : free_window_bytes;
            noc_.async_read(accessor_, data_, bytes_, {.page_id = sp_, .offset_bytes = offset_}, {});
            noc_.async_read_barrier();
        }
        return data_[index % free_window_entries];
    }

    // Write the loaded window back before switching windows or finishing this SP.
    void flush() {
        if (bytes_ == 0) {
            return;
        }
        noc_.async_write(data_, accessor_, bytes_, {}, {.page_id = sp_, .offset_bytes = offset_});
        noc_.async_write_barrier();
        bytes_ = 0;
    }

private:
    const Noc& noc_;
    const Accessor& accessor_;
    const uint32_t sp_;
    CoreLocalMem<volatile uint32_t> data_;
    uint32_t offset_ = 0;
    uint32_t bytes_ = 0;
};

// Return old IDs on reset, then allocate new IDs and update each SP's free count.
void update_free_lists(
    const Noc& noc, const MetadataBuffers& buffers, uint32_t old_pages, uint32_t next_pages, bool reset) {
    for (uint32_t sp = 0; sp < sp_size; ++sp) {
        const uint32_t old_count = pages_on_sp(old_pages, sp);
        const uint32_t next_count = pages_on_sp(next_pages, sp);
        if (reset ? old_count == 0 && next_count == 0 : old_count == next_count) {
            continue;
        }
        FreeListWindow free_ids(noc, buffers.free_acc, sp, buffers.free_l1_address);
        uint32_t count = buffers.counts[sp];
        if (reset) {
            for (uint32_t i = 0; i < old_count; ++i) {
                const uint32_t page = sp + i * sp_size;
                free_ids[count++] = buffers.table[page];
                buffers.table[page] = 0;
            }
        }
        for (uint32_t i = reset ? 0 : old_count; i < next_count; ++i) {
            auto& id = free_ids[--count];
            buffers.table[sp + i * sp_size] = id;
            id = 0;
        }
        free_ids.flush();
        buffers.counts[sp] = count;
    }
}
}  // namespace

// Apply one server-admitted request to this device's replicated metadata.
void kernel_main() {
    Noc noc;
    uint32_t rt_args_idx = 0;
    const auto buffers = make_metadata_buffers(rt_args_idx);
    const uint32_t slot = get_arg_val<uint32_t>(rt_args_idx++);
    const bool reset = get_arg_val<uint32_t>(rt_args_idx++) == 0;
    const uint32_t end = get_arg_val<uint32_t>(rt_args_idx++);

    const uint32_t old_pages = read_allocated_pages(noc, buffers, slot);
    const uint32_t required_pages = end / page_size + (end % page_size != 0);
    const uint32_t next_pages = reset || required_pages > old_pages ? required_pages : old_pages;
    if (!reset && next_pages == old_pages) {
        return;
    }

    read_slot_metadata(noc, buffers, slot);
    update_free_lists(noc, buffers, old_pages, next_pages, reset);
    write_slot_metadata(noc, buffers, slot, next_pages);
}
