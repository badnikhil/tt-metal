# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ring_joint cache-read through the trace-safe metadata path, checked against the host-int path:
  - bit-exact at two chunk depths served by one cached program (logical_n is off the hash on this path);
  - a cached program follows freshly allocated metadata tensors;
  - dense_sp's chunk write lands in the tensor-selected slot and offset;
  - the op's rejections for this path (tensor form, host/tensor mix, rotation and fold preconditions);
  - one captured trace re-targets the cache-user slot by rewriting the metadata tensors between replays.

Extends test_ring_joint_cache_read_sp_vs_ref to two users and two layers so the on-device slot fold
(slot_id[0] * kv_cache_num_layers + kv_cache_layer_idx) is exercised, not just slot 0 / layer 0. Both users hold
DISTINCT K/V, so a read that lands in the other user's slot changes the output instead of reproducing it.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.minimax_m3.tt.attention.dense_sp import dense_sp_attention
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric
from .ring_joint_cache_read_helpers import (
    HEAD_DIM,
    NKV,
    NQ,
    SP_AXIS,
    gather_chunk,
    host_scalar,
    make_kv_chunk,
    make_q_chunk,
    meta_scalar,
    sdpa_configs,
    torch_gqa_causal,
)

NUM_USERS, NUM_LAYERS, LAYER_IDX = 2, 2, 1


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize(
    "n_chunks,chunk_local",
    [(2, 32), (2, 640)],  # 2x256 (quick) and 2x5120 — the REAL M3 prefill chunk (640/chip at SP=8)
    ids=["2x256", "2x5120"],
)
def test_ring_joint_cache_read_metadata_trace(
    mesh_device, device_params, expect_error, n_chunks, chunk_local, reset_seeds
):
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    sp, sp_axis = rows, SP_AXIS
    C = chunk_local
    chunk_global = sp * C
    cache_global = n_chunks * chunk_global
    kv_actual_last = (n_chunks - 1) * chunk_global

    torch.manual_seed(0)
    q = torch.randn(1, NQ, cache_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    k = [torch.randn(1, NKV, cache_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1 for _ in range(NUM_USERS)]
    v = [torch.randn(1, NKV, cache_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1 for _ in range(NUM_USERS)]
    refs = [torch_gqa_causal(q.float(), k[u].float(), v[u].float())[:, :, kv_actual_last:, :] for u in range(NUM_USERS)]

    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)
    cache_k = init_kvpe_cache(
        HEAD_DIM, mesh_device, cache_global, list(mesh_device.shape), sp_axis, NUM_LAYERS, NUM_USERS
    )
    cache_v = init_kvpe_cache(
        HEAD_DIM, mesh_device, cache_global, list(mesh_device.shape), sp_axis, NUM_LAYERS, NUM_USERS
    )

    def make_chunk(src, kv_actual):
        return make_kv_chunk(src, kv_actual, mesh_device, C)

    def make_q(kv_actual):
        return make_q_chunk(q, kv_actual, mesh_device, C)

    def gather(out, kv_actual=kv_actual_last):
        return gather_chunk(out, kv_actual, mesh_device, C)

    # Every chunk of every user goes into (user, LAYER_IDX); the other layer's slots stay zero.
    for u in range(NUM_USERS):
        for c in range(n_chunks):
            kv_actual = c * chunk_global
            for cache, src in ((cache_k, k[u][0]), (cache_v, v[u][0])):
                ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                    cache,
                    make_chunk(src, kv_actual),
                    slot_idx=u,
                    layer_idx=LAYER_IDX,
                    num_layers=NUM_LAYERS,
                    kv_actual_global=kv_actual,
                    cluster_axis=sp_axis,
                )
    ttnn.synchronize_device(mesh_device)

    tt_q = make_q(kv_actual_last)
    prog, kcfg = sdpa_configs(mesh_device)
    common = dict(
        n_kv=NKV,
        cache_global=cache_global,
        head_dim=HEAD_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl,
        program_config=prog,
        compute_kernel_config=kcfg,
        scale=HEAD_DIM**-0.5,
        cluster_axis=sp_axis,
        layer_idx=LAYER_IDX,
        num_layers=NUM_LAYERS,
        write_chunk=False,
    )

    def run_host(u, q_t=tt_q, kv_actual=kv_actual_last):
        out = dense_sp_attention(
            q_t,
            cache_k,
            cache_v,
            None,
            None,
            kv_actual=kv_actual,
            logical_n=kv_actual + chunk_global,
            slot_idx=u,
            **common,
        )
        return gather(out, kv_actual)

    host = [run_host(u) for u in range(NUM_USERS)]
    for u in range(NUM_USERS):
        passing, pcc = comp_pcc(refs[u], host[u], 0.99)
        logger.info(f"host-int path user {u}: pcc={pcc}")
        assert passing, f"host-int cache-read PCC fail for user {u}: {pcc}"
    assert not torch.equal(host[0], host[1]), "users must produce distinct outputs for the slot check to mean anything"

    # Metadata tensors live outside the capture and are the only thing that changes between replays.
    t_slot = meta_scalar(0, mesh_device)
    t_kv = meta_scalar(kv_actual_last, mesh_device)

    def run_meta(slot_t, kv_t, q_t=tt_q, kv_actual=kv_actual_last):
        return dense_sp_attention(
            q_t,
            cache_k,
            cache_v,
            None,
            None,
            kv_actual=kv_actual,
            logical_n=kv_actual + chunk_global,
            slot_id=slot_t,
            kv_actual_isl_tensor=kv_t,
            **common,
        )

    # Depth 0 first, with the real (one-chunk) logical_n: this creates the metadata program. Depth 1 then has to
    # be a program-cache HIT that still gathers and attends over the full two-chunk prefix -- the kernels take the
    # length from kv_actual_isl[0], and nothing on the host may have bounded the program by the depth-0 logical_n.
    tt_q0 = make_q(0)
    host0 = run_host(0, tt_q0, 0)
    passing, pcc = comp_pcc(
        torch_gqa_causal(q.float(), k[0].float(), v[0].float())[:, :, :chunk_global, :], host0, 0.99
    )
    assert passing, f"host-int cache-read PCC fail for user 0 at depth 0: {pcc}"
    entries_before = mesh_device.num_program_cache_entries()
    meta0_d0 = gather(run_meta(t_slot, meta_scalar(0, mesh_device), tt_q0, 0), 0)
    assert torch.equal(
        meta0_d0, host0
    ), f"metadata path != host-int path for user 0 at depth 0: max_abs={(meta0_d0 - host0).abs().max()}"
    entries_d0 = mesh_device.num_program_cache_entries()
    assert entries_d0 > entries_before, "depth-0 metadata call did not create a program (test setup)"

    # Eager metadata call at depth 1: bit-exact with the host-int path, and it warms the ring-gather buffers.
    meta0 = gather(run_meta(t_slot, t_kv))
    assert torch.equal(
        meta0, host[0]
    ), f"metadata path != host-int path for user 0: max_abs={(meta0 - host[0]).abs().max()}"
    assert (
        mesh_device.num_program_cache_entries() == entries_d0
    ), "depth 1 compiled a new program: logical_n is still part of the hash on the metadata path"

    # A caller may hand over freshly allocated metadata tensors on a later call (e.g. one kv_actual scalar per
    # depth). That call hits the cached program, whose kernels hold the tensors' DRAM addresses as plain
    # runtime args, so the op must re-point them. The first tensors stay alive so the new ones cannot land at
    # the same addresses; a stale address would read slot 0 here and reproduce user 0's output.
    t_slot_fresh, t_kv_fresh = meta_scalar(1, mesh_device), meta_scalar(kv_actual_last, mesh_device)
    meta1 = gather(run_meta(t_slot_fresh, t_kv_fresh))
    assert torch.equal(meta1, host[1]), (
        f"cached program did not follow fresh metadata tensors: max_abs vs user 1={(meta1 - host[1]).abs().max()}, "
        f"pcc_vs_user0={comp_pcc(host[0], meta1, 0.0)[1]}"
    )

    # With write_chunk the helper also writes the chunk, and on this path the write must follow the tensors, not
    # the host slot_idx / kv_actual (left at their defaults 0 here on purpose). Blank user 1's last chunk, then
    # let the helper rewrite it with the tensors pointing at user 1 / depth 1: a host-slot write would land in
    # user 0 and a host-offset write in depth 0, and either way the read of user 1 would see zeros.
    zeros = torch.zeros(1, NKV, cache_global, HEAD_DIM, dtype=torch.bfloat16)
    for cache in (cache_k, cache_v):
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache,
            make_chunk(zeros[0], kv_actual_last),
            slot_idx=1,
            layer_idx=LAYER_IDX,
            num_layers=NUM_LAYERS,
            kv_actual_global=kv_actual_last,
            cluster_axis=sp_axis,
        )
    meta1_written = gather(
        dense_sp_attention(
            tt_q,
            cache_k,
            cache_v,
            make_chunk(k[1][0], kv_actual_last),
            make_chunk(v[1][0], kv_actual_last),
            kv_actual=0,
            logical_n=cache_global,
            slot_id=t_slot_fresh,
            kv_actual_isl_tensor=t_kv_fresh,
            **{**common, "write_chunk": True},
        )
    )
    assert torch.equal(meta1_written, host[1]), (
        f"write_chunk on the metadata path did not write user 1's slot: max_abs vs user 1="
        f"{(meta1_written - host[1]).abs().max()}"
    )
    meta0_after = gather(run_meta(t_slot, t_kv))
    assert torch.equal(meta0_after, host[0]), "write_chunk on the metadata path clobbered user 0's slot"

    # The program bakes a DRAM-interleaved single-page accessor for each metadata tensor and the hash never sees
    # the tensor itself, so a scalar of another form must be refused on a cache hit as well: an L1 scalar (a
    # different memory config) and a two-element DRAM scalar (same memory config, so a genuine hit).
    def scalar(values, memory_config):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int64).reshape(1, 1, 1, len(values)),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    with expect_error(RuntimeError, "metadata tensor slot_id must be in DRAM"):
        run_meta(scalar([1], ttnn.L1_MEMORY_CONFIG), t_kv)
    with expect_error(RuntimeError, "metadata tensor slot_id must hold exactly one element"):
        run_meta(scalar([1, 1], ttnn.DRAM_MEMORY_CONFIG), t_kv)

    # Direct op call with dense_sp's kwargs plus the metadata tensors, for the rejections below; `extra`
    # adds or overrides kwargs.
    def direct_meta_call(**extra):
        kwargs = dict(
            persistent_output_buffer_k=ccl.get_ring_gather_buffer(
                "dense_k", NKV, cache_global, HEAD_DIM, ttnn.bfloat8_b
            ),
            persistent_output_buffer_v=ccl.get_ring_gather_buffer(
                "dense_v", NKV, cache_global, HEAD_DIM, ttnn.bfloat8_b
            ),
            joint_strategy="rear",
            logical_n=cache_global,
            program_config=prog,
            compute_kernel_config=kcfg,
            dim=2,
            multi_device_global_semaphore=ccl.ring_attention_ccl_semaphore_handles,
            num_links=ccl.num_links,
            cluster_axis=sp_axis,
            mesh_device=mesh_device,
            topology=ttnn.Topology.Linear,
            ccl_core_grid_offset=ccl.ring_attention_ccl_core_grid_offset,
            use_column_major_ccl=True,
            is_causal=True,
            scale=HEAD_DIM**-0.5,
            slot_id=t_slot,
            kv_actual_isl_tensor=t_kv,
            kv_cache_num_layers=NUM_LAYERS,
            kv_cache_layer_idx=LAYER_IDX,
        )
        kwargs.update(extra)
        return ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tt_q, cache_k, cache_v, None, None, None, **kwargs
        )

    # KV-pad rotation is implied by metadata on chunked shapes, so its preconditions must be enforced on this
    # path too; zigzag balancing is one of them. Sliding window is refused outright on this path.
    with expect_error(RuntimeError, "KV-pad rotation .* does not support balanced"):
        direct_meta_call(is_balanced=True)
    with expect_error(RuntimeError, "sliding window is not supported on the metadata path"):
        direct_meta_call(sliding_window_size=128)

    # A host scalar next to the tensors would still steer the all-gather extent, so the mix is refused.
    with expect_error(RuntimeError, "metadata tensors replace the host"):
        direct_meta_call(kv_actual_isl=kv_actual_last)

    # The readers turn slot_id[0] * kv_cache_num_layers + kv_cache_layer_idx into a DRAM offset unchecked, so
    # the two host factors are bounded here: a layer index past the fold would read the next user's slot.
    with expect_error(RuntimeError, "kv_cache_layer_idx=.* must be < kv_cache_num_layers"):
        direct_meta_call(kv_cache_layer_idx=NUM_LAYERS)
    with expect_error(RuntimeError, "kv_cache_num_layers must be >= 1"):
        direct_meta_call(kv_cache_num_layers=0)
    with expect_error(RuntimeError, "exceeds the KV cache batch"):
        direct_meta_call(kv_cache_num_layers=NUM_USERS * NUM_LAYERS + 1)
    # The fold is one formula on both paths, so the host form is bounded the same way, including the folded
    # index against the cache batch.
    host_form = dict(slot_id=None, kv_actual_isl_tensor=None, kv_actual_isl=kv_actual_last)
    with expect_error(RuntimeError, "kv_cache_layer_idx=.* must be < kv_cache_num_layers"):
        direct_meta_call(kv_cache_batch_idx=1, kv_cache_layer_idx=NUM_LAYERS, **host_form)
    with expect_error(RuntimeError, "is outside the KV cache batch"):
        direct_meta_call(kv_cache_batch_idx=NUM_USERS, **host_form)

    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out_tr = run_meta(t_slot, t_kv)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)

    def replay_expecting(u):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        got = gather(out_tr)
        assert torch.equal(got, host[u]), (
            f"traced replay != host-int path for user {u}: max_abs={(got - host[u]).abs().max()}, "
            f"pcc_vs_user0={comp_pcc(host[0], got, 0.0)[1]}"
        )

    try:
        replay_expecting(0)
        ttnn.copy_host_to_device_tensor(host_scalar(1), t_slot)  # re-target outside the trace
        replay_expecting(1)
        ttnn.copy_host_to_device_tensor(host_scalar(0), t_slot)  # back, to rule out a one-way latch
        replay_expecting(0)
    finally:
        ttnn.release_trace(mesh_device, tid)
    logger.info("ring_joint metadata path: bit-exact vs host-int for both users; trace re-targets 0 -> 1 -> 0")
