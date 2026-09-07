# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared pieces of the ring_joint cache-read tests on (8,4): the GQA shapes, the block-cyclic chunk builders and
gather, the dense SDPA configs, the 1-element uint32 scalars the metadata path reads on-device, and the two checks
the unit (2x256) and nightly (2x5120) metadata tests share: `check_metadata_flow` (dense_sp through the metadata
path vs the host-int path, eager and traced) and `check_metadata_rejections` (the op's contract for that path)."""

from types import SimpleNamespace

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.minimax_m3.tt.attention.dense_sp import dense_sp_attention
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

NQ, NKV, HEAD_DIM = 64, 4, 128
# Two users x two layers, attention on layer 1: exercises the (user, layer) cache fold beyond slot 0 / layer 0.
NUM_USERS, NUM_LAYERS, LAYER_IDX = 2, 2, 1
# SP over mesh rows, TP over mesh cols: the sequence (dim 2) shards over the SP axis, heads (dim 1) over the other.
SP_AXIS = 0
SHARD_DIMS = (2, 1) if SP_AXIS == 0 else (1, 2)
# The host-int path is held to this vs the torch golden (K/V live in the bf8 cache); the metadata path to torch.equal.
PCC_BF8_CACHE = 0.99


def torch_gqa_causal(q, k, v):
    rep = NQ // NKV
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    s = q.shape[2]
    scores = (q @ k.transpose(-1, -2)) * (HEAD_DIM**-0.5)
    causal = torch.triu(torch.full((s, s), float("-inf")), diagonal=1)
    return torch.softmax(scores + causal, dim=-1) @ v  # [1, NQ, S, HD]


def bc_index(kv_actual, sp, chunk_local):
    """Global positions of the chunk starting at kv_actual, in the chip-major block-cyclic order the SP shards hold."""
    pos = rotated_chip_positions(kv_actual, sp, chunk_local)
    return torch.tensor([pos[c][r] for c in range(sp) for r in range(chunk_local)], dtype=torch.long)


def _shard(t, mesh_device, dtype, on_device=True):
    rows, cols = tuple(mesh_device.shape)
    placement = dict(device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) if on_device else {}
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=SHARD_DIMS),
        **placement,
    )


def make_kv_chunk(src, kv_actual, mesh_device, chunk_local):
    """src [heads, S, HD] host K or V -> this chunk, block-cyclic, sharded, in the cache's bf8 dtype."""
    sp = mesh_device.shape[0]
    chunk = src[:, bc_index(kv_actual, sp, chunk_local), :].reshape(1, src.shape[0], sp * chunk_local, HEAD_DIM)
    return _shard(chunk, mesh_device, ttnn.bfloat8_b)


def make_q_chunk(q, kv_actual, mesh_device, chunk_local, on_device=True):
    """q [1, NQ, S, HD] host -> the chunk's queries, block-cyclic, sharded. on_device=False yields the host-side
    twin for copy_host_to_device_tensor (re-targeting a traced Q slab in place)."""
    idx = bc_index(kv_actual, mesh_device.shape[0], chunk_local)
    return _shard(q[:, :, idx, :], mesh_device, ttnn.bfloat16, on_device)


def sdpa_configs(mesh_device):
    """The M3 dense SDPA program / compute configs (minimax3_gqa_causal_perf in
    tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py)."""
    grid = mesh_device.compute_with_storage_grid_size()
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
        q_chunk_size=128,
        k_chunk_size=512,
        exp_approx_mode=False,
    )
    kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    return prog, kcfg


def gather_chunk(out, kv_actual, mesh_device, chunk_local):
    """Per-chip [1, NQ/tp, chunk_local, HD] block-cyclic over the chunk at kv_actual -> [1, NQ, chunk_global, HD] in
    natural order: one composed host read (rows -> seq, cols -> heads), then undo the block-cyclic permutation."""
    rows, cols = tuple(mesh_device.shape)
    chunk_global = rows * chunk_local
    full_bc = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(rows, cols), dims=SHARD_DIMS)
    ).float()
    inv = torch.empty(chunk_global, dtype=torch.long)
    inv[bc_index(kv_actual, rows, chunk_local) - kv_actual] = torch.arange(chunk_global)
    return full_bc[:, :, inv, :]


def meta_scalar(val, mesh_device):
    """1-element uint32 replicated-DRAM scalar, the form update_padded_kv_cache and ring_joint read element [0] of."""
    return ttnn.from_torch(
        torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def host_scalar(val):
    """Host-side twin of meta_scalar, for copy_host_to_device_tensor re-targeting between trace replays."""
    return ttnn.from_torch(
        torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )


def build_cache_read_case(mesh_device, n_chunks, chunk_local, seed=0):
    """Random GQA Q plus DISTINCT K/V per user (so a read from the wrong slot changes the output), the (user,
    layer)-major bf8 caches with every chunk of every user written into (user, LAYER_IDX), the last chunk's queries
    sharded, the dense SDPA configs, and the dense_sp call closures for the host-int and metadata paths."""
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    sp, sp_axis = rows, SP_AXIS
    chunk_global = sp * chunk_local
    cache_global = n_chunks * chunk_global
    kv_actual_last = (n_chunks - 1) * chunk_global

    torch.manual_seed(seed)
    q = torch.randn(1, NQ, cache_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    k = [torch.randn(1, NKV, cache_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1 for _ in range(NUM_USERS)]
    v = [torch.randn(1, NKV, cache_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1 for _ in range(NUM_USERS)]
    refs = [torch_gqa_causal(q.float(), k[u].float(), v[u].float()) for u in range(NUM_USERS)]

    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)
    cache_k = init_kvpe_cache(
        HEAD_DIM, mesh_device, cache_global, list(mesh_device.shape), sp_axis, NUM_LAYERS, NUM_USERS
    )
    cache_v = init_kvpe_cache(
        HEAD_DIM, mesh_device, cache_global, list(mesh_device.shape), sp_axis, NUM_LAYERS, NUM_USERS
    )

    def make_chunk(src, kv_actual):
        return make_kv_chunk(src, kv_actual, mesh_device, chunk_local)

    def make_q(kv_actual, on_device=True):
        return make_q_chunk(q, kv_actual, mesh_device, chunk_local, on_device)

    def gather(out, kv_actual=kv_actual_last):
        return gather_chunk(out, kv_actual, mesh_device, chunk_local)

    def write(cache, src, user, kv_actual):
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache,
            make_chunk(src, kv_actual),
            slot_idx=user,
            layer_idx=LAYER_IDX,
            num_layers=NUM_LAYERS,
            kv_actual_global=kv_actual,
            cluster_axis=sp_axis,
        )

    # Every chunk of every user goes into (user, LAYER_IDX); the other layer's slots stay zero.
    for u in range(NUM_USERS):
        for c in range(n_chunks):
            write(cache_k, k[u][0], u, c * chunk_global)
            write(cache_v, v[u][0], u, c * chunk_global)
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

    def run_host(u, q_t=None, kv_actual=kv_actual_last):
        out = dense_sp_attention(
            tt_q if q_t is None else q_t,
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

    def run_meta(slot_t, kv_t, q_t=None, kv_actual=kv_actual_last):
        return dense_sp_attention(
            tt_q if q_t is None else q_t,
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

    return SimpleNamespace(
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        chunk_local=chunk_local,
        chunk_global=chunk_global,
        cache_global=cache_global,
        kv_actual_last=kv_actual_last,
        q=q,
        k=k,
        v=v,
        refs=refs,
        ccl=ccl,
        cache_k=cache_k,
        cache_v=cache_v,
        tt_q=tt_q,
        prog=prog,
        kcfg=kcfg,
        common=common,
        make_chunk=make_chunk,
        make_q=make_q,
        gather=gather,
        write=write,
        run_host=run_host,
        run_meta=run_meta,
    )


def check_metadata_flow(c):
    """dense_sp through the metadata path vs the host-int path: bit-exact at two chunk depths served by one cached
    program, a cached program follows freshly allocated metadata tensors, the chunk write lands in the
    tensor-selected slot and offset, and one captured trace re-targets the user slot and the depth in place."""
    mesh_device = c.mesh_device
    last = c.kv_actual_last

    host = [c.run_host(u) for u in range(NUM_USERS)]
    for u in range(NUM_USERS):
        passing, pcc = comp_pcc(c.refs[u][:, :, last:, :], host[u], PCC_BF8_CACHE)
        logger.info(f"host-int path user {u}: pcc={pcc}")
        assert passing, f"host-int cache-read PCC fail for user {u}: {pcc}"
    assert not torch.equal(host[0], host[1]), "users must produce distinct outputs for the slot check to mean anything"

    # Metadata tensors live outside the capture and are the only thing that changes between replays.
    t_slot = meta_scalar(0, mesh_device)
    t_kv = meta_scalar(last, mesh_device)

    # Depth 0 first with its real logical_n creates the metadata program; depth 1 must then be a cache HIT that still
    # attends over the full two-chunk prefix, i.e. nothing on the host bounded the program by the depth-0 logical_n.
    tt_q0 = c.make_q(0)
    host0 = c.run_host(0, tt_q0, 0)
    passing, pcc = comp_pcc(c.refs[0][:, :, : c.chunk_global, :], host0, PCC_BF8_CACHE)
    assert passing, f"host-int cache-read PCC fail for user 0 at depth 0: {pcc}"
    entries_before = mesh_device.num_program_cache_entries()
    meta0_d0 = c.gather(c.run_meta(t_slot, meta_scalar(0, mesh_device), tt_q0, 0), 0)
    assert torch.equal(
        meta0_d0, host0
    ), f"metadata path != host-int path for user 0 at depth 0: max_abs={(meta0_d0 - host0).abs().max()}"
    entries_d0 = mesh_device.num_program_cache_entries()
    assert entries_d0 > entries_before, "depth-0 metadata call did not create a program (test setup)"

    # Eager metadata call at depth 1: bit-exact with the host-int path, and it warms the ring-gather buffers.
    meta0 = c.gather(c.run_meta(t_slot, t_kv))
    assert torch.equal(
        meta0, host[0]
    ), f"metadata path != host-int path for user 0: max_abs={(meta0 - host[0]).abs().max()}"
    assert (
        mesh_device.num_program_cache_entries() == entries_d0
    ), "depth 1 compiled a new program: logical_n is still part of the hash on the metadata path"

    # Freshly allocated metadata tensors on a cache hit: the kernels hold the tensors' addresses, so the framework
    # must re-point them. The first tensors stay alive so the new ones land elsewhere; a stale address reads slot 0.
    t_slot_fresh, t_kv_fresh = meta_scalar(1, mesh_device), meta_scalar(last, mesh_device)
    meta1 = c.gather(c.run_meta(t_slot_fresh, t_kv_fresh))
    assert torch.equal(meta1, host[1]), (
        f"cached program did not follow fresh metadata tensors: max_abs vs user 1={(meta1 - host[1]).abs().max()}, "
        f"pcc_vs_user0={comp_pcc(host[0], meta1, 0.0)[1]}"
    )

    # write_chunk on this path must follow the tensors, not the host slot_idx / kv_actual (both left at 0 on
    # purpose): blank user 1's last chunk, rewrite it with the tensors at user 1 / depth 1. A host-slot write would
    # land in user 0 and a host-offset write in depth 0; either way the read of user 1 would see zeros.
    zeros = torch.zeros(NKV, c.cache_global, HEAD_DIM, dtype=torch.bfloat16)
    c.write(c.cache_k, zeros, 1, last)
    c.write(c.cache_v, zeros, 1, last)
    meta1_written = c.gather(
        dense_sp_attention(
            c.tt_q,
            c.cache_k,
            c.cache_v,
            c.make_chunk(c.k[1][0], last),
            c.make_chunk(c.v[1][0], last),
            kv_actual=0,
            logical_n=c.cache_global,
            slot_id=t_slot_fresh,
            kv_actual_isl_tensor=t_kv_fresh,
            **{**c.common, "write_chunk": True},
        )
    )
    assert torch.equal(meta1_written, host[1]), (
        f"write_chunk on the metadata path did not write user 1's slot: max_abs vs user 1="
        f"{(meta1_written - host[1]).abs().max()}"
    )
    meta0_after = c.gather(c.run_meta(t_slot, t_kv))
    assert torch.equal(meta0_after, host[0]), "write_chunk on the metadata path clobbered user 0's slot"

    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out_tr = c.run_meta(t_slot, t_kv)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)

    def replay_expecting(expected, kv_actual, what):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        got = c.gather(out_tr, kv_actual)
        assert torch.equal(got, expected), (
            f"traced replay != host-int path for {what}: max_abs={(got - expected).abs().max()}, "
            f"pcc_vs_user0={comp_pcc(host[0], got, 0.0)[1]}"
        )

    try:
        replay_expecting(host[0], last, "user 0")
        ttnn.copy_host_to_device_tensor(host_scalar(1), t_slot)  # re-target the slot outside the trace
        replay_expecting(host[1], last, "user 1")
        ttnn.copy_host_to_device_tensor(host_scalar(0), t_slot)  # back, to rule out a one-way latch
        replay_expecting(host[0], last, "user 0 again")
        # Depth re-target: Q slab and kv_actual scalar are read by address, so refreshed in place the same captured
        # program attends at depth 0 -- length, Q mapping, ring masks and the gather extent all re-derived on device.
        ttnn.copy_host_to_device_tensor(c.make_q(0, on_device=False), c.tt_q)
        ttnn.copy_host_to_device_tensor(host_scalar(0), t_kv)
        replay_expecting(host0, 0, "user 0 at depth 0")
        ttnn.copy_host_to_device_tensor(c.make_q(last, on_device=False), c.tt_q)
        ttnn.copy_host_to_device_tensor(host_scalar(last), t_kv)
        replay_expecting(host[0], last, "user 0 back at depth 1")
    finally:
        ttnn.release_trace(mesh_device, tid)
    logger.info("ring_joint metadata path: bit-exact vs host-int for both users; trace re-targets slot and depth")


def check_metadata_rejections(c, expect_error):
    """The op's contract for the metadata path, each refused host-side with a message the test pins: tensor form
    (also on a genuine cache hit), pairing, the host/tensor mix, the rotation preconditions, sliding window, and
    the (user, layer) fold bounds on both forms."""
    mesh_device = c.mesh_device
    last = c.kv_actual_last
    t_slot, t_kv = meta_scalar(0, mesh_device), meta_scalar(last, mesh_device)
    c.run_meta(t_slot, t_kv)  # warm the metadata program so the tensor-form cases below can be genuine hits

    # The accessor is baked for a DRAM single-page tensor and the hash never sees the tensor, so another form must be
    # refused on a hit too: an L1 scalar (different memory config) and a two-element DRAM scalar (genuine hit).
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
        c.run_meta(scalar([1], ttnn.L1_MEMORY_CONFIG), t_kv)
    with expect_error(RuntimeError, "metadata tensor slot_id must hold exactly one element"):
        c.run_meta(scalar([1, 1], ttnn.DRAM_MEMORY_CONFIG), t_kv)
    # Both or neither. dense_sp refuses first; the op's own pairing check is shadowed for cache-shaped K/V by the
    # all-gather output-shape check, so only this rejection is pinned.
    with expect_error(ValueError, "must be passed together"):
        c.run_meta(t_slot, None)

    # Direct op call with dense_sp's kwargs plus the metadata tensors; `extra` adds or overrides kwargs.
    def direct_call(**extra):
        kwargs = dict(
            persistent_output_buffer_k=c.ccl.get_ring_gather_buffer(
                "dense_k", NKV, c.cache_global, HEAD_DIM, ttnn.bfloat8_b
            ),
            persistent_output_buffer_v=c.ccl.get_ring_gather_buffer(
                "dense_v", NKV, c.cache_global, HEAD_DIM, ttnn.bfloat8_b
            ),
            joint_strategy="rear",
            logical_n=c.cache_global,
            program_config=c.prog,
            compute_kernel_config=c.kcfg,
            dim=2,
            multi_device_global_semaphore=c.ccl.ring_attention_ccl_semaphore_handles,
            num_links=c.ccl.num_links,
            cluster_axis=c.sp_axis,
            mesh_device=mesh_device,
            topology=ttnn.Topology.Linear,
            ccl_core_grid_offset=c.ccl.ring_attention_ccl_core_grid_offset,
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
            c.tt_q, c.cache_k, c.cache_v, None, None, None, **kwargs
        )

    # Metadata on chunked shapes implies KV-pad rotation, so its preconditions apply; sliding window is refused.
    with expect_error(RuntimeError, "KV-pad rotation .* does not support balanced"):
        direct_call(is_balanced=True)
    with expect_error(RuntimeError, "sliding window is not supported on the metadata path"):
        direct_call(sliding_window_size=128)

    # A host scalar next to the tensors would still steer the all-gather extent, so the mix is refused.
    with expect_error(RuntimeError, "metadata tensors replace the host"):
        direct_call(kv_actual_isl=last)

    # slot_id[0] * kv_cache_num_layers + kv_cache_layer_idx is a DRAM offset used unchecked; bound the host factors.
    with expect_error(RuntimeError, "kv_cache_layer_idx=.* must be < kv_cache_num_layers"):
        direct_call(kv_cache_layer_idx=NUM_LAYERS)
    with expect_error(RuntimeError, "kv_cache_num_layers must be >= 1"):
        direct_call(kv_cache_num_layers=0)
    with expect_error(RuntimeError, "exceeds the KV cache batch"):
        direct_call(kv_cache_num_layers=NUM_USERS * NUM_LAYERS + 1)
    # One fold formula on both paths, so the host form is bounded the same way, folded index included.
    host_form = dict(slot_id=None, kv_actual_isl_tensor=None, kv_actual_isl=last)
    with expect_error(RuntimeError, "kv_cache_layer_idx=.* must be < kv_cache_num_layers"):
        direct_call(kv_cache_batch_idx=1, kv_cache_layer_idx=NUM_LAYERS, **host_form)
    with expect_error(RuntimeError, "is outside the KV cache batch"):
        direct_call(kv_cache_batch_idx=NUM_USERS, **host_form)
