# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V3.2/GLM DSA and DeepSeek-V4 CSA lightning indexers.

A self-contained component owned by ttMLA. It owns the indexer-only state (weights, device
index-key cache, RoPE tables, arch constants) and runs the on-device stems / RoPE / collectives
/ logits / top-k. The shared q_a latent (qr) is passed into forward() by ttMLA; everything else it
reuses from the MLA layer — the SP×TP mesh + axes, compute-kernel configs, softmax scale,
weight-cache location and the CCL handles — is injected through the constructor; the indexer holds
no reference back to ttMLA (and no MLA weights) and runs its own TP/SP collectives. Inert for dense
v3.1 (no indexer weights → ttMLA never builds it).
"""

import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.compressor import TtCSACompressor, rope_table_tokens
from models.demos.deepseek_v3_d_p.tt.mla.mla_config import get_indexer_key_chunk
from models.demos.deepseek_v3_d_p.tt.mla.rope import interleaved_perm_matrix

# DSA indexer weight names are owned by TtIndexer.WEIGHT_NAMES (single source of truth). A
# module-level INDEXER_WEIGHT_NAMES alias is defined at the bottom of this file for back-compat.


# Opt-in diagnostic for the chunked-prefill end-to-end timer. The ring-indexer call is asynchronous,
# so this measures host operation setup / program-cache handling / command submission, not device completion.
# The test driver resets and reads the counters once per chunk.
_fused_ring_host_timing = {"calls": 0, "seconds": 0.0}


def _fused_ring_host_timing_enabled() -> bool:
    # The focused Galaxy prefill pipeline already gives every run a unique summary directory, so enable
    # this diagnostic there without a workflow change. Local/other CI invocations remain opt-in.
    return os.environ.get("TT_FUSED_RING_HOST_TIMING") == "1" or bool(os.environ.get("PREFILL_SUMMARIES"))


def reset_fused_ring_host_timing() -> None:
    if _fused_ring_host_timing_enabled():
        _fused_ring_host_timing["calls"] = 0
        _fused_ring_host_timing["seconds"] = 0.0


def get_fused_ring_host_timing() -> tuple[int, float]:
    return _fused_ring_host_timing["calls"], _fused_ring_host_timing["seconds"]


def normalized_hadamard_matrix(dim: int) -> torch.Tensor:
    """Return the Sylvester-order orthonormal Hadamard matrix used by the decode indexer."""
    assert dim > 0 and dim & (dim - 1) == 0, f"Hadamard dimension must be a power of two, got {dim}"
    matrix = torch.ones(1, 1, dtype=torch.float32)
    while matrix.shape[0] < dim:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)),
            dim=0,
        )
    return (matrix * (dim**-0.5)).to(torch.bfloat16)


class TtIndexerBase:
    """Shared machinery for the lightning-indexer variants: the injected MLA plumbing, the indexer
    geometry / top-k capacity, the weight-cache classmethods, the TP collectives, and the three
    forward stems that are identical across variants (query stem, per-head weights, top-k).

    A variant subclass owns its weight set (``WEIGHT_NAMES`` / ``WEIGHT_DTYPES`` /
    ``REQUIRED_CONFIG_FIELDS`` plus the two weight hooks), how scoring keys are produced
    (``_build_keys``), and how queries are scored against them (``_score``). Everything the base
    provides is expressed in terms of ``self.index_args`` and the uploaded weight attributes, so a
    subclass only has to satisfy those contracts."""

    # Declared, not defined: the base's cache classmethods read these, and a subclass that forgets
    # them fails with AttributeError instead of silently globbing an empty weight set.
    WEIGHT_NAMES: tuple[str, ...]
    WEIGHT_DTYPES: dict
    REQUIRED_CONFIG_FIELDS: tuple[str, ...]

    @classmethod
    def matches_config(cls, config) -> bool:
        """True iff the runtime config carries this variant's indexer fields (dense R1/V3 lacks them)."""
        return all(getattr(config, name, None) is not None for name in cls.REQUIRED_CONFIG_FIELDS)

    @classmethod
    def has_host_weights(cls, state_dict) -> bool:
        """True iff a live state dict carries all indexer host tensors (from-weights callers)."""
        return bool(state_dict) and all(f"{n}.weight" in state_dict for n in cls.WEIGHT_NAMES)

    @classmethod
    def extract_host_weights(cls, state_dict) -> dict:
        """Non-mutating pull of the indexer host tensors out of a state dict (keyed by WEIGHT_NAMES)."""
        return {n: state_dict[f"{n}.weight"] for n in cls.WEIGHT_NAMES if f"{n}.weight" in state_dict}

    @staticmethod
    def _cache_short_name(weight_name: str) -> str:
        """Cache-file stem for a weight name. Override when a device layout differs from what the
        bare name implies, so a tensorbin in the old layout cannot satisfy the new request."""
        return weight_name.split(".")[-1]

    @classmethod
    def check_cache_complete(cls, cache_path, cache_name_prefix: str) -> bool:
        """True iff every indexer tensorbin exists under cache_name_prefix (e.g. 'layer_0.mla') AT ITS
        EXPECTED DTYPE. The glob pins ``_dtype_{name}_`` (WEIGHT_DTYPES) because as_tensor encodes dtype in
        the filename: a dtype-blind glob would accept a stale bf16-only cache for a now-bf8 weight, report
        complete, and let cache-only construction load the empty placeholder as garbage.
        Uses a direct ``Path.glob`` (no `init_checker`/global-state dependency) because this also runs
        at ttMLA construction time — the resolver / __init__ gate — where the global fast-cache checker
        is not necessarily initialized. It's a one-off per-layer check (5 files), so the batch
        fast-checker optimization isn't needed here. Indexer files are `{prefix}.indexer_{short}` — a
        disjoint prefix space from the dense MLA names, so dense and indexer checks never alias."""
        if cache_path is None:
            return False
        cache_path = Path(cache_path)
        for name in cls.WEIGHT_NAMES:
            short = cls._cache_short_name(name)
            dtype_name = cls.WEIGHT_DTYPES[name].name
            if not any(cache_path.glob(f"{cache_name_prefix}.indexer_{short}_dtype_{dtype_name}_*.tensorbin")):
                logger.debug(f"TTNN indexer cache missing: {cache_name_prefix}.indexer_{short} ({dtype_name})")
                return False
        return True

    @classmethod
    def build_ttnn_cache(
        cls, idx_host, cache_path, mesh_device, config, layer_idx, sp_axis: int = 0, tp_axis: int = 1
    ) -> None:
        """Write the indexer tensorbins to disk (device=None, no device copy)."""
        cls._convert_and_cache_weights(
            idx_host, mesh_device, config, layer_idx, sp_axis, tp_axis, cache_path=cache_path, device=None
        )

    @classmethod
    def _convert_and_cache_weights(
        cls, idx_host, mesh_device, config, layer_idx, sp_axis: int = 0, tp_axis: int = 1, cache_path=None, device=None
    ):
        """Variant hook: host indexer weights → device tensors (or cache only, when device is None).
        Returns the device-tensor dict keyed by short name, or None in cache-build mode. A falsy
        ``idx_host`` means cache-only: build host-shaped placeholders and let ``as_tensor`` read the
        existing tensorbins. The weight set is variant-specific, so there is no shared default."""
        raise NotImplementedError

    def _upload_weights(self, idx_host):
        """Variant hook: run ``_convert_and_cache_weights`` against the live mesh and bind the results
        to the instance attributes the forward stems read (``_idx_wq_b`` for ``_q_stem``, ``_idx_wproj``
        for ``_head_weights``, plus whatever the variant's key path needs)."""
        raise NotImplementedError

    def _build_keys(self, *args, **kwargs) -> tuple[ttnn.Tensor, int]:
        """Variant hook: produce the keys this layer scores against, returning them with their count
        ``T``. DSA satisfies this per-token through ``TtIndexer.write_k`` (``T`` = the written prefix);
        a compressed variant returns one entry per compression window (``T`` = S / ratio)."""
        raise NotImplementedError

    def _score(self, q: ttnn.Tensor, keys: ttnn.Tensor, weights: ttnn.Tensor, *args, **kwargs) -> ttnn.Tensor:
        """Variant hook: per-head-weighted, causally-masked logits of ``q`` against ``keys``, shaped
        for ``_topk``. DSA satisfies this with the fused ``ring_indexer_score_dsa`` call in its
        ``forward``, which gathers remote block-cyclic key slabs while it scores."""
        raise NotImplementedError

    def __init__(
        self,
        *,
        config,
        mesh_device,
        sp_axis: int,
        tp_axis: int,
        default_compute_kernel_config,
        hifi4_fp32_compute_kernel_config,
        weight_cache_path,
        layer_idx: int,
        tt_ccl,
        ccl_num_links: int,
        sp_ccl_topology,
        tp_ccl_topology,
        seq_len: int = 1024,
        active_seq_len: int | None = None,
        slot_num: int = 1,
        layer_num: int = 1,
    ):
        """Architecture constants are read from the HF config with no defaults (index_n_heads,
        index_head_dim, index_topk, index_rope_interleave — a sparse config that omits any of them
        fails loudly). θ / YaRN / rope table length come from the same config — single source of
        truth.

        Injected from ttMLA (the indexer keeps no back-reference): the SP×TP mesh + axes,
        compute-kernel configs, weight-cache location, and the CCL handles used by the inlined
        TP collectives and fused SP ring indexer. The indexer derives its own softmax scale
        (index_head_dim**-0.5) — it does NOT reuse MLA's qk_head_dim*mscale scale. The q_a latent
        (qr) is passed into forward(), not held here — so the indexer holds no MLA weights."""
        self.config = config
        self.mesh_device = mesh_device
        self.sp_axis = sp_axis
        self.tp_axis = tp_axis
        mesh_shape = list(mesh_device.shape)
        self.sp_factor = mesh_shape[sp_axis]
        self.tp_factor = mesh_shape[tp_axis]
        self.default_compute_kernel_config = default_compute_kernel_config
        self.hifi4_fp32_compute_kernel_config = hifi4_fp32_compute_kernel_config
        self.weight_cache_path = weight_cache_path
        self.layer_idx = layer_idx
        # Total local layers in the shared key cache. The block-cyclic index_kv_cache is user-major
        # [num_users*layer_num, 1, T, D_idx] — the SAME layout as the MLA KVPE cache — so the flat slot
        # for (user, layer) is cache_user_id*layer_num + cache_layer_idx, where cache_layer_idx is the
        # LOCAL per-rank cache slot passed to forward (mirrors the KVPE cache; NOT self.layer_idx, which is
        # GLOBAL and diverges from the local slot under pipeline parallelism). Computed in forward and used
        # by both write_k and the fused ring indexer's in-kernel slot selection.
        self.layer_num = layer_num
        self.tt_ccl = tt_ccl
        self.ccl_num_links = ccl_num_links
        # Per-axis topology: the TP collectives use tp_ccl_topology, while the fused ring indexer
        # gathers on the SP axis with sp_ccl_topology. Conflating them would deadlock the SP ring
        # under an X-only torus (TP Ring, SP has no physical wrap) — mirrors ttMLA.
        self.sp_ccl_topology = sp_ccl_topology
        self.tp_ccl_topology = tp_ccl_topology
        # Indexer geometry comes from the config with no defaults: a sparse config that omits any of these
        # fields fails loudly here rather than silently binding a wrong-shaped indexer.
        _required = ("index_n_heads", "index_head_dim", "index_topk")
        _missing = [f for f in _required if not hasattr(config, f)]
        assert not _missing, f"sparse MLA config is missing indexer field(s) {_missing}; must define all of {_required}"
        self.index_args = SimpleNamespace(
            index_n_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
            index_topk=config.index_topk,
            index_rope_interleave=getattr(config, "index_rope_interleave", True),
        )
        self.seq_len = seq_len
        # Keep the index tensor width fixed at the configured maximum, except on small-cache test /
        # deployment configurations where the cache itself cannot contain that many entries.  This
        # width is static for the model lifetime; early prefixes fill the unused suffix with the
        # sparse-SDPA sentinel through topk_large_indices(valid_length=...).
        self.index_topk_capacity = min(self.index_args.index_topk, self.seq_len)
        assert 16 <= self.index_topk_capacity <= 2048 and self.index_topk_capacity % 16 == 0, (
            "indexer top-k capacity must be in [16, 2048] and a multiple of 16; " f"got {self.index_topk_capacity}"
        )
        # Chunked sparse MLA has a fixed active prefill chunk.  The TP gathers below are activation
        # collectives, so their maximum output is this chunk (not the growing key-cache length).  Keep
        # the optional default for direct indexer users that predate the explicit active-sequence arg.
        self.active_seq_len = active_seq_len if active_seq_len is not None else seq_len
        assert (
            self.active_seq_len % self.sp_factor == 0
        ), f"active_seq_len ({self.active_seq_len}) must divide SP factor ({self.sp_factor})"
        self.active_seq_len_local = self.active_seq_len // self.sp_factor
        # Persistent TP gather outputs consumed by the collectives below. The subclass sizes and
        # allocates them from its own key geometry; they stay None when TP is 1 (the collectives no-op).
        self._k_all_gather_output = None
        self._weights_all_gather_output = None
        self._topk_indices_all_gather_output = None

    # Inlined TP/SP collectives — the indexer owns its own copy so it depends on tt_ccl, not on ttMLA
    # (the dense MLA forward keeps its own equivalents; both go through the same tt_ccl handles).
    def _tp_rs_ag(self, t, rs_only=False):
        """All-reduce over TP = reduce-scatter (dim 3) then all-gather; rs_only stops after the RS."""
        if self.tp_factor == 1:
            return t
        t = ttnn.experimental.reduce_scatter_minimal_async(
            t,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.tp_ccl_topology,
            cluster_axis=self.tp_axis,
        )
        if rs_only:
            return t
        assert self._k_all_gather_output is not None
        assert tuple(t.shape) == (1, 1, self.active_seq_len_local, self.index_args.index_head_dim // self.tp_factor)
        return ttnn.experimental.high_bw_all_gather(
            t,
            dim=3,
            output_tensor=self._k_all_gather_output,
            num_links=self.ccl_num_links,
            cluster_axis=self.tp_axis,
        )

    def _tp_all_reduce_via_gather(self, t):
        """All-reduce over TP via gather (dim 1) + local reduce, instead of _tp_rs_ag's reduce-scatter
        (dim 3) + all-gather. For a narrow dim-3 width (e.g. wts' H_idx=32) that doesn't divide evenly
        into tile-sized TP shards, _tp_rs_ag's reduce-scatter hits ttnn's composite fallback
        (use_composite_reduce_scatter) and balloons into ~30 tilize/pad/slice ops. Gathering on dim 1 —
        the batch/placeholder axis, always size 1 here — has no tile-alignment constraint, so it always
        takes the fused fast path; fast_reduce_nc then sums the gathered TP axis locally (pure on-device
        compute, no fabric traffic). Mirrors ttMLA._kv_stem's kv_a_proj_with_mqa all-reduce (mla.py:
        917-929), measured cheaper even on an 18x-wider tensor than wts."""
        if self.tp_factor == 1:
            return t
        assert self._weights_all_gather_output is not None
        assert tuple(t.shape) == (1, 1, self.active_seq_len_local, self.index_args.index_n_heads)
        t = ttnn.experimental.high_bw_all_gather(
            t,
            dim=1,
            output_tensor=self._weights_all_gather_output,
            num_links=self.ccl_num_links,
            cluster_axis=self.tp_axis,
        )
        return ttnn.experimental.fast_reduce_nc(
            t, dims=[1], output=None, compute_kernel_config=self.hifi4_fp32_compute_kernel_config
        )

    def _tp_all_gather(self, t, dim):
        """All-gather across the TP axis → the TP-seq-shards reassembled to the SP block's full rows,
        replicated on TP. tp=1: no-op. (Spike helper for TP×SP query parallelism: regathers the top-k
        indices that were computed on TP-seq-sharded query rows back to the [1,1,S/sp,k] contract.)"""
        if self.tp_factor == 1:
            return t
        assert dim == 2, "the indexer only regathers TP-split sequence rows"
        assert self._topk_indices_all_gather_output is not None
        assert tuple(t.shape) == (
            1,
            1,
            self.active_seq_len_local // self.tp_factor,
            self.index_topk_capacity,
        )
        return ttnn.experimental.high_bw_all_gather(
            t,
            dim=dim,
            output_tensor=self._topk_indices_all_gather_output,
            num_links=self.ccl_num_links,
            cluster_axis=self.tp_axis,
        )

    # Forward stems shared by every variant. They read the weights bound by _upload_weights.
    def _q_stem(self, qr: ttnn.Tensor) -> ttnn.Tensor:
        """Shared q_a latent (qr) -> indexer wq_b -> per-head queries [1, H_idx, S/sp, D_idx]. Stops
        short of RoPE and the Hadamard rotation, which are variant-specific."""
        q = ttnn.linear(
            qr,
            self._idx_wq_b,
            compute_kernel_config=self.default_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # Preserve the unquantized values through Hadamard, then materialize BFP8 once downstream.
            dtype=ttnn.bfloat16,
        )  # [1, 1, S/sp, H_idx*D_idx] — ALL heads (wq_b replicated); queries stay SP-sharded (rotation-safe)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q,
            num_heads=self.index_args.index_n_heads,
            num_kv_heads=0,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, H_idx, S/sp, D_idx] — all heads resident
        return q

    def _head_weights(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """weights_proj: device stem -> FULL all-reduce over tp (all H_idx heads, matching the replicated
        wq_b heads) -> scale -> the per-head weights [1, H_idx, S/sp, 1] the score op wants."""
        wts = ttnn.linear(
            hidden_states,
            self._idx_wproj,
            compute_kernel_config=self.default_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # H_idx=32 doesn't divide evenly into tile-sized TP=4 shards (8 < tile width), so _tp_rs_ag's
        # dim-3 reduce-scatter would hit ttnn's composite fallback (~30 extra tilize/pad/slice ops, see
        # use_composite_reduce_scatter). Gather-then-local-reduce on dim 1 has no such tile constraint.
        wts = self._tp_all_reduce_via_gather(wts)  # full all-reduce over tp -> all H_idx head-weights, replicated
        # Indexer softmax scale = index_head_dim**-0.5 (NO mscale), matching the reference IndexerCPU
        # (model.py: softmax_scale = head_dim**-0.5). Distinct from MLA's qk_head_dim*mscale**2 scale —
        # though as a uniform positive multiplier it cannot change the top-k selection regardless.
        wts = ttnn.multiply(
            wts, self.index_args.index_n_heads**-0.5 * self.index_args.index_head_dim**-0.5
        )  # [1,1,S/sp,H_idx] repl on tp
        # the score op wants per-head weights [1, H_idx, S/sp, 1]; wts is [1, 1, S/sp, H_idx].
        return ttnn.permute(wts, (0, 3, 2, 1))

    def _topk(self, logits: ttnn.Tensor, valid_length: int) -> ttnn.Tensor:
        """Top-k key indices [1,1,S/sp,k] (ROW_MAJOR uint32) over ``logits``, regathered across TP when
        the queries were TP-seq-split. Future/pad -inf columns surface as the 0xFFFFFFFF sentinel that
        sparse_mla drops. The score/cache contract requires a 16-element-aligned key prefix; this is
        independent of the fixed top-k capacity."""
        assert valid_length % 16 == 0, f"indexer cache prefix must be 16-element aligned; got end_pos={valid_length}"
        # Block-cyclic logits are the full preallocated width T with a stale [valid_length, T) tail (kv_len
        # only wrote the real prefix); valid_length bounds top-k to that prefix so the tail is never read
        # or ranked — it is the future top-k would drop anyway (causally -inf), so the selection is unchanged.
        idx = ttnn.experimental.topk_large_indices(logits, k=self.index_topk_capacity, valid_length=valid_length)
        # TP×SP: topk ran on the TP-seq-sharded rows ([1,1,S/(sp·tp),k]); regather over TP back to the
        # [1,1,S/sp,k] contract so sparse_sdpa/mla.py are unchanged. (Redundant TP-round-trip for GLM's
        # head→seq reshard, which re-splits it; correct regardless. tp=1: no-op.)
        if self.tp_factor > 1:
            # Regather the TP-seq-sharded top-k indices back to [1,1,S/sp,k]. topk_large_indices emits
            # ROW_MAJOR uint32, and an all-gather on a ROW_MAJOR tensor is routed by use_composite_all_gather
            # to composite_all_gather -> all_broadcast, whose multicast over a partial cluster-axis line of a
            # 2D (SP×TP) mesh DEADLOCKS the fabric (erisc routers stall in run_receiver_channel_step; device
            # unrecoverable, system_memory_manager.cpp TIMEOUT). Gather in TILE layout so it takes the NATIVE
            # minimal all-gather instead — the tile-aligned gather dim keeps it off the composite path, and
            # the native path handles this TP cluster-axis correctly (as _tp_rs_ag does, and as the canonical
            # top-k-index gather in tt_sampling.py does). Round-trip RM->TILE->gather->RM.
            idx_local = idx
            idx_tiled = ttnn.to_layout(idx, ttnn.TILE_LAYOUT)
            idx_gathered = self._tp_all_gather(idx_tiled, dim=2)  # native all-gather over TP; [1,1,S/sp,k] TILE
            idx = ttnn.to_layout(idx_gathered, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(idx_local)
            ttnn.deallocate(idx_tiled)
            # high_bw_all_gather returns a fresh wrapper around model-owned scratch; do not
            # deallocate its backing buffer on the hot path.
        return idx

    def _init_block_cyclic_cache_layout(self, first_layer_idx: int | None) -> None:
        """Initialize the user/layer slot mapping shared by block-cyclic indexers."""
        num_full = num_full_indexer_layers(self.config)
        self._is_index_compact = num_full is not None
        base = full_indexer_rank(self.config, first_layer_idx) if first_layer_idx is not None else 0
        if not self._is_index_compact:
            self._index_layer_idx = self.layer_idx
            self._index_cache_layers = self.layer_num
        else:
            self._index_layer_idx = full_indexer_rank(self.config, self.layer_idx) - base
            self._index_cache_layers = (
                num_full
                if first_layer_idx is None
                else full_indexer_rank(self.config, first_layer_idx + self.layer_num) - base
            )

    def _cache_slot(self, cache_layer_idx: int) -> int:
        """Translate a normal layer slot into the compact index-cache slot."""
        return self._index_layer_idx if self._is_index_compact else cache_layer_idx

    @property
    def index_cache_layers(self) -> int:
        """Layer stride of the block-cyclic key cache, i.e. the ``num_layers`` the write op is given.
        Equals ``layer_num`` normally and the compact full-layer count under GLM-5.2 indexer reuse, so a
        caller validating a cache it owns has to size ``shape[0]`` against this, not against
        ``layer_num``."""
        return self._index_cache_layers

    def _alloc_indexer_buffers(self, *, need_k_all_gather: bool) -> None:
        """Allocate the stable TP scratch used by both DSA and CSA indexers."""
        if self.tp_factor == 1:
            return
        assert (
            self.active_seq_len_local % self.tp_factor == 0
        ), f"local active_seq_len ({self.active_seq_len_local}) must divide TP factor ({self.tp_factor})"
        assert (self.active_seq_len_local // self.tp_factor) % ttnn.TILE_SIZE == 0, (
            "the TP-local active sequence length must be tile aligned for high_bw_all_gather; "
            f"got {self.active_seq_len_local // self.tp_factor}"
        )
        if need_k_all_gather:
            assert self.index_args.index_head_dim % (self.tp_factor * ttnn.TILE_SIZE) == 0, (
                "the TP-local index head dimension must be tile aligned for high_bw_all_gather; "
                f"got {self.index_args.index_head_dim // self.tp_factor}"
            )
            self._k_all_gather_output = self.tt_ccl.get_mla_high_bw_all_gather_buffer(
                name="indexer_k_all_reduce",
                shape=[1, 1, self.active_seq_len_local, self.index_args.index_head_dim],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        self._weights_all_gather_output = self.tt_ccl.get_mla_high_bw_all_gather_buffer(
            name="indexer_weights_all_reduce",
            shape=[1, self.tp_factor, self.active_seq_len_local, self.index_args.index_n_heads],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        self._topk_indices_all_gather_output = self.tt_ccl.get_mla_high_bw_all_gather_buffer(
            name="indexer_topk_indices",
            shape=[1, 1, self.active_seq_len_local, self.index_topk_capacity],
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
        )

    def _init_index_hadamard(self) -> None:
        """Upload the decode-compatible normalized Hadamard basis."""
        hadamard = normalized_hadamard_matrix(self.index_args.index_head_dim).reshape(
            1, 1, self.index_args.index_head_dim, self.index_args.index_head_dim
        )
        self._index_hadamard = ttnn.from_torch(
            hadamard,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _apply_index_hadamard(self, tensor: ttnn.Tensor, *, dtype) -> ttnn.Tensor:
        return ttnn.matmul(
            tensor,
            self._index_hadamard,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )

    def _ring_score_topk(
        self,
        q: ttnn.Tensor,
        weights: ttnn.Tensor,
        index_kv_cache: ttnn.Tensor,
        *,
        cache_batch_idx: int,
        seq_len: int,
        start_pos: int,
        key_compression_ratio: int = 1,
    ) -> ttnn.Tensor:
        """Shared GLM ring-score path, with query positions mapped to compressed keys when requested."""
        assert key_compression_ratio > 0
        end_pos = start_pos + seq_len * self.sp_factor
        assert end_pos % key_compression_ratio == 0
        valid_keys = end_pos // key_compression_ratio

        tpsp = self.tp_factor > 1
        if tpsp:
            q_full, weights_full = q, weights
            q = ttnn.mesh_partition(q, dim=2, cluster_axis=self.tp_axis)
            weights = ttnn.mesh_partition(weights, dim=2, cluster_axis=self.tp_axis)
            ttnn.deallocate(q_full)
            ttnn.deallocate(weights_full)
            sq_local = seq_len // self.tp_factor
            q_chunk = 64 if sq_local % 64 == 0 else 32
        else:
            q_chunk = 64

        key_chunk = get_indexer_key_chunk(self.index_args.index_n_heads)
        program_config = ttnn.IndexerScoreProgramConfig(
            q_chunk_size=q_chunk,
            k_chunk_size=min(key_chunk, valid_keys),
            head_group_size=0,
        )
        gathered_k = self.tt_ccl.get_indexer_ring_k_buffer(local_k=index_kv_cache, sp_axis=self.sp_axis)
        host_start = time.perf_counter() if _fused_ring_host_timing_enabled() else None
        logits = ttnn.experimental.ring_indexer_score_dsa(
            q,
            gathered_k,
            weights,
            index_kv_cache,
            self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.sp_axis),
            cluster_axis=self.sp_axis,
            topology=self.sp_ccl_topology,
            num_links=self.ccl_num_links,
            chunk_start_idx=start_pos,
            program_config=program_config,
            seq_subshard_axis=self.tp_axis if tpsp else None,
            cache_batch_idx=cache_batch_idx,
            block_cyclic_sp_axis=self.sp_axis,
            block_cyclic_chunk_local=seq_len,
            kv_len=valid_keys,
            key_compression_ratio=key_compression_ratio,
        )
        if host_start is not None:
            _fused_ring_host_timing["calls"] += 1
            _fused_ring_host_timing["seconds"] += time.perf_counter() - host_start
        return self._topk(logits, valid_keys)


class TtIndexer(TtIndexerBase):
    """DSA lightning indexer for one MLA layer. Self-contained: owns the indexer weights, the
    grown-by-concat device index-key cache and the indexer RoPE tables, and runs its own TP/SP
    collectives. All MLA-layer dependencies it reuses are injected at construction (no ttMLA ref)."""

    # --- DSA ownership: weight names, config-field detection, host/cache API. ttMLA routes all
    # indexer weight-name / config-field / cache-file / placeholder decisions through these so the
    # DSA facts live here, not in ttMLA. Mirrors dense ttMLA's check/build/convert cache pattern.
    WEIGHT_NAMES = (
        "indexer.wq_b",
        "indexer.wk",
        "indexer.k_norm",
        "indexer.k_norm_bias",
        "indexer.weights_proj",
    )
    # Per-weight device dtype, read by both the converter and check_cache_complete. as_tensor stamps
    # dtype.name into the tensorbin filename, so the completeness glob must pin the SAME dtype it will
    # request; a bare-stem glob accepts a stale bf16-only cache for a bf8 request, then silently loads
    # the empty placeholders as garbage weights. wq_b/wk are bf8; the rest bf16.
    WEIGHT_DTYPES = {
        "indexer.wq_b": ttnn.bfloat8_b,
        "indexer.wk": ttnn.bfloat8_b,
        "indexer.k_norm": ttnn.bfloat16,
        "indexer.k_norm_bias": ttnn.bfloat16,
        "indexer.weights_proj": ttnn.bfloat16,
    }
    # Config fields that mark a runtime config as DSA-sparse (index_rope_interleave is optional and
    # defaults to False; the three below are the discriminator vs a dense DeepSeek-V3 / R1 config).
    REQUIRED_CONFIG_FIELDS = ("index_topk", "index_n_heads", "index_head_dim")

    @staticmethod
    def _cache_short_name(weight_name: str) -> str:
        # wq_b's replicated layout uses a distinct cache stem so the pre-replication TP-sharded
        # tensorbin cannot satisfy completeness checks or be loaded by cache-only construction.
        return "wq_b_repl" if weight_name == "indexer.wq_b" else weight_name.split(".")[-1]

    @classmethod
    def _convert_and_cache_weights(
        cls, idx_host, mesh_device, config, layer_idx, sp_axis: int = 0, tp_axis: int = 1, cache_path=None, device=None
    ):
        """Indexer weights → device (or cache). Mirrors dense MLA's converter:
        - host tensors present: transpose/shard/replicate and (optionally) write the cache;
        - `idx_host` falsy + `device=mesh_device`: build `torch.empty()` placeholders in the host
          (pre-transpose) shapes and rely on existing tensorbins (`as_tensor` ignores the placeholder
          on a cache hit);
        - `device=None`: build the cache only, return None.
        Returns the device-tensor dict keyed by short name (wq_b/wk/weights_proj/k_norm/k_norm_bias),
        or None when device is None. Cache filenames stay byte-compatible with the previously
        opportunistic `layer_{i}.mla.indexer_*` files (same dtype/layout/mapper)."""
        # A sparse config must carry the indexer geometry; assert loudly rather than silently defaulting
        # to garbage shapes.
        _missing = [f for f in ("index_n_heads", "index_head_dim") if not hasattr(config, f)]
        assert not _missing, f"indexer weight conversion requires config field(s) {_missing}"
        index_n_heads = config.index_n_heads
        index_head_dim = config.index_head_dim
        q_lora_rank = config.q_lora_rank
        hidden_size = config.hidden_size

        def _cache_name(short):
            return str(cache_path / f"layer_{layer_idx}.mla.indexer_{short}") if cache_path else None

        # A device load with no host weights must be backed by a complete tensorbin set, else
        # `as_tensor` converts the empty placeholders into garbage indexer weights. Mirror dense MLA's
        # lenient placeholder load (don't block construction) — but, unlike dense which is silent, WARN
        # loudly so the misuse is visible. The layer still stays sparse (binds TtIndexer); it does not
        # fall back to dense. (Build mode, device=None, is gated upstream by ttMLA.build_ttnn_cache.)
        if not idx_host and device is not None and not cls.check_cache_complete(cache_path, f"layer_{layer_idx}.mla"):
            logger.warning(
                f"Sparse MLA layer {layer_idx}: indexer has neither host weights nor a complete cache at "
                f"{cache_path!r}; loading from empty placeholders — indexer output will be garbage. "
                f"Build the indexer cache or pass the indexer weights."
            )

        if idx_host:
            wq_b = idx_host["indexer.wq_b"]
            wk = idx_host["indexer.wk"]
            wproj = idx_host["indexer.weights_proj"]
            knorm = idx_host["indexer.k_norm"]
            knorm_b = idx_host["indexer.k_norm_bias"]
        else:  # cache-only: placeholders in host (pre-transpose) shapes; as_tensor ignores them on a hit
            wq_b = torch.empty(index_n_heads * index_head_dim, q_lora_rank)
            wk = torch.empty(index_head_dim, hidden_size)
            wproj = torch.empty(index_n_heads, hidden_size)
            knorm = torch.empty(index_head_dim)
            knorm_b = torch.empty(index_head_dim)

        mem = ttnn.DRAM_MEMORY_CONFIG if device else None

        def repl(
            t, short, transpose=False, dtype=ttnn.bfloat16
        ):  # replicate across TP (transpose=True: host [out,in] -> device [in,out])
            return ttnn.as_tensor(
                (t.T if transpose else t).contiguous().to(torch.bfloat16),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=mem,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                cache_file_name=_cache_name(short),
            )

        def shard(
            t, axis, short, dtype=ttnn.bfloat16
        ):  # host [out, in] -> device [in, out], dim `axis` sharded across tp
            dims = [None, None]
            dims[tp_axis] = axis
            return ttnn.as_tensor(
                t.T.contiguous().to(torch.bfloat16),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=mem,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
                cache_file_name=_cache_name(short),
            )

        # wq_b is REPLICATED (all H_idx heads on every chip) so the indexer_score head-sum is COMPLETE
        # on-chip — no TP logit all-reduce. (Was col-parallel/TP-head-sharded; replicating trades a small
        # matmul/score compute bump for dropping the ~end_pos-wide 2-CCL logit all-reduce.) Cache name
        # "wq_b_repl" (not "wq_b") so a stale col-sharded tensorbin can never alias this layout. wk /
        # weights_proj contract over hidden (TP-sharded) → upload transposed+sharded, reduced by _tp_rs_ag.
        # wq_b/wk (Q/K projections) load as bf8, halving their DRAM read. wk tolerates it because the k_norm
        # LayerNorm applied right after it (write_k) cancels the bf8 magnitude error. wq_b has NO downstream
        # normalizer (RoPE only rotates), but the indexer score drives top-k SELECTION, not a value: the
        # head-summed logit absorbs the rounding without moving the selected keys across the top-k boundary
        # (per-layer PCC gate confirms). weights_proj (per-head gate) stays bf16 — the gate is precision-sensitive.
        result = {
            "wq_b": repl(
                wq_b, cls._cache_short_name("indexer.wq_b"), transpose=True, dtype=cls.WEIGHT_DTYPES["indexer.wq_b"]
            ),  # [q_lora_rank, H_idx*D_idx] replicated (all heads)
            "wk": shard(wk, 0, "wk", dtype=cls.WEIGHT_DTYPES["indexer.wk"]),  # [dim, D_idx] sharded on dim
            "weights_proj": shard(
                wproj, 0, "weights_proj", dtype=cls.WEIGHT_DTYPES["indexer.weights_proj"]
            ),  # [dim, H_idx] sharded on dim
            "k_norm": repl(knorm, "k_norm", dtype=cls.WEIGHT_DTYPES["indexer.k_norm"]),  # [D_idx]
            "k_norm_bias": repl(knorm_b, "k_norm_bias", dtype=cls.WEIGHT_DTYPES["indexer.k_norm_bias"]),  # [D_idx]
        }
        if device is None:
            for v in result.values():
                del v
            return None
        return result

    def __init__(
        self,
        idx_host,
        *,
        config,
        mesh_device,
        sp_axis: int,
        tp_axis: int,
        default_compute_kernel_config,
        hifi4_fp32_compute_kernel_config,
        weight_cache_path,
        layer_idx: int,
        tt_ccl,
        ccl_num_links: int,
        sp_ccl_topology,
        tp_ccl_topology,
        seq_len: int = 1024,
        active_seq_len: int | None = None,
        slot_num: int = 1,
        layer_num: int = 1,
        first_layer_idx: int | None = None,
    ):
        """DSA-specific state on top of the shared plumbing in TtIndexerBase.__init__: the
        block-cyclic device index-key cache slot math, the TP gather buffers sized on DSA's key
        geometry, the indexer weights, the RoPE half-split reconciliation and the Hadamard basis."""
        super().__init__(
            config=config,
            mesh_device=mesh_device,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            default_compute_kernel_config=default_compute_kernel_config,
            hifi4_fp32_compute_kernel_config=hifi4_fp32_compute_kernel_config,
            weight_cache_path=weight_cache_path,
            layer_idx=layer_idx,
            tt_ccl=tt_ccl,
            ccl_num_links=ccl_num_links,
            sp_ccl_topology=sp_ccl_topology,
            tp_ccl_topology=tp_ccl_topology,
            seq_len=seq_len,
            active_seq_len=active_seq_len,
            slot_num=slot_num,
            layer_num=layer_num,
        )
        # Block-cyclic key-cache path: mirrors the MLA KVPE cache — a persistent, per-user/layer,
        # block-cyclic ND-sharded key cache written by update_padded_kv_cache and scored by
        # indexer_score_dsa's block-cyclic reader. The rope is always the interleaved on-device INDEXED op
        # (rotary_embedding_indexed). GLM is natively interleaved; DS-v3.2's half-split rope is reconciled
        # by _rope_perm below (permute q/k rope halves so the interleaved op matches the DS reference).
        # ALWAYS block-cyclic: single-shot is folded onto this path as one full-seq chunk at start_pos=0
        # (numerically identical to natural order — the block-cyclic reorder degenerates to a contiguous
        # per-chip SP shard), so the key cache persists layer-stacked and migrates to decode. The MLA that
        # owns this indexer feeds it the indexed rope tables and a caller-allocated index_kv_cache in both
        # modes.
        # Block-cyclic key cache (persistent [num_users*_index_cache_layers,1,S/sp,D_idx]) is NOT owned
        # here: exactly like the MLA KVPE cache, the caller allocates it and passes it into
        # forward(index_kv_cache=...) every call; the indexer never self-allocates it. write_k applies the
        # decode-compatible Hadamard transform and typecasts the key to the cache's dtype before the in-place
        # write, so the caller controls the dtype.
        # GLM-5.2 cross-layer indexer reuse: the index key cache is allocated for full layers only, so this
        # layer writes/reads its compacted rank among them and the folded (user-major) slot stride is the
        # cache's full-layer count, not its layer count. _index_cache_layers is that stride.
        # `first_layer_idx` declares this instance a pipeline stage owning global layers
        # [first_layer_idx, first_layer_idx + layer_num): the cache then holds THAT stage's full layers
        # only, numbered from 0. None means the cache spans the whole model -- what a layer built outside
        # the transformer (unit tests) allocates.
        self._init_block_cyclic_cache_layout(first_layer_idx)
        # Stable, worst-case TP gather outputs (declared None by the base).  Indexer layers execute
        # serially, so TT_CCL shares each buffer across them.  This keeps the high-bandwidth gathers
        # allocation-free and their output address fixed on the hot forward path.
        self._alloc_indexer_buffers(need_k_all_gather=True)
        self._upload_weights(idx_host)
        # DS block-cyclic uses the interleaved rotary_embedding_indexed op, but DS weights emit the
        # half-split (rotate_half) rope arrangement. Permute the rope half (half-split -> interleaved) so
        # the interleaved op pairs the right dims with each frequency; applied to BOTH q and k, the
        # permutation cancels in q·k, so the score (hence top-k) matches the DS half-split reference. GLM
        # is natively interleaved -> no permute. NOTE: the stored key is then in interleaved layout —
        # reindex by rope.interleaved_to_halfsplit_perm to compare it against a half-split reference.
        self._rope_perm = None
        if not self.index_args.index_rope_interleave:
            # rope.interleaved_perm_matrix owns the half-split -> interleaved convention (single source).
            self._rope_perm = ttnn.from_torch(
                interleaved_perm_matrix(64).to(torch.bfloat16),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        # Blaze stores and scores indexer keys in the orthonormal Hadamard basis. Apply the same
        # transform to both Q and K so standalone prefill scores are unchanged while the persistent
        # index cache remains byte-compatible with decode after migration.
        self._init_index_hadamard()

    def _upload_weights(self, idx_host):
        """Indexer weights → device via the shared converter. `idx_host` may be a full host dict
        (from-weights) or falsy (cache-only: the converter builds placeholders and reads the
        `layer_{idx}.mla.indexer_*` tensorbins). The converter also writes the cache opportunistically
        on a from-weights load, exactly as before."""
        w = self._convert_and_cache_weights(
            idx_host,
            self.mesh_device,
            self.config,
            self.layer_idx,
            self.sp_axis,
            self.tp_axis,
            cache_path=self.weight_cache_path,
            device=self.mesh_device,
        )
        self._idx_wq_b = w["wq_b"]
        self._idx_wk = w["wk"]
        self._idx_wproj = w["weights_proj"]
        self._idx_knorm_w = w["k_norm"]
        self._idx_knorm_b = w["k_norm_bias"]

    def _bc_rope_pe(self, x: ttnn.Tensor, rope_tensors: dict, kv_actual_global: int) -> ttnn.Tensor:
        """Block-cyclic INDEXED RoPE on the rope half (first 64) of the last dim (block-cyclic path only).
        x [1, n_heads, S/sp, D_idx] SP-sharded on seq; rope_tensors are the whole-cache block-cyclic
        cos/sin/trans built by RotarySetup.get_rope_tensors_indexed — reused verbatim from ttMLA (the
        interleaved table, same 64-dim, shared with the MLA q_pe/k_pe rope). The op derives each shard-row's
        block-cyclic global position on-device from kv_actual_global, exactly as MLA's _apply_rope_padded,
        so keys land at the same positions update_padded_kv_cache writes them to. For DS (half-split
        weights) self._rope_perm first reorders the rope half into the interleaved arrangement so this
        interleaved op matches the DS reference (the permutation cancels in q·k, applied to both q and k)."""
        h, n = x.shape[1], x.shape[2]
        pe = ttnn.slice(x, [0, 0, 0, 0], [1, h, n, 64])
        nope = ttnn.slice(x, [0, 0, 0, 64], [1, h, n, self.index_args.index_head_dim])
        if self._rope_perm is not None:  # DS: half-split -> interleaved arrangement for the interleaved op
            pe_i = ttnn.linear(pe, self._rope_perm, compute_kernel_config=self.hifi4_fp32_compute_kernel_config)
            ttnn.deallocate(pe)
            pe = pe_i
        pe = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            pe,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            kv_actual_global=kv_actual_global,
            cluster_axis=self.sp_axis,
        )
        out = ttnn.concat([pe, nope], dim=-1)
        ttnn.deallocate(pe)
        ttnn.deallocate(nope)
        return out

    def write_k(
        self, hidden_states, seq_len, start_pos, rope_tensors=None, cache_user_id=0, cache_layer_idx=0, index_kbuf=None
    ):
        """Device K stem (wk + TP all-reduce + k_norm + device rope) written into the device index-key
        cache. forward() calls this on every chunk so the key-cache stays complete — else later chunks
        score against missing keys for the early prefix. (Dense v3.1 binds a NullIndexer, so write_k never
        runs there.) Always block-cyclic (single-shot is folded onto it as one full-seq chunk at
        start_pos=0): rope the PER-CHIP shard at its block-cyclic positions, then write it in place via
        update_padded_kv_cache (per-(user,layer) slot, pad-aware kv_actual_global offset) — no SP
        all-gather, no O(n^2) concat; the cache stays SP-sharded."""
        k = ttnn.linear(
            hidden_states,
            self._idx_wk,
            compute_kernel_config=self.default_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # per-chip partial [1, 1, S/sp, D_idx]
        k = self._tp_rs_ag(k)  # all-reduce over TP
        k = ttnn.layer_norm(
            k,
            weight=self._idx_knorm_w,
            bias=self._idx_knorm_b,
            epsilon=1e-6,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )

        # Rope the per-chip shard at its block-cyclic positions (kv_actual_global=start_pos), then write it
        # into this (user, layer) slot. update_padded_kv_cache places each chip's rows at the block-cyclic
        # offset (pad-aware) — the same math the query/key rope above uses. Single-shot is folded onto this
        # path as one full-seq chunk at start_pos=0, so the indexer is always block-cyclic. num_layers is the
        # compacted stride (_index_cache_layers) so it matches the cache_batch_idx computed in forward().
        cache_layer_idx = self._cache_slot(cache_layer_idx)
        k = self._bc_rope_pe(k, rope_tensors, start_pos)  # [1, 1, S/sp, D_idx] bf16
        k_h = ttnn.matmul(
            k,
            self._index_hadamard,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )
        ttnn.deallocate(k)
        k = k_h
        if k.dtype != index_kbuf.dtype:  # write dtype must match the cache (update_padded_kv_cache asserts)
            k = ttnn.typecast(k, index_kbuf.dtype)
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            index_kbuf,
            k,
            slot_idx=cache_user_id,
            layer_idx=cache_layer_idx,
            num_layers=self._index_cache_layers,
            kv_actual_global=start_pos,
            cluster_axis=self.sp_axis,
        )
        ttnn.deallocate(k)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        qr: ttnn.Tensor,
        seq_len: int,
        start_pos: int = 0,
        rope_tensors: dict = None,
        cache_user_id: int = 0,
        cache_layer_idx: int = 0,
        index_kv_cache: ttnn.Tensor = None,
    ) -> ttnn.Tensor:
        """Indexer forward → top-k key indices [1, 1, S/sp, k] over the device index-key cache, SP-sharded
        on the query axis (each chip scores its own S/sp rows; no Q/W all-gather). Fully on-device:
        stems, RoPE, cache, logits, topk — no host.

        ``index_kv_cache``: the persistent per-user key cache, allocated by the caller and passed in every
        call — the same ownership as ttMLA's KVPE ``kvpe_cache``. ALWAYS required (the indexer never
        self-allocates it): the indexer is always block-cyclic, and single-shot is folded onto that path as
        one full-seq chunk at offset 0, so there is no natural path that skips the cache.

        ``qr`` is the shared q_a latent (q_a_proj + TP all-reduce + q_a_layernorm) — ttMLA computes it once
        and passes it in; the indexer applies wq_b to it (no q_a stem of its own). ``qr`` is NOT deallocated
        here — ttMLA's _q_stem consumes it afterwards. ``hidden_states`` is still needed for the K stem
        (write_k) and the per-head weights (weights_proj). (write_k is called internally here; ttMLA.forward
        only ever calls self._indexer.forward — it never calls write_k directly.)

        ``rope_tensors`` (the MLA's block-cyclic indexed cos/sin/trans) and ``cache_user_id`` (per-user slot)
        drive the per-user block-cyclic key cache + block-cyclic scoring. Scoring and transport are bounded
        by the written prefix, rounded to complete block-cyclic slabs for the fixed-size ring protocol."""
        cache_layer_idx = self._cache_slot(cache_layer_idx)
        # Block-cyclic key cache is caller-owned (like the KVPE cache) — required, never self-allocated.
        assert index_kv_cache is not None, (
            "block-cyclic indexer requires an externally-allocated index_kv_cache passed to forward() "
            "(same ownership as the MLA KVPE cache); none was provided"
        )
        # Flat user-major slot into the shared [num_users*_index_cache_layers, 1, T, D_idx] cache — same
        # formula as ttMLA._cache_batch_idx for the KVPE cache (cache_layer_idx is the LOCAL per-rank cache
        # slot, compacted to the full-layer rank above for GLM-5.2 cross-layer reuse). Written by write_k
        # and selected in-kernel by the fused ring indexer.
        cache_batch_idx = cache_user_id * self._index_cache_layers + cache_layer_idx
        self.write_k(
            hidden_states,
            seq_len,
            start_pos,
            rope_tensors=rope_tensors,
            cache_user_id=cache_user_id,
            cache_layer_idx=cache_layer_idx,
            index_kbuf=index_kv_cache,
        )

        # Q stem: the shared q_a latent (qr) -> indexer wq_b -> per-head queries.
        q = self._q_stem(qr)
        # block-cyclic indexed rope (same op/tables as the key rope + MLA q_pe)
        q_dev = self._bc_rope_pe(q, rope_tensors, start_pos)
        q_h = self._apply_index_hadamard(q_dev, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(q_dev)
        q_dev = q_h

        # Per-head weights [1, H_idx, S/sp, 1]: weights_proj stem + TP all-reduce + indexer scale.
        weights = self._head_weights(hidden_states)
        return self._ring_score_topk(
            q_dev,
            weights,
            index_kv_cache,
            cache_batch_idx=cache_batch_idx,
            seq_len=seq_len,
            start_pos=start_pos,
        )


class TtCsaIndexer(TtIndexerBase):
    """V4 ratio-4 indexer using the same block-cyclic cache and ring scorer as GLM.

    Units. This indexer scores COMPRESSED ENTRIES but is driven in TOKENS, and ``__init__`` converts
    between them, so the inherited field names do not all mean the same thing. ``TtIndexerBase`` is
    shared with the GLM ``TtIndexer``, where one entry is one token and the distinction does not exist,
    so the fields keep their names and the units are recorded here instead:

    - ``__init__(seq_len=)`` and ``__init__(active_seq_len=)``: TOKENS, global.
    - ``self.seq_len``: ENTRIES -- ``seq_len // compress_rate``. Also readable as ``index_entries``.
    - ``self.index_topk_capacity``: ENTRIES, since it is capped by ``self.seq_len``.
    - ``self.max_token_seq_len``: TOKENS -- the ``seq_len`` argument, kept unconverted.
    - ``self.active_seq_len``: TOKENS, global -- the ``active_seq_len`` argument.
    - ``self.active_seq_len_local``: TOKENS, per chip -- what ``forward(seq_len=)`` must equal.
    """

    WEIGHT_NAMES = (
        "compressor.indexer.q_b_proj",
        "compressor.indexer.kv_proj",
        "compressor.indexer.gate_proj",
        "compressor.indexer.position_bias",
        "compressor.indexer.kv_norm",
        "compressor.indexer.scorer.weights_proj",
    )
    WEIGHT_DTYPES = {
        "compressor.indexer.q_b_proj": ttnn.bfloat8_b,
        "compressor.indexer.kv_proj": ttnn.bfloat8_b,
        "compressor.indexer.gate_proj": ttnn.bfloat8_b,
        "compressor.indexer.position_bias": ttnn.bfloat16,
        "compressor.indexer.kv_norm": ttnn.bfloat16,
        "compressor.indexer.scorer.weights_proj": ttnn.bfloat16,
    }
    REQUIRED_CONFIG_FIELDS = ("index_topk", "index_n_heads", "index_head_dim", "compress_rates")

    @classmethod
    def matches_config(cls, config) -> bool:
        return super().matches_config(config) and config.compress_rates.get("compressed_sparse_attention") is not None

    @staticmethod
    def _cache_short_name(weight_name: str) -> str:
        short = weight_name.split(".")[-1]
        if short == "q_b_proj":
            short = "q_b_repl"
        return f"csa_{short}"

    @classmethod
    def has_host_weights(cls, state_dict) -> bool:
        if not state_dict:
            return False
        return all(
            (name in state_dict if name.endswith("position_bias") else f"{name}.weight" in state_dict)
            for name in cls.WEIGHT_NAMES
        )

    @classmethod
    def extract_host_weights(cls, state_dict) -> dict:
        if not state_dict:
            return {}
        result = {}
        for name in cls.WEIGHT_NAMES:
            key = name if name.endswith("position_bias") else f"{name}.weight"
            if key in state_dict:
                result[name] = state_dict[key]
        return result

    @classmethod
    def from_reference(cls, reference, **kwargs) -> "TtCsaIndexer":
        idx_host = {
            "compressor.indexer.q_b_proj": reference.q_b_proj.weight,
            "compressor.indexer.kv_proj": reference.kv_proj.weight,
            "compressor.indexer.gate_proj": reference.gate_proj.weight,
            "compressor.indexer.position_bias": reference.position_bias,
            "compressor.indexer.kv_norm": reference.kv_norm.weight,
            "compressor.indexer.scorer.weights_proj": reference.scorer.weights_proj.weight,
        }
        return cls(idx_host, rotary_emb=reference.rotary_emb, **kwargs)

    @classmethod
    def _convert_and_cache_weights(
        cls, idx_host, mesh_device, config, layer_idx, sp_axis: int = 0, tp_axis: int = 1, cache_path=None, device=None
    ):
        index_n_heads = config.index_n_heads
        index_head_dim = config.index_head_dim
        hidden_size = config.hidden_size
        q_lora_rank = config.q_lora_rank
        compress_rate = config.compress_rates["compressed_sparse_attention"]

        def cache_name(name):
            short = cls._cache_short_name(name)
            return str(cache_path / f"layer_{layer_idx}.mla.indexer_{short}") if cache_path else None

        if idx_host:
            tensors = dict(idx_host)
        else:
            tensors = {
                "compressor.indexer.q_b_proj": torch.empty(index_n_heads * index_head_dim, q_lora_rank),
                "compressor.indexer.kv_proj": torch.empty(2 * index_head_dim, hidden_size),
                "compressor.indexer.gate_proj": torch.empty(2 * index_head_dim, hidden_size),
                "compressor.indexer.position_bias": torch.empty(compress_rate, 2 * index_head_dim),
                "compressor.indexer.kv_norm": torch.empty(index_head_dim),
                "compressor.indexer.scorer.weights_proj": torch.empty(index_n_heads, hidden_size),
            }

        memory_config = ttnn.DRAM_MEMORY_CONFIG if device else None

        def as_tensor(name, tensor, *, transpose=False, shard_hidden=False, reshape=None):
            value = tensor.detach() if hasattr(tensor, "detach") else tensor
            if transpose:
                value = value.T
            if reshape is not None:
                value = value.reshape(reshape)
            mapper = ttnn.ReplicateTensorToMesh(mesh_device)
            if shard_hidden:
                dims = [None, None]
                dims[tp_axis] = 0
                mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)
            return ttnn.as_tensor(
                value.contiguous().to(torch.bfloat16),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                dtype=cls.WEIGHT_DTYPES[name],
                memory_config=memory_config,
                mesh_mapper=mapper,
                cache_file_name=cache_name(name),
            )

        result = {
            "q_b_proj": as_tensor(
                "compressor.indexer.q_b_proj", tensors["compressor.indexer.q_b_proj"], transpose=True
            ),
            "kv_proj": as_tensor(
                "compressor.indexer.kv_proj",
                tensors["compressor.indexer.kv_proj"],
                transpose=True,
                shard_hidden=True,
            ),
            "gate_proj": as_tensor(
                "compressor.indexer.gate_proj",
                tensors["compressor.indexer.gate_proj"],
                transpose=True,
                shard_hidden=True,
            ),
            "position_bias": as_tensor(
                "compressor.indexer.position_bias",
                tensors["compressor.indexer.position_bias"],
                reshape=(1, 1, compress_rate, 2 * index_head_dim),
            ),
            "kv_norm": as_tensor(
                "compressor.indexer.kv_norm",
                tensors["compressor.indexer.kv_norm"],
                reshape=(1, 1, 1, index_head_dim),
            ),
            "weights_proj": as_tensor(
                "compressor.indexer.scorer.weights_proj",
                tensors["compressor.indexer.scorer.weights_proj"],
                transpose=True,
                shard_hidden=True,
            ),
        }
        if device is None:
            for value in result.values():
                del value
            return None
        return result

    def __init__(
        self,
        idx_host,
        *,
        rotary_emb,
        config,
        mesh_device,
        sp_axis: int,
        tp_axis: int,
        default_compute_kernel_config,
        hifi4_fp32_compute_kernel_config,
        weight_cache_path,
        layer_idx: int,
        tt_ccl,
        ccl_num_links: int,
        sp_ccl_topology,
        tp_ccl_topology,
        seq_len: int = 1024,
        active_seq_len: int | None = None,
        slot_num: int = 1,
        layer_num: int = 1,
        first_layer_idx: int | None = None,
    ):
        self.compress_rate = int(config.compress_rates["compressed_sparse_attention"])
        assert self.compress_rate == 4, f"V4 CSA requires compression ratio 4, got {self.compress_rate}"
        assert seq_len % self.compress_rate == 0
        active_tokens = active_seq_len if active_seq_len is not None else seq_len
        assert active_tokens % (self.compress_rate * mesh_device.shape[sp_axis]) == 0
        assert (
            seq_len // self.compress_rate // mesh_device.shape[sp_axis]
        ) % ttnn.TILE_SIZE == 0, "the local compressed cache must be tile aligned"
        assert (
            active_tokens // self.compress_rate
        ) % 16 == 0, "each compressed prefix increment must satisfy topk_large_indices alignment"
        super().__init__(
            config=config,
            mesh_device=mesh_device,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            default_compute_kernel_config=default_compute_kernel_config,
            hifi4_fp32_compute_kernel_config=hifi4_fp32_compute_kernel_config,
            weight_cache_path=weight_cache_path,
            layer_idx=layer_idx,
            tt_ccl=tt_ccl,
            ccl_num_links=ccl_num_links,
            sp_ccl_topology=sp_ccl_topology,
            tp_ccl_topology=tp_ccl_topology,
            seq_len=seq_len // self.compress_rate,
            active_seq_len=active_tokens,
            slot_num=slot_num,
            layer_num=layer_num,
        )
        self.max_token_seq_len = seq_len  # TOKENS; self.seq_len is the same context in ENTRIES
        self.rope_head_dim = int(config.qk_rope_head_dim)
        self._init_block_cyclic_cache_layout(first_layer_idx)
        self._alloc_indexer_buffers(need_k_all_gather=False)
        self._upload_weights(idx_host)
        self._init_index_hadamard()

        self._compressor = TtCSACompressor(
            mesh_device,
            kv_proj_weight=None,
            gate_proj_weight=None,
            position_bias=None,
            kv_norm_weight=None,
            head_dim=self.index_args.index_head_dim,
            compress_rate=self.compress_rate,
            rope_head_dim=self.rope_head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            # The inner compressor projects on the TP axis and exchanges overlap state on the SP axis, so
            # it needs both; one value for both is the deadlock resolve_per_axis_topology describes. Pass
            # the pair only when they differ, since the pair form is positional and asserts sp_axis=0.
            topology=(sp_ccl_topology, tp_ccl_topology) if sp_ccl_topology != tp_ccl_topology else sp_ccl_topology,
            preloaded_weights={
                "kv_proj": self._idx_kv_proj,
                "gate_proj": self._idx_gate_proj,
                "position_bias": self._idx_position_bias,
                "kv_norm": self._idx_knorm,
            },
        )
        self._compressor.alloc_tables(seq_len, active_tokens)
        token_table_rows = rope_table_tokens(seq_len, active_tokens)
        self._query_rope = self._compressor.ops.build_rope_table(token_table_rows, 1)
        self._query_index = self._compressor.ops.rope_index_base(self.active_seq_len_local)
        self.reset_overlap_state()

    def _upload_weights(self, idx_host):
        weights = self._convert_and_cache_weights(
            idx_host,
            self.mesh_device,
            self.config,
            self.layer_idx,
            self.sp_axis,
            self.tp_axis,
            cache_path=self.weight_cache_path,
            device=self.mesh_device,
        )
        self._idx_wq_b = weights["q_b_proj"]
        self._idx_wproj = weights["weights_proj"]
        self._idx_kv_proj = weights["kv_proj"]
        self._idx_gate_proj = weights["gate_proj"]
        self._idx_position_bias = weights["position_bias"]
        self._idx_knorm = weights["kv_norm"]

    def reset_overlap_state(self) -> None:
        """Start over from no predecessor window. The inner compressor owns the layout, since it is the
        one that consumes and emits these.

        ``TtCSA.alloc_state`` calls this so this state and the block's own overlap state begin at the
        same position; ``write_k`` advances it from there, one chunk at a time."""
        if hasattr(self, "_overlap_kv_state"):
            ttnn.deallocate(self._overlap_kv_state)
            ttnn.deallocate(self._overlap_score_state)
        self._overlap_kv_state, self._overlap_score_state = self._compressor.alloc_overlap_state()

    @property
    def index_entries(self) -> int:
        """``self.seq_len`` under the name that says its unit: compressed entries, not tokens. Reading
        it through this alias is what keeps a call site from quietly comparing it against a token
        count."""
        return self.seq_len

    def _rotate_query(self, q: ttnn.Tensor, start_pos: int) -> ttnn.Tensor:
        batch, heads, rows, head_dim = q.shape
        nope_dim = head_dim - self.rope_head_dim
        nope = ttnn.slice(q, [0, 0, 0, 0], [batch, heads, rows, nope_dim])
        rope = ttnn.slice(q, [0, 0, 0, nope_dim], [batch, heads, rows, head_dim])
        index = self._compressor.ops.rope_index(self._query_index, start_pos)
        cos, sin = self._compressor.ops.rope_gather(self._query_rope, index)
        rope = ttnn.experimental.rotary_embedding_llama(
            rope, cos, sin, self._compressor.trans_mat, is_decode_mode=False
        )
        return ttnn.concat([nope, rope], dim=-1)

    def write_k(
        self,
        hidden_states: ttnn.Tensor,
        *,
        seq_len: int,
        start_pos: int,
        cache_user_id: int,
        cache_layer_idx: int,
        index_kbuf: ttnn.Tensor,
        seq_len_actual: int | None = None,
    ) -> None:
        """``seq_len_actual`` is the chunk's real pre-pad length, defaulting to the whole padded slab.
        It has to be the real one whenever another chunk follows: the outgoing overlap state is what that
        chunk's first window reads, so a state that advanced over pad rows would hand it the wrong
        predecessor. The cache write itself always covers the padded width, which the next chunk
        overwrites and the score op's causal mask ignores until then."""
        assert start_pos % self.compress_rate == 0
        prior_kv_state = self._overlap_kv_state
        prior_score_state = self._overlap_score_state
        local_keys, _, kv_state, score_state = self._compressor(
            hidden_states,
            prior_kv_state,
            prior_score_state,
            seq_len_actual=seq_len * self.sp_factor if seq_len_actual is None else seq_len_actual,
            first_window_position=start_pos,
            gather_sp=False,
        )
        keys_h = self._apply_index_hadamard(local_keys, dtype=ttnn.bfloat16)
        ttnn.deallocate(local_keys)
        if keys_h.dtype != index_kbuf.dtype:
            keys = ttnn.typecast(keys_h, index_kbuf.dtype)
            ttnn.deallocate(keys_h)
        else:
            keys = keys_h
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            index_kbuf,
            keys,
            slot_idx=cache_user_id,
            layer_idx=self._cache_slot(cache_layer_idx),
            num_layers=self._index_cache_layers,
            kv_actual_global=start_pos // self.compress_rate,
            cluster_axis=self.sp_axis,
        )
        ttnn.deallocate(keys)
        ttnn.deallocate(prior_kv_state)
        ttnn.deallocate(prior_score_state)
        self._overlap_kv_state = self._compressor.terminal_state(kv_state)
        self._overlap_score_state = self._compressor.terminal_state(score_state)

    def _build_keys(
        self,
        hidden_states,
        *,
        seq_len,
        start_pos,
        cache_user_id,
        cache_layer_idx,
        index_kbuf,
        seq_len_actual=None,
    ):
        self.write_k(
            hidden_states,
            seq_len=seq_len,
            start_pos=start_pos,
            cache_user_id=cache_user_id,
            cache_layer_idx=cache_layer_idx,
            index_kbuf=index_kbuf,
            seq_len_actual=seq_len_actual,
        )
        # The written prefix, which covers the padded slab whatever the real length was.
        return index_kbuf, (start_pos + seq_len * self.sp_factor) // self.compress_rate

    def _score(self, q, keys, weights, **kwargs):
        return self._ring_score_topk(q, weights, keys, key_compression_ratio=self.compress_rate, **kwargs)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        qr: ttnn.Tensor,
        seq_len: int,
        start_pos: int = 0,
        cache_user_id: int = 0,
        cache_layer_idx: int = 0,
        index_kv_cache: ttnn.Tensor = None,
        seq_len_actual: int | None = None,
    ) -> ttnn.Tensor:
        """``seq_len`` is the padded LOCAL slab width in TOKENS, fixed for the whole prefill;
        ``seq_len_actual`` the chunk's real global pre-pad length, also in tokens, which only the overlap
        state needs (see write_k). Neither is an entry count -- see the class docstring on units."""
        assert index_kv_cache is not None, "CSA indexer requires a caller-owned block-cyclic index key cache"
        assert seq_len == self.active_seq_len_local, (
            f"forward(seq_len=) is the local slab width in TOKENS, not entries: got {seq_len}, expected "
            f"{self.active_seq_len_local} (which is {self.active_seq_len_local // self.compress_rate} entries)"
        )
        assert start_pos % self.compress_rate == 0
        cache_layer_idx = self._cache_slot(cache_layer_idx)
        cache_batch_idx = cache_user_id * self._index_cache_layers + cache_layer_idx
        self._build_keys(
            hidden_states,
            seq_len=seq_len,
            start_pos=start_pos,
            cache_user_id=cache_user_id,
            cache_layer_idx=cache_layer_idx,
            index_kbuf=index_kv_cache,
            seq_len_actual=seq_len_actual,
        )
        q = self._q_stem(qr)
        q_rotated = self._rotate_query(q, start_pos)
        ttnn.deallocate(q)
        q_h = self._apply_index_hadamard(q_rotated, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(q_rotated)
        weights = self._head_weights(hidden_states)
        return self._score(
            q_h,
            index_kv_cache,
            weights,
            cache_batch_idx=cache_batch_idx,
            seq_len=seq_len,
            start_pos=start_pos,
        )


class NullIndexer:
    """Dense v3.1 stand-in for TtIndexer: forward() is a no-op returning None (no top-k, no K-cache
    write). Lets ttMLA bind self._indexer once at construction and call it unconditionally in forward.
    Mirrors TtIndexer.forward's contract — keep the two in sync if that signature/return changes."""

    def forward(self, *args, **kwargs):
        return None


class ReuseIndexer:
    """GLM-5.2 ``shared`` DSA layer stand-in: owns no indexer weights and never computes. The layer is
    still sparse (top-k SDPA) but reuses a prior ``full`` layer's top-k indices, injected at
    ttMLA.forward(indexer_indices=...). forward() is unreachable there (the injected indices short-
    circuit it); it raises if ever called, so a shared layer missing its reused indices fails loudly
    instead of silently going dense."""

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "ReuseIndexer.forward called: a GLM-5.2 shared DSA layer must receive reused top-k indices "
            "via MLA.forward(indexer_indices=...)."
        )


# Back-compat alias; TtIndexer.WEIGHT_NAMES is the single source of truth.
INDEXER_WEIGHT_NAMES = TtIndexer.WEIGHT_NAMES


def resolve_has_indexer(config, state_dict=None, explicit=None, weight_cache_path=None, cache_name_prefix=None) -> bool:
    """Single source of truth for "is this a sparse DSA layer?", used by every ttMLA cache/check/
    build/load path so they cannot disagree. Resolution order:
      1. explicit override when not None,
      2. config.has_indexer when present (absence = unknown, NOT False),
      3. TtIndexer.matches_config(config) — runtime config carries DSA index_* fields,
      4. TtIndexer.has_host_weights(state_dict) — live from-weights callers,
      5. TtIndexer.check_cache_complete(...) — cache-only callers with a complete indexer cache,
      6. otherwise dense.
    Never resolve sparse detection through getattr(config, "has_indexer", False): a default-False
    flag silently disables sparse for cache-only construction (the bug this whole path fixes)."""
    if explicit is not None:
        return explicit
    flag = getattr(config, "has_indexer", None)
    if flag is not None:
        return bool(flag)
    if TtIndexer.matches_config(config):
        return True
    if TtIndexer.has_host_weights(state_dict):
        return True
    if weight_cache_path is not None and cache_name_prefix is not None:
        return TtIndexer.check_cache_complete(weight_cache_path, cache_name_prefix)
    return False


def indexer_layer_is_reused(config, layer_idx: int) -> bool:
    """GLM-5.2 ``shared`` layer: sparse attention but owns NO indexer (it reuses a prior ``full`` layer's
    top-k). True iff ``config.indexer_types[layer_idx] == "shared"``. Absent the map (v3.1 / v3.2 /
    GLM-5.1) every layer is a full indexer owner -> current behavior. Single source of truth for the
    device construction (ReuseIndexer binding) and the cache build (skip the indexer tensorbins)."""
    types = getattr(config, "indexer_types", None)
    return bool(types) and layer_idx < len(types) and types[layer_idx] == "shared"


def num_full_indexer_layers(config):
    """Count how many entries equal ``"full"`` in ``config.indexer_types``. Returns ``None`` when the list
    is absent or empty."""
    types = getattr(config, "indexer_types", None)
    if not types:
        return None
    return sum(1 for t in types if t == "full")


def full_indexer_rank(config, layer_idx: int) -> int:
    """Prefix rank over ``config.indexer_types``: count how many entries equal ``"full"`` before position
    ``layer_idx`` (exclusive), renumbering the matching positions into dense 0-based ranks. Returns
    ``layer_idx`` unchanged when the list is absent or empty."""
    types = getattr(config, "indexer_types", None)
    if not types:
        return layer_idx
    return sum(1 for t in types[:layer_idx] if t == "full")
