# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Nightly (BH Galaxy) twin of tests/unit/test_ring_joint_cache_read_metadata_trace.py at the real M3 prefill chunk
(2x5120: 640 tokens per chip at SP=8), plus the op-level contract of ring_joint's metadata path -- every rejection
the op raises for that path, pinned to its message. Both checks live in tests/unit/ring_joint_cache_read_helpers.

Scheduled by tests/pipeline_reorg/blaze_models_prefill_tests.yaml; skips where no (8,4) mesh fits.
"""

import pytest

from ..test_factory import parametrize_mesh_with_fabric
from ..unit.ring_joint_cache_read_helpers import build_cache_read_case, check_metadata_flow, check_metadata_rejections


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("n_chunks,chunk_local", [(2, 640)], ids=["2x5120"])
def test_ring_joint_cache_read_metadata(mesh_device, device_params, expect_error, n_chunks, chunk_local, reset_seeds):
    case = build_cache_read_case(mesh_device, n_chunks, chunk_local)
    check_metadata_flow(case)
    check_metadata_rejections(case, expect_error)
