# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""dense_sp cache-read through ring_joint's trace-safe metadata path at the quick 2x256 shape, checked against the
host-int path (see check_metadata_flow in ring_joint_cache_read_helpers):
  - bit-exact at two chunk depths served by one cached program (logical_n is off the hash on this path);
  - a cached program follows freshly allocated metadata tensors;
  - dense_sp's chunk write lands in the tensor-selected slot and offset;
  - one captured trace re-targets the user slot and the depth by rewriting tensors in place between replays.

Two users x two layers with DISTINCT K/V, so a read from the wrong slot or layer changes the output instead of
reproducing it. The real M3 chunk (2x5120) and the op's rejections for this path run nightly:
tests/nightly/test_ring_joint_cache_read_metadata.py.
"""

import pytest

from ..test_factory import parametrize_mesh_with_fabric
from .ring_joint_cache_read_helpers import build_cache_read_case, check_metadata_flow


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("n_chunks,chunk_local", [(2, 32)], ids=["2x256"])
def test_ring_joint_cache_read_metadata_trace(mesh_device, device_params, n_chunks, chunk_local, reset_seeds):
    check_metadata_flow(build_cache_read_case(mesh_device, n_chunks, chunk_local))
