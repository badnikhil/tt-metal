# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Shared device-fixture profiles for scoped prefill tests.

Return a fresh dictionary for every ``pytest.param`` so cases cannot share mutable fixture state.
"""

import re
from pathlib import Path

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size


def fabric2d_device_params(*, fabric_payload_size=None, **overrides) -> dict:
    """Unwrapped local Fabric2D profile for 2x2/2x4/4x2 Blackhole tests."""
    params = {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D,
        "fabric_router_config": create_fabric_router_config(
            max_payload_size=get_max_payload_size() if fabric_payload_size is None else fabric_payload_size
        ),
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    }
    params.update(overrides)
    return params


def fabric_1d_device_params(*, fabric_payload_size=None, **overrides) -> dict:
    """Unwrapped 1D fabric profile."""
    params = {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "fabric_router_config": create_fabric_router_config(
            max_payload_size=get_max_payload_size() if fabric_payload_size is None else fabric_payload_size
        ),
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    }
    params.update(overrides)
    return params


def fabric_1d_plain_device_params(**overrides) -> dict:
    """Bare 1D fabric profile: ``fabric_config`` and nothing else.

    ``fabric_1d_device_params`` above additionally pins a router config and RELAXED_INIT, which the
    dispatch/combine tests want. The op-level CSA tests deliberately do not -- they open a 1D fabric on
    its defaults, which is what they were characterized against, and adding a router config to them is
    a behavioral change that needs its own hardware run. This exists so those tests still get a FRESH
    dict per ``pytest.param`` (a shared literal is mutable fixture state) and so the profile has one
    home instead of one copy per test file.
    """
    params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}
    params.update(overrides)
    return params


def fabric_disabled_device_params(**overrides) -> dict:
    """No fabric, for the single-device (1x1) cases that run no collective at all."""
    params = {"fabric_config": ttnn.FabricConfig.DISABLED}
    params.update(overrides)
    return params


def torus_y_device_params(*, fabric_payload_size=None, **overrides) -> dict:
    """Fabric2D Ring/Linear profile for an Nx1 mesh."""
    params = {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
        "fabric_router_config": create_fabric_router_config(
            max_payload_size=get_max_payload_size() if fabric_payload_size is None else fabric_payload_size
        ),
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    }
    params.update(overrides)
    return params


def torus_x_device_params(*, fabric_payload_size=None, **overrides) -> dict:
    """Fabric2D Linear/Ring profile for a 1xN mesh."""
    params = {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
        "fabric_router_config": create_fabric_router_config(
            max_payload_size=get_max_payload_size() if fabric_payload_size is None else fabric_payload_size
        ),
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    }
    params.update(overrides)
    return params


def torus_xy_device_params(*, fabric_payload_size=None, **overrides) -> dict:
    """Production 8x4 Ring/Ring profile; requires a cabling-certified explicit descriptor."""
    params = {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        "fabric_router_config": create_fabric_router_config(
            max_payload_size=get_max_payload_size() if fabric_payload_size is None else fabric_payload_size
        ),
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    }
    params.update(overrides)
    return params


def assert_torus_xy_descriptor(path: str) -> None:
    """Fail before device open unless every declared device mesh is 8x4 Ring/Ring."""
    descriptor_path = Path(path).resolve()
    text = descriptor_path.read_text()
    declared = len(re.findall(r"device_topology\s*\{", text))
    topologies = re.findall(
        r"device_topology\s*\{[^}]*dims:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\][^}]*"
        r"dim_types:\s*\[\s*(\w+)\s*,\s*(\w+)\s*\]",
        text,
    )
    assert topologies, f"{descriptor_path}: no two-dimensional device topology found"
    assert len(topologies) == declared, (
        f"{descriptor_path}: {declared} device_topology block(s) declared but only "
        f"{len(topologies)} are two-dimensional; TorusXY requires every mesh to be 8x4 Ring/Ring"
    )
    assert all(
        topology == ("8", "4", "RING", "RING") for topology in topologies
    ), f"{descriptor_path}: TorusXY requires every device topology to be 8x4 Ring/Ring; found {topologies}"
