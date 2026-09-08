# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math
import numpy as np  # remove this
from loguru import logger
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout, update_process_id
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
import ttnn
from tt_lib.utils import (
    _nearest_y,
)


@pytest.mark.parametrize(
    "input_height, input_width, num_cores, shard_grid, shard_strategy",
    (
        (2048, 320, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (512, 640, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (2048, 1280, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (128, 1280, 40, (8, 5), ttnn.ShardStrategy.WIDTH),
        (8192, 320, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (2048, 640, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (512, 1280, 40, (8, 5), ttnn.ShardStrategy.BLOCK),
        (128, 1280, 32, (4, 8), ttnn.ShardStrategy.BLOCK),
        (512, 1280, 64, (8, 8), ttnn.ShardStrategy.BLOCK),
        # Wt=10 per core (=ceil(1280/32/4)): pre-fix w_blk=min(Wt,8)=8 did not divide Wt and
        # deadlocked / corrupted on batch_b>1. Post-fix factory picks w_blk=5 (largest divisor
        # <=8), so chunks are full (no partial w-block). in0=in1=2 also covers h partial-block.
        (128, 1280, 4, (8, 5), ttnn.ShardStrategy.WIDTH),
    ),
)
@pytest.mark.parametrize(
    "in0_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "in1_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "op",
    [ttnn.BcastOpMath.ADD, ttnn.BcastOpMath.MUL],
)
@pytest.mark.parametrize("in1_batch_size", [1, 2])
@pytest.mark.parametrize("in0_batch_size", [1, 2])
@pytest.mark.parametrize(
    "orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_bcast(
    device,
    orientation,
    in0_batch_size,
    in1_batch_size,
    input_height,
    input_width,
    num_cores,
    shard_grid,
    shard_strategy,
    in0_dtype,
    in1_dtype,
    op,
):
    torch.manual_seed(0)
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        if shard_strategy == ttnn.ShardStrategy.BLOCK:
            shard_grid = (
                (shard_grid[0], 4)
                if shard_grid[1] == 8 and orientation == ttnn.ShardOrientation.COL_MAJOR
                else shard_grid
            )
            shard_grid = (
                (4, shard_grid[1])
                if shard_grid[0] == 8 and orientation == ttnn.ShardOrientation.ROW_MAJOR
                else shard_grid
            )
    input_shape = [in0_batch_size, 1, input_height, input_width]
    input = torch.rand(input_shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT, dtype=in0_dtype
    )
    input_2d_height = input_tensor.padded_shape[0] * input_tensor.padded_shape[1] * input_tensor.padded_shape[2]
    input_2d_width = input_tensor.padded_shape[3]
    if shard_strategy == ttnn.ShardStrategy.BLOCK:
        input_2d_height_padded = _nearest_y(input_2d_height, shard_grid[0] * 32)
        shard_height = math.ceil(input_2d_height_padded / shard_grid[0])
        shard_width = math.ceil(input_2d_width / shard_grid[1])
        shard_orientation = orientation
        core_grid = (
            ttnn.CoreGrid(y=shard_grid[0], x=shard_grid[1])
            if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR
            else ttnn.CoreGrid(y=shard_grid[1], x=shard_grid[0])
        )
    else:
        shard_height = input_2d_height
        shard_width = math.ceil(input_2d_width / num_cores)
        shard_orientation = orientation
        core_grid = get_shard_grid_from_num_cores(num_cores, device)

    logger.debug(f"core_grid={core_grid}")
    logger.debug(f"input_2d_height={input_2d_height} and input_2d_width={input_2d_width}")
    logger.debug(f"shard_height={shard_height} and shard_width={shard_width}")

    in_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(
            (shard_height, shard_width)
            if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR
            else (shard_width, shard_height)
        ),
        core_grid=core_grid,
        strategy=shard_strategy,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    tt_input = ttnn.to_memory_config(input_tensor, memory_config=in_sharded_mem_config)

    if in0_batch_size == 1 and in1_batch_size > 1:
        input = input.reshape(in1_batch_size, 1, input_height // in1_batch_size, input_width)

    b_weights_shape = [in1_batch_size, 1, 1, input_width]
    B_pyt = torch.rand(size=b_weights_shape).bfloat16()
    if op == ttnn.BcastOpMath.ADD:
        torch_ref_output = torch.add(input, B_pyt)
    elif op == ttnn.BcastOpMath.MUL:
        torch_ref_output = torch.mul(input, B_pyt)

    if in0_batch_size == 1 and in1_batch_size > 1:
        torch_ref_output = torch_ref_output.reshape(1, 1, input_height, input_width)

    B_pyt = B_pyt.reshape(b_weights_shape)
    tt_weight = ttnn.from_torch(B_pyt, device=device, layout=ttnn.TILE_LAYOUT, dtype=in1_dtype)
    tt_output = ttnn.bcast(
        tt_input,
        tt_weight,
        op,
        ttnn.BcastOpDim.H,
        memory_config=ttnn.get_memory_config(tt_input),
    )

    output_tensor = ttnn.to_torch(tt_output).float()
    output_tensor = output_tensor.reshape(input_shape)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_ref_output, output_tensor, 0.999)
    logger.info(pcc_msg)
    assert passing


# With the batch in the C dim (a = [1, C, H, W], C > 1) and input_b's padded_shape[0] == C, the op
# routes to the non-optimised BcastShardedHProgramFactory. Pre-fix that factory passed B = N*C to
# compute while the reader only produces Ht*Wt tiles, so it deadlocked. Existing coverage only puts
# the batch in N ([N, 1, H, W]), so N*C == 1 there and the bug is never exercised.
@pytest.mark.parametrize(
    "batch, height_per_batch, width, num_cores",
    (
        (2, 64, 256, 8),
        (2, 1024, 1280, 40),
    ),
)
@pytest.mark.parametrize(
    "op",
    [ttnn.BcastOpMath.ADD, ttnn.BcastOpMath.MUL],
)
def test_bcast_h_width_sharded_batched_channel(device, batch, height_per_batch, width, num_cores, op):
    torch.manual_seed(0)

    # input_a: [1, batch, H, W] -> flattened 2D height = batch * H (batch folded into C).
    a_shape = [1, batch, height_per_batch, width]
    a_torch = torch.rand(a_shape, dtype=torch.bfloat16)

    # input_b: one broadcast row per batch. b.padded_shape[0] == batch (!= a.padded_shape[0] == 1),
    # which forces the non-optimised sharded-H factory.
    b_torch = torch.rand([batch, 1, 1, width], dtype=torch.bfloat16)

    if op == ttnn.BcastOpMath.ADD:
        torch_ref_output = a_torch + b_torch.reshape(1, batch, 1, width)
    else:
        torch_ref_output = a_torch * b_torch.reshape(1, batch, 1, width)

    input_tensor = ttnn.from_torch(
        a_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    shard_height = batch * height_per_batch
    shard_width = width // num_cores
    core_grid = get_shard_grid_from_num_cores(num_cores, device)
    sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(shard_height, shard_width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_input = ttnn.to_memory_config(input_tensor, memory_config=sharded_mem_config)

    tt_weight = ttnn.from_torch(b_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    tt_output = ttnn.bcast(
        tt_input,
        tt_weight,
        op,
        ttnn.BcastOpDim.H,
        memory_config=ttnn.get_memory_config(tt_input),
    )
    output_tensor = ttnn.to_torch(tt_output).float()

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_ref_output.float(), output_tensor, 0.999)
    logger.info(pcc_msg)
    assert passing, pcc_msg


# Same non-optimised BcastShardedHProgramFactory path as the width-sharded case above, but for
# BLOCK_SHARDED input (batch folded into C, distributed across grid rows). Pre-fix B = N*C was passed
# for block sharding too, so this deadlocked identically.
@pytest.mark.parametrize(
    "batch, height_per_batch, width, shard_grid",
    ((2, 128, 1280, (8, 5)),),
)
@pytest.mark.parametrize(
    "op",
    [ttnn.BcastOpMath.ADD, ttnn.BcastOpMath.MUL],
)
@pytest.mark.parametrize(
    "orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_bcast_h_block_sharded_batched_channel(device, batch, height_per_batch, width, shard_grid, op, orientation):
    torch.manual_seed(0)
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        if shard_grid[1] == 8 and orientation == ttnn.ShardOrientation.COL_MAJOR:
            shard_grid = (shard_grid[0], 4)
        if shard_grid[0] == 8 and orientation == ttnn.ShardOrientation.ROW_MAJOR:
            shard_grid = (4, shard_grid[1])

    # input_a: [1, batch, H, W] -> flattened 2D height = batch * H (batch folded into C).
    a_shape = [1, batch, height_per_batch, width]
    a_torch = torch.rand(a_shape, dtype=torch.bfloat16)

    # input_b: one broadcast row per batch; b.padded_shape[0] == batch forces the non-optimised factory.
    b_torch = torch.rand([batch, 1, 1, width], dtype=torch.bfloat16)

    if op == ttnn.BcastOpMath.ADD:
        torch_ref_output = a_torch + b_torch.reshape(1, batch, 1, width)
    else:
        torch_ref_output = a_torch * b_torch.reshape(1, batch, 1, width)

    input_tensor = ttnn.from_torch(
        a_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    input_2d_height = input_tensor.padded_shape[0] * input_tensor.padded_shape[1] * input_tensor.padded_shape[2]
    input_2d_width = input_tensor.padded_shape[3]
    input_2d_height_padded = _nearest_y(input_2d_height, shard_grid[0] * 32)
    shard_height = math.ceil(input_2d_height_padded / shard_grid[0])
    shard_width = math.ceil(input_2d_width / shard_grid[1])
    core_grid = (
        ttnn.CoreGrid(y=shard_grid[0], x=shard_grid[1])
        if orientation == ttnn.ShardOrientation.ROW_MAJOR
        else ttnn.CoreGrid(y=shard_grid[1], x=shard_grid[0])
    )
    in_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(
            (shard_height, shard_width)
            if orientation == ttnn.ShardOrientation.ROW_MAJOR
            else (shard_width, shard_height)
        ),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=orientation,
        use_height_and_width_as_shard_shape=True,
    )
    tt_input = ttnn.to_memory_config(input_tensor, memory_config=in_sharded_mem_config)

    tt_weight = ttnn.from_torch(b_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    tt_output = ttnn.bcast(
        tt_input,
        tt_weight,
        op,
        ttnn.BcastOpDim.H,
        memory_config=ttnn.get_memory_config(tt_input),
    )
    output_tensor = ttnn.to_torch(tt_output).float()

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_ref_output.float(), output_tensor, 0.999)
    logger.info(pcc_msg)
    assert passing, pcc_msg


@pytest.mark.parametrize(
    "a_shape, out_shape",
    (
        # oversized: the work split is input_a's tile count, so most of the caller's buffer is left stale
        ([1, 1, 32, 32], [1, 1, 64, 64]),
        # undersized: the writer emits input_a's tile count into a smaller buffer and runs off the end
        ([1, 1, 64, 64], [1, 1, 32, 32]),
        # same volume, different shape
        ([1, 1, 32, 64], [1, 1, 64, 32]),
        # page count that is not a multiple of the input's
        ([1, 1, 32, 32], [1, 1, 32, 96]),
        # dropped batch
        ([1, 2, 32, 32], [1, 1, 32, 32]),
    ),
)
@pytest.mark.parametrize("math_op", [ttnn.BcastOpMath.ADD, ttnn.BcastOpMath.MUL])
def test_bcast_preallocated_output_shape_mismatch(device, a_shape, out_shape, math_op):
    """A preallocated output_tensor whose shape is not the shape bcast produces must be rejected.

    The guard used to compare against compute_output_specs(), which early-returns the preallocated
    tensor's own spec, so it compared the shape with itself and never fired.
    """
    a = ttnn.from_torch(torch.rand(a_shape, dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    b = ttnn.from_torch(torch.rand([1, 1, 32, 32], dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    out = ttnn.from_torch(torch.rand(out_shape, dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)

    with pytest.raises(RuntimeError, match="preallocated output tensor needs a shape of"):
        ttnn.bcast(a, b, math_op, ttnn.BcastOpDim.HW, output_tensor=out)


@pytest.mark.parametrize("math_op", [ttnn.BcastOpMath.ADD, ttnn.BcastOpMath.MUL])
def test_bcast_preallocated_output_shape_match(device, math_op):
    """The matching case must still run and write the caller's buffer in place."""
    a_torch = torch.rand([1, 1, 64, 64], dtype=torch.bfloat16)
    b_torch = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    a = ttnn.from_torch(a_torch, device=device, layout=ttnn.TILE_LAYOUT)
    b = ttnn.from_torch(b_torch, device=device, layout=ttnn.TILE_LAYOUT)
    out = ttnn.from_torch(
        torch.full([1, 1, 64, 64], -99.0, dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT
    )
    out_addr = out.buffer_address()

    result = ttnn.bcast(a, b, math_op, ttnn.BcastOpDim.HW, output_tensor=out)

    assert result.buffer_address() == out_addr
    scalar = b_torch[0, 0, 0, 0].float()
    expected = a_torch.float() + scalar if math_op == ttnn.BcastOpMath.ADD else a_torch.float() * scalar
    passing, pcc_msg = check_with_pcc_without_tensor_printout(expected, ttnn.to_torch(out).float(), 0.999)
    assert passing, pcc_msg
