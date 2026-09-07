// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
// Previously reached transitively through minimal_matmul_program_factory.hpp, which this header no
// longer includes; supplies ttnn::MeshCoordinate and the unqualified tensor types below.
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "minimal_matmul_device_operation_types.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct MinimalMatmulDeviceOperation {
    using operation_attributes_t = MinimalMatmulParams;
    using tensor_args_t = MinimalMatmulInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    // Descriptor-based program construction. Every per-dispatch value is a buffer address, so the
    // Buffer* bindings declared in create_descriptor would be correct on their own; the override
    // exists for dispatch cost. Bindings-only patching measured ~6.9% slower than the pre-migration
    // op on a 13x10 grid, because apply_resolved_bindings takes one
    // GetRuntimeArgs(program, kernel, core) lookup per (kernel, core) group, where the override
    // hoists five grid references and indexes them directly. With the override the descriptor path
    // is at parity (5-run mean -1.1%).
    //
    // The hook has to live on a factory struct - the adapter static-asserts against placing it on
    // the DeviceOperation - hence this single-alternative program_factory_t.
    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const MinimalMatmulParams& operation_attributes,
            const MinimalMatmulInputs& tensor_args,
            std::vector<Tensor>& tensor_return_value);

        static void override_runtime_arguments(
            tt::tt_metal::Program& program,
            const MinimalMatmulParams& operation_attributes,
            const MinimalMatmulInputs& tensor_args,
            std::vector<Tensor>& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const Tensor& weight_tensor,
        const std::optional<Tensor>& bias_tensor,
        std::optional<operations::unary::UnaryWithParam> fused_activation,
        const std::optional<const MinimalMatmulConfig>& config,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<const DataType> dtype,
        std::optional<DeviceComputeKernelConfig> compute_kernel_config,
        int32_t chunks = 1,
        int32_t dim = -1,
        std::optional<float> fused_ternary_scalar = std::nullopt,
        const std::optional<Tensor>& fused_ternary_input_a = std::nullopt,
        const std::optional<Tensor>& fused_ternary_input_b = std::nullopt,
        bool fuse_swiglu = false);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> minimal_matmul(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<Tensor>& bias_tensor,
    std::optional<operations::unary::UnaryWithParam> fused_activation,
    const std::optional<const experimental::prim::MinimalMatmulConfig>& config,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    int32_t chunks = 1,
    int32_t dim = -1,
    std::optional<float> fused_ternary_scalar = std::nullopt,
    const std::optional<Tensor>& fused_ternary_input_a = std::nullopt,
    const std::optional<Tensor>& fused_ternary_input_b = std::nullopt,
    bool fuse_swiglu = false,
    // Fused concat (concat-free): when set, in0's K is input_tensor (prefix) then optional_input_tensor
    // (suffix); the split point is input_tensor's K width and the weight is stacked [W_prefix; W_suffix].
    const std::optional<Tensor>& optional_input_tensor = std::nullopt);

}  // namespace ttnn::prim
