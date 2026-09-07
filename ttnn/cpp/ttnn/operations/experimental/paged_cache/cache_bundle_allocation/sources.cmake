# Source files for cache bundle allocation within ttnn_op_experimental_paged_cache.

list(APPEND TTNN_OP_EXPERIMENTAL_PAGED_CACHE_API_HEADERS ${CMAKE_CURRENT_LIST_DIR}/cache_bundle_allocation.hpp)
list(
    APPEND
    TTNN_OP_EXPERIMENTAL_PAGED_CACHE_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/device/update_cache_bundle_allocation_device_operation.cpp
)
list(
    APPEND
    TTNN_OP_EXPERIMENTAL_PAGED_CACHE_NANOBIND_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/cache_bundle_allocation_nanobind.cpp
)
list(
    APPEND
    TTNN_OP_EXPERIMENTAL_PAGED_CACHE_KERNELS
    ${CMAKE_CURRENT_LIST_DIR}/kernels/update_cache_bundle_allocation.cpp
)
