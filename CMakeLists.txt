cmake_minimum_required(VERSION 3.20)

# Minimum macOS version. Must precede project(): project() creates this cache
# entry (empty, i.e. the host SDK version), and a later set(... CACHE) does not
# overwrite it, so the dylib declared minos = host OS (27.0) instead of 13.0.
# 15.0, not 13.0: a 13.0 build warned on 272 unguarded uses of newer APIs
# (reinterpretCastTensor 13.2, bf16/FFT/complex parts 14.0, sliceUpdateDataTensor
# and reciprocalSquareRoot 15.0) that would raise unrecognized-selector at runtime.
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum macOS version")

project(jax_metallib LANGUAGES C CXX OBJCXX)

# Dependency install prefix (used by scripts/setup_deps.sh)
set(JAX_METALLIB_DEPS_PREFIX
    "$ENV{HOME}/.local/jax-metallib-deps"
    CACHE PATH "Install prefix for jax-metallib native dependencies")

# Auto-bootstrap dependencies when missing. Disable with:
#   CMAKE_ARGS="-DJAX_METALLIB_AUTO_SETUP_DEPS=OFF"
option(
    JAX_METALLIB_AUTO_SETUP_DEPS
    "Automatically run scripts/setup_deps.sh if dependencies are missing"
    ON
)

# Search this prefix first for MLIR/StableHLO/XLA/Protobuf/Abseil
list(APPEND CMAKE_PREFIX_PATH "${JAX_METALLIB_DEPS_PREFIX}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# Export compile commands for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use Accelerate's modern LAPACK entry points. Required: linalg_ops.mm uses
# __LAPACK_int / spotrf_ from the new interface (CLAPACK is deprecated).
if(APPLE)
    add_compile_options(-DACCELERATE_NEW_LAPACK)
endif()

# Explicitly set sysroot so compile_commands.json works with any clang-tidy
execute_process(
    COMMAND xcrun --show-sdk-path
    OUTPUT_VARIABLE MACOS_SDK_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_compile_options(-isysroot ${MACOS_SDK_PATH})

# Find required Apple frameworks
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

# Bootstrap native dependencies once if they are not present.
set(_XLA_HEADER_NAME "xla/pjrt/c/pjrt_c_api.h")
find_path(XLA_INCLUDE_DIR
    NAMES ${_XLA_HEADER_NAME}
    HINTS ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES include
)
if(NOT XLA_INCLUDE_DIR AND JAX_METALLIB_AUTO_SETUP_DEPS)
    set(_SETUP_DEPS_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/setup_deps.sh")
    if(NOT EXISTS "${_SETUP_DEPS_SCRIPT}")
        message(FATAL_ERROR "Dependency setup script not found: ${_SETUP_DEPS_SCRIPT}")
    endif()

    message(STATUS "XLA PJRT headers not found. Running ${_SETUP_DEPS_SCRIPT}...")
    execute_process(
        COMMAND /bin/bash "${_SETUP_DEPS_SCRIPT}" --prefix "${JAX_METALLIB_DEPS_PREFIX}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMAND_ECHO STDOUT
        RESULT_VARIABLE SETUP_DEPS_RESULT
    )
    if(NOT SETUP_DEPS_RESULT EQUAL 0)
        message(
            FATAL_ERROR
            "Automatic dependency setup failed with exit code ${SETUP_DEPS_RESULT}. "
            "Run scripts/setup_deps.sh manually and try again."
        )
    endif()

    # Retry after bootstrap.
    find_path(XLA_INCLUDE_DIR
        NAMES ${_XLA_HEADER_NAME}
        HINTS ${CMAKE_PREFIX_PATH}
        PATH_SUFFIXES include
    )
endif()

# XLA PJRT headers (installed by setup_deps.sh)
if(XLA_INCLUDE_DIR)
    message(STATUS "Found XLA headers: ${XLA_INCLUDE_DIR}")
else()
    message(
        FATAL_ERROR
        "XLA PJRT headers not found in ${CMAKE_PREFIX_PATH}. "
        "Run scripts/setup_deps.sh or configure with "
        "-DJAX_METALLIB_AUTO_SETUP_DEPS=ON."
    )
endif()

# Find Protobuf and Abseil (static builds from setup_deps.sh for wheel distribution)
# Prefer static libraries to avoid runtime dependencies
set(CMAKE_FIND_LIBRARY_SUFFIXES_ORIG ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

find_package(absl REQUIRED)

# Find protoc executable from our prefix path before finding Protobuf package
find_program(Protobuf_PROTOC_EXECUTABLE protoc
    HINTS ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES bin
)
find_package(Protobuf REQUIRED)
message(STATUS "Found Protobuf: ${Protobuf_VERSION}")
message(STATUS "Found protoc: ${Protobuf_PROTOC_EXECUTABLE}")

set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_ORIG})

# Generate C++ from proto files
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS src/pjrt_plugin/proto/device_assignment.proto)

# Find MLIR and LLVM packages
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found MLIR: ${MLIR_DIR}")
message(STATUS "Found LLVM: ${LLVM_DIR}")

# Include MLIR/LLVM cmake modules
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)

# Find StableHLO headers
find_path(STABLEHLO_INCLUDE_DIR
    NAMES stablehlo/dialect/StablehloOps.h
    HINTS ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES include
)
if(STABLEHLO_INCLUDE_DIR)
    message(STATUS "Found StableHLO headers: ${STABLEHLO_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "StableHLO headers not found. Run scripts/setup_deps.sh first.")
endif()

# Find StableHLO static libraries
find_library(STABLEHLO_OPS_LIB StablehloOps HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(STABLEHLO_SERIALIZATION_LIB StablehloSerialization HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(VHLO_OPS_LIB VhloOps HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(CHLO_OPS_LIB ChloOps HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(STABLEHLO_BASE_LIB StablehloBase HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(STABLEHLO_REGISTER_LIB StablehloRegister HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(VHLO_TYPES_LIB VhloTypes HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(STABLEHLO_TYPE_INFERENCE_LIB StablehloTypeInference HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(STABLEHLO_ASSEMBLY_FORMAT_LIB StablehloAssemblyFormat HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(STABLEHLO_BROADCAST_UTILS_LIB StablehloBroadcastUtils HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(STABLEHLO_BROADCAST_LOWERING_LIB StablehloBroadcastLowering HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(STABLEHLO_PASSES_LIB StablehloPasses HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(STABLEHLO_REPLICA_GROUP_UTILS_LIB StablehloReplicaGroupUtils HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(STABLEHLO_OPTIMIZATION_PASSES_LIB StablehloOptimizationPasses HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(STABLEHLO_PASS_UTILS_LIB StablehloPassUtils HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(STABLEHLO_TYPE_CONVERSION_LIB StablehloTypeConversion HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(VERSION_LIB Version HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)

# Source files
set(PJRT_SOURCES
    # API layer (PJRT C API)
    src/pjrt_plugin/api/pjrt_api.cc
    src/pjrt_plugin/api/pjrt_client.cc
    src/pjrt_plugin/api/pjrt_device.cc
    src/pjrt_plugin/api/pjrt_memory.cc
    src/pjrt_plugin/api/pjrt_buffer.cc
    src/pjrt_plugin/api/pjrt_executable.cc
    src/pjrt_plugin/api/pjrt_event.cc
    src/pjrt_plugin/api/pjrt_topology.cc
    # Core (MPS backend)
    src/pjrt_plugin/core/mps_client.mm
    src/pjrt_plugin/core/mps_device.mm
    src/pjrt_plugin/core/mps_buffer.mm
    src/pjrt_plugin/core/mps_executable.mm
    src/pjrt_plugin/core/stablehlo_parser.mm
    src/pjrt_plugin/core/type_utils.mm
    # Runtime (Metal kernels & native execution)
    src/pjrt_plugin/runtime/metal_kernels.mm
    # Operations
    src/pjrt_plugin/ops/binary_ops.mm
    src/pjrt_plugin/ops/unary_ops.mm
    src/pjrt_plugin/ops/shape_ops.mm
    src/pjrt_plugin/ops/bitwise_ops.mm
    src/pjrt_plugin/ops/fft_ops.mm
    src/pjrt_plugin/ops/tensor_creation_ops.mm
    src/pjrt_plugin/ops/random_ops.mm
    src/pjrt_plugin/ops/convolution_ops.mm
    src/pjrt_plugin/ops/reduction_ops.mm
    src/pjrt_plugin/ops/control_flow_ops.mm
    src/pjrt_plugin/ops/collective_ops.mm
    src/pjrt_plugin/ops/higher_order_ops.mm
    src/pjrt_plugin/ops/sort_ops.mm
    src/pjrt_plugin/ops/linalg_ops.mm
    ${PROTO_SRCS}
)

add_library(pjrt_plugin_silicon SHARED ${PJRT_SOURCES})

# Link-time optimization (experimental; measurable on dispatch-heavy paths).
if(JAX_METALLIB_ENABLE_LTO)
    cmake_policy(SET CMP0069 NEW)
    set_property(TARGET pjrt_plugin_silicon PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "LTO enabled for pjrt_plugin_silicon")
endif()

target_include_directories(pjrt_plugin_silicon PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${XLA_INCLUDE_DIR}
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${STABLEHLO_INCLUDE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}  # For generated proto headers
    ${Protobuf_INCLUDE_DIRS}
)

# Debug logging level (0=error, 1=warn, 2=info, 3=debug)
if(DEFINED ENV{MPS_LOG_LEVEL})
    target_compile_definitions(pjrt_plugin_silicon PRIVATE MPS_LOG_LEVEL=$ENV{MPS_LOG_LEVEL})
endif()

# Core MLIR libraries (order matters for static linking)
set(MLIR_LIBS
    MLIRReconcileUnrealizedCasts
    MLIRPass
    MLIRTransforms
    MLIRFuncInlinerExtension
    MLIRFuncDialect
    MLIRFuncTransforms
    MLIRArithDialect
    MLIRComplexDialect
    MLIRQuantDialect
    MLIRShapeDialect
    MLIRTensorDialect
    MLIRDataLayoutInterfaces
    MLIRInferTypeOpInterface
    MLIRSideEffectInterfaces
    MLIRTransformUtils
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRBytecodeReader
    MLIRBytecodeWriter
    MLIRSupport
    LLVMSupport
)

# Collect found StableHLO libraries
set(STABLEHLO_LIBS "")
foreach(lib
    STABLEHLO_SERIALIZATION_LIB
    STABLEHLO_PASSES_LIB
    STABLEHLO_REPLICA_GROUP_UTILS_LIB
    STABLEHLO_OPTIMIZATION_PASSES_LIB
    STABLEHLO_PASS_UTILS_LIB
    STABLEHLO_TYPE_CONVERSION_LIB
    STABLEHLO_OPS_LIB
    VHLO_OPS_LIB
    CHLO_OPS_LIB
    STABLEHLO_BASE_LIB
    STABLEHLO_REGISTER_LIB
    VHLO_TYPES_LIB
    STABLEHLO_TYPE_INFERENCE_LIB
    STABLEHLO_ASSEMBLY_FORMAT_LIB
    STABLEHLO_BROADCAST_UTILS_LIB
    STABLEHLO_BROADCAST_LOWERING_LIB
    VERSION_LIB)
    if(${lib})
        list(APPEND STABLEHLO_LIBS ${${lib}})
    endif()
endforeach()

message(STATUS "StableHLO libraries: ${STABLEHLO_LIBS}")

# Abseil libraries needed by protobuf and our code (for static linking)
set(ABSL_LIBS
    absl::log
    absl::log_internal_check_op
    absl::log_internal_message
    # Required by protobuf when statically linked
    absl::cord
    absl::strings
    absl::str_format
    absl::synchronization
    absl::hash
    absl::raw_hash_set
    absl::status
    absl::statusor
    absl::span
    absl::optional
    absl::variant
    absl::any
    absl::flat_hash_map
    absl::flat_hash_set
    absl::inlined_vector
)

# Find utf8_range libraries (required by protobuf)
find_library(UTF8_RANGE_LIB utf8_range HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
find_library(UTF8_VALIDITY_LIB utf8_validity HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
set(UTF8_LIBS "")
if(UTF8_RANGE_LIB)
    list(APPEND UTF8_LIBS ${UTF8_RANGE_LIB})
endif()
if(UTF8_VALIDITY_LIB)
    list(APPEND UTF8_LIBS ${UTF8_VALIDITY_LIB})
endif()

find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

target_link_libraries(pjrt_plugin_silicon PRIVATE
    ${METAL_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${MPSGRAPH_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${ACCELERATE_FRAMEWORK}
    ${STABLEHLO_LIBS}
    ${MLIR_LIBS}
    protobuf::libprotobuf
    ${UTF8_LIBS}
    ${ABSL_LIBS}
)

# For wheel distribution: ensure RPATH is set correctly
set_target_properties(pjrt_plugin_silicon PROPERTIES
    OUTPUT_NAME "pjrt_plugin_silicon"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "@loader_path"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Install target - this is used by scikit-build-core
# Note: scikit-build's wheel.install-dir adds "jax_plugins/silicon" prefix,
# so we only need "lib" here to get "jax_plugins/silicon/lib" in the wheel
install(TARGETS pjrt_plugin_silicon
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
)
