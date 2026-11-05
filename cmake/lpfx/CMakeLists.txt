# Build support of the LPFx copy in include/lpfx; cmake/dependencies.json records its LPFx revision.
cmake_minimum_required(VERSION 3.24)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

include(LPFxOptions)
lpfx_validate_configuration()

project(
  LPFx
  VERSION 0.1.0
  DESCRIPTION "Low-overhead Cartesian field and particle infrastructure"
  LANGUAGES CXX
)

find_package(Threads REQUIRED)

if(LPFX_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

if(LPFX_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

include(LPFxDeviceBackend)

set(LPFX_CONFIG_PROFILING_NVTX 0)
set(LPFX_CONFIG_PROFILING_ROCTX 0)
set(LPFX_CONFIG_PROFILING_ROCTX_LEGACY 0)
if(LPFX_PROFILING_BACKEND STREQUAL "NVTX")
  find_package(LPFxNVTX REQUIRED MODULE)
  set(LPFX_CONFIG_PROFILING_NVTX 1)
elseif(LPFX_PROFILING_BACKEND STREQUAL "ROCTX")
  find_package(LPFxROCTX REQUIRED MODULE)
  set(LPFX_CONFIG_PROFILING_ROCTX 1)
  if(LPFxROCTX_API STREQUAL "LEGACY")
    set(LPFX_CONFIG_PROFILING_ROCTX_LEGACY 1)
  endif()
endif()
message(STATUS "LPFx tracing backend: ${LPFX_PROFILING_BACKEND}")
if(LPFX_ENABLE_CRAYPAT)
  find_package(LPFxCrayPat REQUIRED MODULE)
endif()
message(STATUS "LPFx CrayPat recording switch: ${LPFX_ENABLE_CRAYPAT}")

# The nonblocking parent with a blocking split child (ncclCommSplit with
# splitShare) has distinct vendor minima. Use these same versions for build
# discovery, installed consumers and runtime admission.
set(LPFX_NCCL_MINIMUM_VERSION 2.18.1)
set(LPFX_RCCL_MINIMUM_VERSION 2.18.3)
set(LPFX_CONFIG_XCCL_MINIMUM_VERSION 0)
foreach(_lpfx_vendor IN ITEMS NCCL RCCL)
  if(LPFX_ENABLE_${_lpfx_vendor})
    find_package(${_lpfx_vendor} ${LPFX_${_lpfx_vendor}_MINIMUM_VERSION} REQUIRED MODULE)
    string(REPLACE "." ";" _lpfx_version "${LPFX_${_lpfx_vendor}_MINIMUM_VERSION}")
    list(GET _lpfx_version 0 _lpfx_major)
    list(GET _lpfx_version 1 _lpfx_minor)
    list(GET _lpfx_version 2 _lpfx_patch)
    math(EXPR LPFX_CONFIG_XCCL_MINIMUM_VERSION
      "${_lpfx_major} * 10000 + ${_lpfx_minor} * 100 + ${_lpfx_patch}")
  endif()
endforeach()

cmake_path(SET LPFX_INCLUDE_DIR NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
set(LPFX_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
file(MAKE_DIRECTORY "${LPFX_GENERATED_INCLUDE_DIR}/lpfx")

set(LPFX_CONFIG_DEVICE_BACKEND_CUDA 0)
set(LPFX_CONFIG_DEVICE_BACKEND_HIP 0)
if(LPFX_DEVICE_BACKEND STREQUAL "CUDA")
  set(LPFX_CONFIG_DEVICE_BACKEND_CUDA 1)
elseif(LPFX_DEVICE_BACKEND STREQUAL "HIP")
  set(LPFX_CONFIG_DEVICE_BACKEND_HIP 1)
endif()

foreach(_lpfx_boolean_option IN ITEMS
    ENABLE_CRAYPAT
    ENABLE_OPENMP
    ENABLE_DEVICE_AWARE_MPI
    ENABLE_NCCL
    ENABLE_RCCL)
  if(LPFX_${_lpfx_boolean_option})
    set(LPFX_CONFIG_${_lpfx_boolean_option} 1)
  else()
    set(LPFX_CONFIG_${_lpfx_boolean_option} 0)
  endif()
endforeach()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/lpfx_config.hpp.in"
  "${LPFX_GENERATED_INCLUDE_DIR}/lpfx/config.hpp"
  @ONLY
)

include(LPFxTargets)
lpfx_create_targets()
