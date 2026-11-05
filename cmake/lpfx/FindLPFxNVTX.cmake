# NVTX3 is header-only and does not require the CUDA language/runtime.
find_path(LPFxNVTX_INCLUDE_DIR NAMES nvtx3/nvToolsExt.h
  HINTS ${CUDAToolkit_INCLUDE_DIRS} ENV NVTX3_ROOT ENV CUDAToolkit_ROOT
    ENV CUDA_PATH ENV CUDA_HOME
  PATHS /usr/local/cuda /opt/cuda
  PATH_SUFFIXES include)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LPFxNVTX REQUIRED_VARS LPFxNVTX_INCLUDE_DIR)
if(LPFxNVTX_FOUND AND NOT TARGET LPFxNVTX::NVTX)
  add_library(LPFxNVTX::NVTX INTERFACE IMPORTED)
  set_target_properties(LPFxNVTX::NVTX PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${LPFxNVTX_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${CMAKE_DL_LIBS}")
endif()
mark_as_advanced(LPFxNVTX_INCLUDE_DIR)
