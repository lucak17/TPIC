include_guard(GLOBAL)

function(_lpfx_add_interface_component target export_name)
  add_library(${target} INTERFACE)
  set_target_properties(${target} PROPERTIES EXPORT_NAME "${export_name}")
  target_compile_features(${target} INTERFACE cxx_std_20)
endfunction()

function(lpfx_create_targets)
  _lpfx_add_interface_component(lpfx_core core)
  add_library(LPFx::core ALIAS lpfx_core)
  target_link_libraries(lpfx_core INTERFACE Threads::Threads)
  target_include_directories(lpfx_core INTERFACE
    $<BUILD_INTERFACE:${LPFX_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${LPFX_GENERATED_INCLUDE_DIR}>
  )

  _lpfx_add_interface_component(lpfx_profiling profiling)
  add_library(LPFx::profiling ALIAS lpfx_profiling)
  target_link_libraries(lpfx_profiling INTERFACE lpfx_core)
  if(LPFX_PROFILING_BACKEND STREQUAL "NVTX")
    target_link_libraries(lpfx_profiling INTERFACE LPFxNVTX::NVTX)
  elseif(LPFX_PROFILING_BACKEND STREQUAL "ROCTX")
    target_link_libraries(lpfx_profiling INTERFACE LPFxROCTX::ROCTX)
  endif()
  if(LPFX_ENABLE_CRAYPAT)
    target_link_libraries(lpfx_profiling INTERFACE LPFxCrayPat::API)
  endif()

  if(LPFX_ENABLE_OPENMP)
    _lpfx_add_interface_component(lpfx_openmp openmp)
    add_library(LPFx::openmp ALIAS lpfx_openmp)
    target_link_libraries(lpfx_openmp INTERFACE lpfx_core OpenMP::OpenMP_CXX)
  endif()

  if(LPFX_ENABLE_MPI)
    _lpfx_add_interface_component(lpfx_mpi mpi)
    add_library(LPFx::mpi ALIAS lpfx_mpi)
    target_link_libraries(lpfx_mpi INTERFACE lpfx_core MPI::MPI_CXX)
  endif()

  if(NOT LPFX_DEVICE_BACKEND STREQUAL "NONE")
    _lpfx_add_interface_component(lpfx_device device)
    add_library(LPFx::device ALIAS lpfx_device)
    target_link_libraries(
      lpfx_device INTERFACE lpfx_core ${LPFX_DEVICE_RUNTIME_TARGET})
    if(LPFX_DEVICE_BACKEND STREQUAL "CUDA")
      target_compile_features(lpfx_device INTERFACE cuda_std_20)
    elseif(LPFX_DEVICE_BACKEND STREQUAL "HIP")
      target_compile_options(lpfx_device INTERFACE
        $<$<COMPILE_LANGUAGE:HIP>:-std=c++20>)
      # Expose HIP's synchronized collectives on 6.2 and its additional native
      # reduction overloads on 7.x. Header-side version gates retain the
      # portable fallback for older toolchains.
      target_compile_definitions(lpfx_device INTERFACE
        HIP_ENABLE_WARP_SYNC_BUILTINS
        HIP_ENABLE_EXTRA_WARP_SYNC_TYPES)
    endif()
  endif()

  if(LPFX_ENABLE_NCCL)
    _lpfx_add_interface_component(lpfx_nccl nccl)
    add_library(LPFx::nccl ALIAS lpfx_nccl)
    target_compile_definitions(
      lpfx_nccl INTERFACE LPFX_DISTRIBUTED_ENABLE_NCCL_BACKEND=1)
    target_link_libraries(
      lpfx_nccl INTERFACE lpfx_mpi lpfx_device NCCL::NCCL)
  endif()

  if(LPFX_ENABLE_RCCL)
    _lpfx_add_interface_component(lpfx_rccl rccl)
    add_library(LPFx::rccl ALIAS lpfx_rccl)
    target_compile_definitions(
      lpfx_rccl INTERFACE LPFX_DISTRIBUTED_ENABLE_RCCL_BACKEND=1)
    target_link_libraries(
      lpfx_rccl INTERFACE lpfx_mpi lpfx_device RCCL::RCCL)
  endif()

endfunction()
