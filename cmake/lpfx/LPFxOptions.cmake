include_guard(GLOBAL)

set(LPFX_PROFILING_BACKEND "NONE" CACHE STRING
  "Host tracing annotations: NONE, NVTX, or ROCTX (independent of device backend)")
set_property(CACHE LPFX_PROFILING_BACKEND PROPERTY STRINGS NONE NVTX ROCTX)
option(LPFX_ENABLE_CRAYPAT
  "Compile LPFX_PROFILE_RECORD to CrayPat's PAT_record (independent of the tracing backend)"
  OFF)

option(LPFX_ENABLE_OPENMP "Build the OpenMP host-execution component" OFF)
option(LPFX_ENABLE_MPI "Build the MPI control and host-payload component" OFF)
option(LPFX_ENABLE_DEVICE_AWARE_MPI
  "Assert that MPI accepts direct pointers from the configured device backend"
  OFF)
option(LPFX_ENABLE_NCCL
  "Build the CUDA/NCCL device-payload component (MPI control plane required)"
  OFF)
option(LPFX_ENABLE_RCCL
  "Build the HIP/RCCL device-payload component (MPI control plane required)"
  OFF)
set(LPFX_DEVICE_BACKEND "NONE" CACHE STRING
  "Consumer device backend: NONE, CUDA, or HIP")
set_property(CACHE LPFX_DEVICE_BACKEND PROPERTY STRINGS NONE CUDA HIP)
set(LPFX_DEVICE_ARCHITECTURES "" CACHE STRING
  "Semicolon-separated architectures for the selected device backend")
set(LPFX_PARTICLE_CAPACITY_HEADROOM_PERCENT "20" CACHE STRING
  "Extra particle capacity percentage for initial allocation and growth (0 disables extra percentage)")
set(LPFX_PARTICLE_CAPACITY_MULTIPLE "64" CACHE STRING
  "Round particle allocation capacity to this power-of-two multiple (1 disables rounding)")
if(LPFX_ENABLE_MPI)
  set(_lpfx_test_use_mpiexec_default ON)
else()
  set(_lpfx_test_use_mpiexec_default OFF)
endif()
option(LPFX_TEST_USE_MPIEXEC
  "Launch native CTest executables through MPIEXEC_EXECUTABLE"
  ${_lpfx_test_use_mpiexec_default})
unset(_lpfx_test_use_mpiexec_default)

function(lpfx_validate_configuration)
  if(NOT LPFX_PROFILING_BACKEND MATCHES "^(NONE|NVTX|ROCTX)$")
    message(FATAL_ERROR "LPFX_PROFILING_BACKEND must be NONE, NVTX, or ROCTX; received '${LPFX_PROFILING_BACKEND}'")
  endif()
  if(NOT LPFX_PARTICLE_CAPACITY_HEADROOM_PERCENT MATCHES "^(0|[1-9][0-9]*)$")
    message(FATAL_ERROR
      "LPFX_PARTICLE_CAPACITY_HEADROOM_PERCENT must be a decimal integer from "
      "0 to 4294967295 without leading zeros; received "
      "'${LPFX_PARTICLE_CAPACITY_HEADROOM_PERCENT}'")
  endif()
  string(LENGTH "${LPFX_PARTICLE_CAPACITY_HEADROOM_PERCENT}" _lpfx_headroom_digits)
  if(_lpfx_headroom_digits GREATER 10 OR
     LPFX_PARTICLE_CAPACITY_HEADROOM_PERCENT GREATER 4294967295)
    message(FATAL_ERROR
      "LPFX_PARTICLE_CAPACITY_HEADROOM_PERCENT exceeds 4294967295; received "
      "'${LPFX_PARTICLE_CAPACITY_HEADROOM_PERCENT}'")
  endif()
  if(NOT LPFX_PARTICLE_CAPACITY_MULTIPLE MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR
      "LPFX_PARTICLE_CAPACITY_MULTIPLE must be a decimal power of two from "
      "1 to 2147483648 without leading zeros; received "
      "'${LPFX_PARTICLE_CAPACITY_MULTIPLE}'")
  endif()
  string(LENGTH "${LPFX_PARTICLE_CAPACITY_MULTIPLE}" _lpfx_multiple_digits)
  if(_lpfx_multiple_digits GREATER 10 OR
     LPFX_PARTICLE_CAPACITY_MULTIPLE GREATER 2147483648)
    message(FATAL_ERROR
      "LPFX_PARTICLE_CAPACITY_MULTIPLE exceeds 2147483648; received "
      "'${LPFX_PARTICLE_CAPACITY_MULTIPLE}'")
  endif()
  math(EXPR _lpfx_multiple_remainder
    "${LPFX_PARTICLE_CAPACITY_MULTIPLE} & (${LPFX_PARTICLE_CAPACITY_MULTIPLE} - 1)")
  if(NOT _lpfx_multiple_remainder EQUAL 0)
    message(FATAL_ERROR
      "LPFX_PARTICLE_CAPACITY_MULTIPLE must be a power of two; received "
      "'${LPFX_PARTICLE_CAPACITY_MULTIPLE}'")
  endif()
  if(NOT LPFX_DEVICE_BACKEND STREQUAL "NONE"
     AND NOT LPFX_DEVICE_BACKEND STREQUAL "CUDA"
     AND NOT LPFX_DEVICE_BACKEND STREQUAL "HIP")
    message(FATAL_ERROR
      "LPFX_DEVICE_BACKEND must be exactly NONE, CUDA, or HIP; received "
      "'${LPFX_DEVICE_BACKEND}'")
  endif()
  if(LPFX_ENABLE_DEVICE_AWARE_MPI AND NOT LPFX_ENABLE_MPI)
    message(FATAL_ERROR
      "LPFX_ENABLE_DEVICE_AWARE_MPI requires LPFX_ENABLE_MPI=ON")
  endif()
  if(LPFX_ENABLE_DEVICE_AWARE_MPI AND LPFX_DEVICE_BACKEND STREQUAL "NONE")
    message(FATAL_ERROR
      "LPFX_ENABLE_DEVICE_AWARE_MPI requires a CUDA or HIP device backend")
  endif()
  if(LPFX_ENABLE_NCCL AND LPFX_ENABLE_RCCL)
    message(FATAL_ERROR
      "LPFX_ENABLE_NCCL and LPFX_ENABLE_RCCL are mutually exclusive because "
      "one LPFx build selects exactly one device backend")
  endif()
  if(LPFX_ENABLE_NCCL AND NOT LPFX_ENABLE_MPI)
    message(FATAL_ERROR "LPFX_ENABLE_NCCL requires LPFX_ENABLE_MPI=ON")
  endif()
  if(LPFX_ENABLE_NCCL AND NOT LPFX_DEVICE_BACKEND STREQUAL "CUDA")
    message(FATAL_ERROR
      "LPFX_ENABLE_NCCL requires LPFX_DEVICE_BACKEND=CUDA; received "
      "'${LPFX_DEVICE_BACKEND}'")
  endif()
  if(LPFX_ENABLE_RCCL AND NOT LPFX_ENABLE_MPI)
    message(FATAL_ERROR "LPFX_ENABLE_RCCL requires LPFX_ENABLE_MPI=ON")
  endif()
  if(LPFX_ENABLE_RCCL AND NOT LPFX_DEVICE_BACKEND STREQUAL "HIP")
    message(FATAL_ERROR
      "LPFX_ENABLE_RCCL requires LPFX_DEVICE_BACKEND=HIP; received "
      "'${LPFX_DEVICE_BACKEND}'")
  endif()
endfunction()
