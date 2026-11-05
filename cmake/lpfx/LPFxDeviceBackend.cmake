include_guard(GLOBAL)

set(LPFX_DEVICE_RUNTIME_TARGET "")

if(NOT LPFX_DEVICE_BACKEND STREQUAL "NONE")
  set(_lpfx_cmake_architectures_variable
    "CMAKE_${LPFX_DEVICE_BACKEND}_ARCHITECTURES")
  set(_lpfx_cmake_architectures
    "${${_lpfx_cmake_architectures_variable}}")
  if(LPFX_DEVICE_ARCHITECTURES AND _lpfx_cmake_architectures AND
     NOT "${LPFX_DEVICE_ARCHITECTURES}" STREQUAL
         "${_lpfx_cmake_architectures}")
    message(FATAL_ERROR
      "LPFX_DEVICE_ARCHITECTURES ('${LPFX_DEVICE_ARCHITECTURES}') conflicts "
      "with ${_lpfx_cmake_architectures_variable} "
      "('${_lpfx_cmake_architectures}')")
  endif()
  if(LPFX_DEVICE_ARCHITECTURES)
    set(_lpfx_device_architectures "${LPFX_DEVICE_ARCHITECTURES}")
  elseif(_lpfx_cmake_architectures)
    set(_lpfx_device_architectures "${_lpfx_cmake_architectures}")
  else()
    message(FATAL_ERROR
      "A ${LPFX_DEVICE_BACKEND} build requires an explicit device "
      "architecture. Set LPFX_DEVICE_ARCHITECTURES or the standard "
      "${_lpfx_cmake_architectures_variable} variable.")
  endif()
  set(LPFX_DEVICE_ARCHITECTURES "${_lpfx_device_architectures}" CACHE STRING
    "Semicolon-separated architectures for the selected device backend"
    FORCE)
  set(${_lpfx_cmake_architectures_variable} "${_lpfx_device_architectures}"
    CACHE STRING "${LPFX_DEVICE_BACKEND} GPU architectures" FORCE)
endif()

if(LPFX_DEVICE_BACKEND STREQUAL "CUDA")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(LPFX_DEVICE_RUNTIME_TARGET CUDA::cudart)
  message(STATUS "LPFx CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
elseif(LPFX_DEVICE_BACKEND STREQUAL "HIP")
  # CMake's HIP language consumes CMAKE_HIP_ARCHITECTURES, while ROCm's
  # separate hip package still consumes GPU_TARGETS.  Populate the latter
  # before find_package so configure does not try to probe a GPU (which is
  # unavailable on accelerator login nodes such as LUMI's uan nodes).
  if(NOT GPU_TARGETS)
    set(GPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}")
  endif()
  enable_language(HIP)
  find_package(hip CONFIG REQUIRED)
  if(TARGET hip::host)
    set(LPFX_DEVICE_RUNTIME_TARGET hip::host)
  elseif(TARGET hip::device)
    set(LPFX_DEVICE_RUNTIME_TARGET hip::device)
  else()
    message(FATAL_ERROR
      "The selected HIP package provides neither hip::host nor hip::device")
  endif()
  message(STATUS "LPFx HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
endif()

unset(_lpfx_cmake_architectures_variable)
unset(_lpfx_cmake_architectures)
unset(_lpfx_device_architectures)
