if(TPIC_BUILD_DEVICE)
  enable_language(${LPFX_DEVICE_BACKEND})
  if(UNIX AND NOT APPLE AND (TPIC_BUILD_TESTS OR TPIC_VALIDATE_PINNED_TRANSFERS))
    add_library(tpic_pinned_transfer_audit OBJECT tests/support/pinned_transfer_audit.cpp)
    tpic_device_source(tpic_pinned_transfer_audit tests/support/pinned_transfer_audit.cpp)
    target_link_libraries(tpic_pinned_transfer_audit PRIVATE LPFx::device)
    string(TOLOWER "${LPFX_DEVICE_BACKEND}" _tpic_copy_runtime)
    function(tpic_enable_pinned_transfer_audit target)
      target_link_libraries(${target} PRIVATE tpic_pinned_transfer_audit)
      target_link_options(${target} PRIVATE
        "LINKER:--wrap=${_tpic_copy_runtime}MemcpyAsync"
        "LINKER:--wrap=${_tpic_copy_runtime}Memcpy2DAsync")
    endfunction()
    if(TPIC_VALIDATE_PINNED_TRANSFERS)
      tpic_enable_pinned_transfer_audit(tpic)
    endif()
  endif()
  add_library(tpic_communication INTERFACE)
  add_library(TPIC::communication ALIAS tpic_communication)
  string(TOLOWER "${TPIC_COMMUNICATION_BACKEND}" _tpic_communication_component)
  target_link_libraries(tpic_communication INTERFACE
    TPIC::model LPFx::${_tpic_communication_component})
  target_compile_definitions(tpic_communication INTERFACE
    TPIC_CONFIG_COMMUNICATION_NCCL=$<STREQUAL:${TPIC_COMMUNICATION_BACKEND},NCCL>
    TPIC_CONFIG_COMMUNICATION_RCCL=$<STREQUAL:${TPIC_COMMUNICATION_BACKEND},RCCL>)
  add_library(tpic_device_initialization INTERFACE)
  add_library(TPIC::device_initialization ALIAS tpic_device_initialization)
  target_link_libraries(tpic_device_initialization INTERFACE TPIC::initialization LPFx::device)
  # LPFx's public Cartesian coordinate factory is constexpr without device
  # annotations. Owner-flag decoding calls it with runtime GPU coordinates.
  target_compile_options(tpic_device_initialization INTERFACE
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
  # Particle startup draws from the CUDA/HIP device RNG (cuRAND/hipRAND) in a fixed sequence.
  if(LPFX_DEVICE_BACKEND STREQUAL "CUDA")
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(tpic_device_initialization INTERFACE CUDA::curand)
  else()
    if(NOT GPU_TARGETS)
      set(GPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}")
    endif()
    find_package(hiprand REQUIRED CONFIG)
    if(TARGET hip::hiprand)
      set(_tpic_hiprand_target hip::hiprand)
    elseif(TARGET roc::hiprand)
      set(_tpic_hiprand_target roc::hiprand)
    else()
      message(FATAL_ERROR "The selected hipRAND package exports no supported CMake target")
    endif()
    target_link_libraries(tpic_device_initialization INTERFACE ${_tpic_hiprand_target})
    unset(_tpic_hiprand_target)
  endif()
  target_link_libraries(tpic_ecsim INTERFACE LPFx::device)
  target_compile_options(tpic_ecsim INTERFACE
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
  add_library(tpic_simulation STATIC src/simulation/run_distributed.cpp)
  add_library(TPIC::simulation ALIAS tpic_simulation)
  tpic_device_source(tpic_simulation src/simulation/run_distributed.cpp)
  # One translation unit per compiled precision pair holds its Simulation
  # instantiations: simulation_dispatch.cpp, compiled with the pair's two structs.
  set(_tpic_pair_units ${TPIC_PRECISION_PAIR_UNITS})
  while(_tpic_pair_units)
    list(POP_FRONT _tpic_pair_units _storage_id _compute_id)
    list(GET _tpic_precision_structs ${_storage_id} _storage_struct)
    list(GET _tpic_precision_structs ${_compute_id} _compute_struct)
    set(_unit tpic_simulation_pair_${_storage_id}_${_compute_id})
    add_library(${_unit} OBJECT src/simulation/simulation_dispatch.cpp)
    tpic_device_source(${_unit} src/simulation/simulation_dispatch.cpp)
    target_compile_definitions(${_unit} PRIVATE
      TPIC_DISPATCH_STORAGE=config::Storage${_storage_struct} TPIC_DISPATCH_COMPUTE=config::Compute${_compute_struct})
    target_link_libraries(${_unit} PRIVATE TPIC::device_initialization
      TPIC::communication TPIC::ecsim TPIC::io TPIC::simulation_support)
    target_sources(tpic_simulation PRIVATE $<TARGET_OBJECTS:${_unit}>)
  endwhile()
  target_link_libraries(tpic_simulation PUBLIC TPIC::device_initialization
    TPIC::communication TPIC::ecsim TPIC::io TPIC::simulation_support)
  target_link_libraries(tpic PRIVATE TPIC::simulation)
  add_library(tpic_schema_check OBJECT tests/compile/schema_check.cpp)
  tpic_device_source(tpic_schema_check tests/compile/schema_check.cpp)
  target_link_libraries(tpic_schema_check PRIVATE
    TPIC::device_initialization TPIC::communication TPIC::ecsim)

endif()
