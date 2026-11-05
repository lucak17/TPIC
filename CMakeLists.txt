cmake_minimum_required(VERSION 3.24)
project(TPIC VERSION 0.1.0 LANGUAGES CXX)

include(cmake/TpicOptions.cmake)

add_subdirectory(cmake/lpfx "${CMAKE_CURRENT_BINARY_DIR}/lpfx")

add_library(tpic_model INTERFACE)
add_library(TPIC::model ALIAS tpic_model)
target_include_directories(tpic_model INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(tpic_model INTERFACE cxx_std_20)
target_compile_definitions(tpic_model INTERFACE
  TPIC_CONFIG_MATRIX_MASS=$<BOOL:${TPIC_BUILD_DEVICE}>
  TPIC_CONFIG_MAX_SPECIES=${TPIC_MAX_SPECIES}
  TPIC_CONFIG_PARTICLE_MIGRATION_HOPS=${TPIC_PARTICLE_MIGRATION_HOPS}
  TPIC_HAS_ADIOS2=$<BOOL:${TPIC_ENABLE_ADIOS2}>
  TPIC_HAS_HDF5=$<BOOL:${TPIC_ENABLE_HDF5}>
  TPIC_DEFAULT_STORAGE_PRECISION_ID=${TPIC_DEFAULT_STORAGE_PRECISION_ID}
  TPIC_DEFAULT_COMPUTE_PRECISION_ID=${TPIC_DEFAULT_COMPUTE_PRECISION_ID}
  TPIC_PRECISION_PAIR_MASK=${TPIC_PRECISION_PAIR_MASK})
target_link_libraries(tpic_model INTERFACE LPFx::core LPFx::profiling)

add_library(tpic_config STATIC
  src/config/simulation_parameters.cpp src/config/domain.cpp src/config/physics.cpp
  src/config/particle_plan.cpp src/config/output.cpp src/config/output_selection.cpp)
add_library(TPIC::config ALIAS tpic_config)
target_link_libraries(tpic_config PUBLIC TPIC::model)

add_library(tpic_initialization STATIC src/initialization/initial_case.cpp
  src/initialization/prepared_initialization.cpp src/initialization/cases/harris_sheet.cpp
  src/initialization/cases/force_free.cpp src/initialization/cases/weibel.cpp
  src/initialization/cases/whistler.cpp src/initialization/cases/taylor_green_vortex.cpp
  src/initialization/cases/two_stream.cpp src/initialization/cases/flux_rope.cpp
  src/initialization/cases/gaussian_expansion.cpp src/initialization/cases/whistler_turbulence.cpp)
add_library(TPIC::initialization ALIAS tpic_initialization)
target_link_libraries(tpic_initialization PUBLIC TPIC::config)

# Public module targets reflect ownership even when numerical implementation is
# header-only. Host policy and output can be tested without enabling a GPU.
add_library(tpic_ecsim INTERFACE)
add_library(TPIC::ecsim ALIAS tpic_ecsim)
target_link_libraries(tpic_ecsim INTERFACE TPIC::model)

add_library(tpic_io STATIC src/io/diagnostics_writer.cpp src/io/vtk_field_engine.cpp
  src/io/async_output_writer.cpp src/io/output_engine.cpp src/io/vtk_particle_engine.cpp
  src/io/checkpoint.cpp)
add_library(TPIC::io ALIAS tpic_io)
target_link_libraries(tpic_io PUBLIC TPIC::config PRIVATE LPFx::mpi)
if(TPIC_ENABLE_ADIOS2 OR TPIC_ENABLE_HDF5)
  target_sources(tpic_io PRIVATE src/io/dataset_engine.cpp)
endif()
if(TPIC_ENABLE_ADIOS2)
  # An explicit CXX component avoids selecting MPI in an MPI-enabled ADIOS2
  # installation: the ADIOS2 engines write rank-local files.
  find_package(adios2 2.10 CONFIG REQUIRED COMPONENTS CXX)
  target_sources(tpic_io PRIVATE src/io/adios2_engine.cpp)
  target_link_libraries(tpic_io PRIVATE adios2::cxx)
endif()
if(TPIC_ENABLE_HDF5)
  enable_language(C)
  find_package(HDF5 REQUIRED COMPONENTS C)
  target_sources(tpic_io PRIVATE src/io/hdf5_engine.cpp)
  target_link_libraries(tpic_io PRIVATE HDF5::HDF5)
endif()

add_library(tpic_simulation_support STATIC src/simulation/run_plan.cpp
  src/simulation/run_reporter.cpp src/simulation/startup_report.cpp
  src/simulation/timing_regions.cpp src/simulation/moment_submission_error.cpp)
add_library(TPIC::simulation_support ALIAS tpic_simulation_support)
target_link_libraries(tpic_simulation_support PUBLIC TPIC::initialization TPIC::io)

function(tpic_device_source target source)
  set_source_files_properties(${source} PROPERTIES LANGUAGE ${LPFX_DEVICE_BACKEND})
  set_target_properties(${target} PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES
    HIP_STANDARD 20 HIP_STANDARD_REQUIRED YES)
endfunction()

if(TPIC_BUILD_TESTS)
  enable_testing()
endif()

# Input checking stays independent of MPI/GPU initialization in every build.
add_executable(tpic src/app/main.cpp)
target_link_libraries(tpic PRIVATE TPIC::simulation_support)
target_compile_definitions(tpic PRIVATE TPIC_HAS_DEVICE_STARTUP=$<BOOL:${TPIC_BUILD_DEVICE}>)

include(cmake/TpicDevice.cmake)

if(TPIC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
