include(${PROJECT_SOURCE_DIR}/cmake/TpicTests.cmake)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

tpic_add_test_case(SUITE tpic_documentation NAME documentation
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/compile/documentation.py")
tpic_add_test_case(SUITE tpic_architecture NAME module_dependencies
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/compile/module_dependencies.py")
tpic_add_test_case(SUITE tpic_architecture NAME foreign_references
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/compile/foreign_references.py")
foreach(check IN ITEMS test_runs test_suite_runner test_whistler_turbulence)
  tpic_add_test_case(SUITE tpic_tools NAME ${check}
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/support/${check}.py")
endforeach()

# LPFx mock backend is an explicit test-only dependency.
add_library(tpic_test_support INTERFACE)
target_include_directories(tpic_test_support INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/support")

# Golden tests compare with tables in tests/data/golden/ of the source tree;
# started with --print they write their table instead.
function(tpic_golden_test suite name source)
  add_executable(tpic_test_${name} ${source})
  target_link_libraries(tpic_test_${name} PRIVATE ${ARGN})
  tpic_add_native_case(SUITE ${suite} NAME ${name} TARGET tpic_test_${name}
    ARGUMENTS "${PROJECT_SOURCE_DIR}")
endfunction()

foreach(test IN ITEMS simulation_parameters domain physics particle_plan output)
  tpic_host_test(tpic_config ${test} config/${test}_tests.cpp TPIC::config)
endforeach()
tpic_golden_test(tpic_config output_selection config/output_selection_tests.cpp TPIC::config)
tpic_host_test(tpic_config precision config/precision_tests.cpp TPIC::simulation_support LPFx::mpi)
tpic_golden_test(tpic_config resolved_parameters config/resolved_parameters_tests.cpp TPIC::config)

tpic_host_test(tpic_initialization recipe initialization/recipe_tests.cpp TPIC::initialization)
tpic_golden_test(tpic_initialization field_profile initialization/field_profile_tests.cpp
  TPIC::simulation_support)
tpic_host_test(tpic_initialization field_image initialization/field_image_tests.cpp TPIC::simulation_support)
tpic_host_test(tpic_initialization field_upload initialization/field_upload_tests.cpp
  tpic_test_support TPIC::simulation_support)

foreach(test IN ITEMS mass_mfma_layout stencil_region cell_to_node cic)
  tpic_host_test(tpic_numerics ${test} ecsim/${test}_tests.cpp TPIC::ecsim)
endforeach()
tpic_host_test(tpic_communication field_exchange communication/field_exchange_tests.cpp TPIC::model)
tpic_host_test(tpic_communication field_halo communication/field_halo_tests.cpp
  tpic_test_support TPIC::simulation_support)
tpic_host_test(tpic_communication mass_halo_plan communication/mass_halo_plan_tests.cpp TPIC::simulation_support)

tpic_host_test(tpic_output checkpoint io/checkpoint_tests.cpp TPIC::io)
tpic_host_test(tpic_output async_output_writer io/async_output_writer_tests.cpp TPIC::io LPFx::mpi)
# Every rank writes its nodes of the shared field files.
tpic_host_test(tpic_output vtk_field_output io/vtk_field_output_tests.cpp TPIC::io LPFx::mpi)
tpic_add_native_case(SUITE tpic_output NAME vtk_field_output_3 TARGET tpic_test_vtk_field_output RANKS 3)
if(TPIC_ENABLE_ADIOS2 OR TPIC_ENABLE_HDF5)
  tpic_host_test(tpic_output dataset_engine io/dataset_engine_tests.cpp TPIC::io LPFx::mpi)
  if(TPIC_ENABLE_ADIOS2)
    target_link_libraries(tpic_test_dataset_engine PRIVATE adios2::cxx)
    find_program(TPIC_PVPYTHON_EXECUTABLE NAMES pvpython)
    if(TPIC_PVPYTHON_EXECUTABLE)
      tpic_add_test_case(SUITE tpic_output NAME adios_paraview_reader SKIP_RETURN_CODE 77
        COMMAND "${TPIC_PVPYTHON_EXECUTABLE}" --disable-registry --force-offscreen-rendering
          "${CMAKE_CURRENT_SOURCE_DIR}/io/test_adios_paraview_reader.py"
          $<TARGET_FILE:tpic_test_dataset_engine>)
    endif()
  endif()
  if(TPIC_ENABLE_HDF5)
    target_link_libraries(tpic_test_dataset_engine PRIVATE HDF5::HDF5)
  endif()
endif()
add_executable(tpic_write_field_fixture io/write_field_fixture.cpp)
target_link_libraries(tpic_write_field_fixture PRIVATE TPIC::io LPFx::mpi)
# The reader test launches the fixture like a rank-one native case.
set(_tpic_field_fixture $<TARGET_FILE:tpic_write_field_fixture>)
if(LPFX_TEST_USE_MPIEXEC)
  set(_tpic_field_fixture ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
    ${_tpic_field_fixture} ${MPIEXEC_POSTFLAGS})
endif()
tpic_add_test_case(SUITE tpic_output NAME field_reader
  ENVIRONMENT "MPLCONFIGDIR=${CMAKE_CURRENT_BINARY_DIR}/matplotlib"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/io/test_field_reader.py"
    ${_tpic_field_fixture})

foreach(test IN ITEMS execution_streams deposition_ledger)
  tpic_host_test(tpic_simulation ${test} simulation/${test}_tests.cpp TPIC::model)
endforeach()
# The timing tables print through src/simulation/timing_regions.cpp.
tpic_host_test(tpic_simulation stage_timing simulation/stage_timing_tests.cpp TPIC::simulation_support)
tpic_host_test(tpic_simulation pipeline simulation/pipeline_tests.cpp TPIC::simulation_support)
tpic_host_test(tpic_simulation device_validation device/device_validation_tests.cpp
  tpic_test_support TPIC::model)
tpic_host_test(tpic_simulation magnetic_startup simulation/magnetic_startup_tests.cpp
  TPIC::simulation_support)
tpic_host_test(tpic_simulation memory_estimator simulation/memory_estimator_tests.cpp
  tpic_test_support TPIC::simulation_support)
foreach(test IN ITEMS admission prepared_input run_plan run_options moment_submission_error
                 prepared_plans kernel_records)
  tpic_host_test(tpic_simulation ${test} simulation/${test}_tests.cpp TPIC::simulation_support)
endforeach()
tpic_golden_test(tpic_simulation input_decks simulation/input_decks.cpp TPIC::simulation_support)
tpic_add_test_case(SUITE tpic_cli NAME check_input
  COMMAND "${CMAKE_COMMAND}" -DTPIC_EXECUTABLE=$<TARGET_FILE:tpic>
    -DTPIC_MATRIX_MASS=$<BOOL:${TPIC_BUILD_DEVICE}>
    -DTPIC_ADIOS2=$<BOOL:${TPIC_ENABLE_ADIOS2}> -DTPIC_HDF5=$<BOOL:${TPIC_ENABLE_HDF5}>
    -DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/check-input-fixture
    -P "${CMAKE_CURRENT_SOURCE_DIR}/config/check_input.cmake")

if(TPIC_BUILD_DEVICE)
  foreach(test IN ITEMS particle_loading particle_centering field_upload)
    tpic_device_test(tpic_device_initialization ${test} initialization/${test}_device.cpp
      TPIC::device_initialization TPIC::communication TPIC::ecsim TPIC::simulation_support)
  endforeach()
  tpic_device_test(tpic_device_runtime magnetic_startup simulation/magnetic_startup_device.cpp
    TPIC::device_initialization TPIC::communication TPIC::ecsim TPIC::simulation_support)
  foreach(test IN ITEMS scalar_diagnostics async_field_snapshot)
    tpic_device_test(tpic_device_diagnostics ${test} diagnostics/${test}_device.cpp
      TPIC::ecsim TPIC::simulation_support)
  endforeach()
  tpic_device_test(tpic_device_diagnostics async_particle_snapshot diagnostics/async_particle_snapshot_device.cpp
    TPIC::device_initialization TPIC::communication TPIC::ecsim TPIC::simulation_support)
  foreach(test IN ITEMS sorted_mass sorted_current_mover gmres maxwell_overlap mass_apply particle_sort)
    tpic_device_test(tpic_device_numerics ${test} ecsim/${test}_device.cpp
      TPIC::ecsim TPIC::simulation_support)
  endforeach()
  foreach(test IN ITEMS particle_migration mass_halo)
    tpic_device_test(tpic_device_communication ${test} communication/${test}_device.cpp
      TPIC::device_initialization TPIC::communication TPIC::ecsim TPIC::simulation_support)
  endforeach()
  tpic_device_test(tpic_device_runtime stage_timing simulation/stage_timing_device.cpp
    TPIC::ecsim TPIC::simulation_support)
  tpic_device_test(tpic_device_runtime execution_streams simulation/execution_streams_device.cpp TPIC::ecsim)
  # The rejection fixture compiles device templates but exercises host checks
  # without requiring a GPU, so it belongs in the host simulation suite.
  tpic_host_test(tpic_simulation device_submission_rejection ecsim/device_submission_rejection.cpp
    TPIC::ecsim TPIC::simulation_support)
  tpic_device_source(tpic_test_device_submission_rejection ecsim/device_submission_rejection.cpp)
  if(UNIX AND NOT APPLE)
    foreach(test IN ITEMS particle_loading particle_centering scalar_diagnostics)
      tpic_enable_pinned_transfer_audit(tpic_test_${test}_device)
    endforeach()
  endif()
endif()

if(TPIC_TEST_INTEGRATION)
  if(NOT MPIEXEC_NUMPROC_FLAG)
    message(FATAL_ERROR "TPIC integration tests require the MPI process-count flag")
  endif()
  separate_arguments(_tpic_integration_launcher NATIVE_COMMAND "${TPIC_INTEGRATION_LAUNCHER}")
  if(NOT _tpic_integration_launcher OR NOT TPIC_INTEGRATION_RANKS)
    message(FATAL_ERROR "TPIC integration tests require a launcher and process counts")
  endif()
  foreach(ranks IN LISTS TPIC_INTEGRATION_RANKS)
    if(NOT ranks MATCHES "^[12]$")
      message(FATAL_ERROR "TPIC_INTEGRATION_RANKS supports the existing one-/two-rank fixtures")
    endif()
  endforeach()
  # One-rank migration and halo checks run in tpic_device_communication.
  if(2 IN_LIST TPIC_INTEGRATION_RANKS)
    foreach(test IN ITEMS particle_migration mass_halo)
      tpic_add_native_case(SUITE tpic_integration_transport NAME ${test}_2rank
        TARGET tpic_test_${test}_device RANKS 2 LABELS device integration
        RUN_SERIAL SKIP_RETURN_CODE 77 TIMEOUT ${TPIC_INTEGRATION_TIMEOUT})
    endforeach()
  endif()
  # Every kind of mass-halo direction, on three-dimensional rank grids.
  foreach(count IN ITEMS 8 12)
    tpic_add_native_case(SUITE tpic_integration_transport NAME mass_halo_${count}rank
      TARGET tpic_test_mass_halo_device RANKS ${count} LABELS device integration
      RUN_SERIAL SKIP_RETURN_CODE 77 TIMEOUT ${TPIC_INTEGRATION_TIMEOUT})
  endforeach()
  set(integration_suites overlap mass_apply)
  if(TPIC_INTEGRATION_BASELINE)
    if(NOT EXISTS "${TPIC_INTEGRATION_BASELINE}")
      message(FATAL_ERROR "TPIC_INTEGRATION_BASELINE must name an existing executable")
    endif()
    list(APPEND integration_suites snapshot_overlap)
  endif()
  foreach(suite IN LISTS integration_suites)
    set(baseline)
    if(suite STREQUAL "snapshot_overlap")
      set(baseline "-DTPIC_BASELINE=${TPIC_INTEGRATION_BASELINE}")
    endif()
    tpic_add_test_case(SUITE tpic_integration_${suite} NAME ${suite}
      LABELS device integration RUN_SERIAL TIMEOUT 3600
      COMMAND "${CMAKE_COMMAND}"
        -DTPIC_PYTHON=${Python3_EXECUTABLE}
        -DTPIC_SUITE=${CMAKE_CURRENT_SOURCE_DIR}/simulation/${suite}_integration.py
        -DTPIC_EXECUTABLE=$<TARGET_FILE:tpic> ${baseline}
        -DTPIC_WORK_ROOT=${CMAKE_CURRENT_BINARY_DIR}/integration-results
        "-DTPIC_LAUNCHER=${TPIC_INTEGRATION_LAUNCHER}"
        "-DTPIC_RANKS=${TPIC_INTEGRATION_RANKS}"
        -DTPIC_JOB_TIMEOUT=${TPIC_INTEGRATION_TIMEOUT}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/simulation/run_integration.cmake")
  endforeach()
endif()

foreach(module IN ITEMS model config initialization ecsim io simulation_support)
  if(NOT TARGET TPIC::${module})
    message(FATAL_ERROR "Missing module target TPIC::${module}")
  endif()
endforeach()

# Compile supported host headers individually, using only the production include directory.
foreach(header IN ITEMS
    device/block_threads device/thread_indexing config/limits config/precision data/output/scalar_sample data/mass_matrix/mass_matrix_layout
    config/domain config/simulation_parameters config/physics config/boundary_conditions config/output
    config/output_selection
    config/numerics_options
    initialization/prepared_initialization initialization/cases/initial_case
    ecsim/maxwell/maxwell ecsim/particles/cic
    ecsim/fields/node_smoothing ecsim/fields/field_boundary communication/particles/particle_geometry
    simulation/planning/run_plan simulation/execution/execution_layout simulation/execution/deposition_ledger
    simulation/driver/pipeline
    io/diagnostics_writer
    io/output_engine io/async_output_writer data/output/output_snapshot)
  string(REPLACE "/" "_" name "${header}")
  set(source "${CMAKE_CURRENT_BINARY_DIR}/public_${name}.cpp")
  file(GENERATE OUTPUT "${source}" CONTENT "#include <tpic/${header}.hpp>\n")
  add_library(tpic_public_${name} OBJECT "${source}")
  target_link_libraries(tpic_public_${name} PRIVATE TPIC::model)
endforeach()

tpic_register_test_suites()
