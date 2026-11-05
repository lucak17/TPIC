# Host configuration tests can run without a GPU toolchain. This switch does
# not enable a CPU simulation; the numerical application remains device-only.
option(TPIC_BUILD_DEVICE "Build CUDA/HIP simulation and schema checks" ON)
option(TPIC_BUILD_TESTS "Build the TPIC CTest suites" ON)
option(TPIC_ENABLE_ADIOS2 "Enable rank-local ADIOS2 BP5 field, particle and checkpoint output" OFF)
option(TPIC_ENABLE_HDF5 "Enable rank-local HDF5 field, particle and checkpoint output" OFF)
option(TPIC_VALIDATE_PINNED_TRANSFERS "Audit native application host/device transfer pointers" OFF)
if(TPIC_VALIDATE_PINNED_TRANSFERS AND (NOT TPIC_BUILD_DEVICE OR NOT UNIX OR APPLE))
  message(FATAL_ERROR "TPIC_VALIDATE_PINNED_TRANSFERS requires a CUDA/HIP build and a linker supporting --wrap")
endif()
option(TPIC_TEST_INTEGRATION "Register real GPU/MPI integration suites" OFF)
set(TPIC_INTEGRATION_LAUNCHER "mpiexec" CACHE STRING "MPI launcher command for integration suites")
set(TPIC_INTEGRATION_RANKS "1;2" CACHE STRING "MPI process counts for integration suites")
set(TPIC_INTEGRATION_TIMEOUT 90 CACHE STRING "Timeout in seconds per integration MPI job")
set(TPIC_INTEGRATION_BASELINE "" CACHE FILEPATH "Baseline executable for snapshot comparison")
if(TPIC_TEST_INTEGRATION AND (NOT TPIC_BUILD_TESTS OR NOT TPIC_BUILD_DEVICE))
  message(FATAL_ERROR "TPIC_TEST_INTEGRATION requires TPIC_BUILD_TESTS and TPIC_BUILD_DEVICE")
endif()
set(TPIC_COMMUNICATION_BACKEND "MPI" CACHE STRING "LPFx device payload transport: MPI, NCCL or RCCL")
set_property(CACHE TPIC_COMMUNICATION_BACKEND PROPERTY STRINGS MPI NCCL RCCL)
if(NOT TPIC_COMMUNICATION_BACKEND MATCHES "^(MPI|NCCL|RCCL)$")
  message(FATAL_ERROR "TPIC_COMMUNICATION_BACKEND must be MPI, NCCL or RCCL")
endif()
set(TPIC_MAX_SPECIES 10 CACHE STRING "Compiled capacity of real species; a run uses the ns of its input")
if(NOT TPIC_MAX_SPECIES MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "TPIC_MAX_SPECIES must be a positive integer")
endif()
set(TPIC_PARTICLE_MIGRATION_HOPS 1 CACHE STRING
    "Migration routing rounds per species and cycle; one covers every admitted flight")
if(NOT TPIC_PARTICLE_MIGRATION_HOPS MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "TPIC_PARTICLE_MIGRATION_HOPS must be a positive integer")
endif()
# Precision: a run selects one storage level and one compute level. Each compiled
# pair is one translation unit with its Simulation instantiations. The
# default build compiles the five matched pairs, selected at run time with
# --precision: the same level for storage and compute, except that the
# double-mass level keeps single-particles-moments-fields storage
# (matched_precision in include/tpic/config/precision.hpp).
set(_tpic_precision_levels double single-position single-position-velocity
    single-particles-moments-fields-double-mass single-particles-moments-fields)
list(LENGTH _tpic_precision_levels _tpic_precision_level_count)
set(_tpic_precision_structs Double SinglePosition SinglePositionVelocity
    SingleParticlesMomentsFieldsDoubleMass SingleParticlesMomentsFields)
option(TPIC_BUILD_ALL_PRECISION_PAIRS
       "Compile all 25 storage/compute precision pairs instead of the 5 matched ones" OFF)
set(TPIC_PRECISION_PAIRS "" CACHE STRING
    "Explicit ;-list of compiled storage/compute pairs, such as double/double;single-position/double; empty selects the matched pairs, or all with TPIC_BUILD_ALL_PRECISION_PAIRS")
set(TPIC_DEFAULT_STORAGE_PRECISION "double" CACHE STRING "Default --storage-precision")
set(TPIC_DEFAULT_COMPUTE_PRECISION "double" CACHE STRING "Default --compute-precision")
set_property(CACHE TPIC_DEFAULT_STORAGE_PRECISION PROPERTY STRINGS ${_tpic_precision_levels})
set_property(CACHE TPIC_DEFAULT_COMPUTE_PRECISION PROPERTY STRINGS ${_tpic_precision_levels})
if(TPIC_PRECISION_PAIRS)
  set(_tpic_precision_pairs ${TPIC_PRECISION_PAIRS})
elseif(TPIC_BUILD_ALL_PRECISION_PAIRS)
  set(_tpic_precision_pairs)
  foreach(_storage IN LISTS _tpic_precision_levels)
    foreach(_compute IN LISTS _tpic_precision_levels)
      list(APPEND _tpic_precision_pairs "${_storage}/${_compute}")
    endforeach()
  endforeach()
else()
  set(_tpic_precision_pairs)
  foreach(_level IN LISTS _tpic_precision_levels)
    if(_level STREQUAL "single-particles-moments-fields-double-mass")
      list(APPEND _tpic_precision_pairs "single-particles-moments-fields/${_level}")
    else()
      list(APPEND _tpic_precision_pairs "${_level}/${_level}")
    endif()
  endforeach()
endif()
list(REMOVE_DUPLICATES _tpic_precision_pairs)
# TPIC_PRECISION_PAIR_UNITS holds "<storage id>;<compute id>" flattened; bit
# storage * <level count> + compute of TPIC_PRECISION_PAIR_MASK marks a compiled pair.
set(TPIC_PRECISION_PAIR_MASK 0)
set(TPIC_PRECISION_PAIR_UNITS)
foreach(_pair IN LISTS _tpic_precision_pairs)
  if(NOT _pair MATCHES "^([a-z-]+)/([a-z-]+)$")
    message(FATAL_ERROR "TPIC_PRECISION_PAIRS entry '${_pair}' must be <storage>/<compute>. "
            "Leave the list empty (cmake -U TPIC_PRECISION_PAIRS) for the matched pairs, "
            "or set TPIC_BUILD_ALL_PRECISION_PAIRS=ON for every pair.")
  endif()
  list(FIND _tpic_precision_levels "${CMAKE_MATCH_1}" _storage_id)
  list(FIND _tpic_precision_levels "${CMAKE_MATCH_2}" _compute_id)
  if(_storage_id EQUAL -1 OR _compute_id EQUAL -1)
    message(FATAL_ERROR "TPIC_PRECISION_PAIRS entry '${_pair}': a level must be one of ${_tpic_precision_levels}")
  endif()
  math(EXPR TPIC_PRECISION_PAIR_MASK
       "${TPIC_PRECISION_PAIR_MASK} | (1 << (${_storage_id} * ${_tpic_precision_level_count} + ${_compute_id}))")
  list(APPEND TPIC_PRECISION_PAIR_UNITS ${_storage_id} ${_compute_id})
endforeach()
if(TPIC_TEST_INTEGRATION)
  # The integration suites run every storage level with double compute.
  set(_tpic_integration_mask 0)
  foreach(_storage_level RANGE 1 ${_tpic_precision_level_count})
    math(EXPR _tpic_integration_mask
         "${_tpic_integration_mask} | (1 << ((${_storage_level} - 1) * ${_tpic_precision_level_count}))")
  endforeach()
  math(EXPR _tpic_integration_pairs "${TPIC_PRECISION_PAIR_MASK} & ${_tpic_integration_mask}")
  if(NOT _tpic_integration_pairs EQUAL _tpic_integration_mask)
    message(FATAL_ERROR "TPIC_TEST_INTEGRATION needs every storage level with double compute: set TPIC_BUILD_ALL_PRECISION_PAIRS=ON")
  endif()
endif()
list(FIND _tpic_precision_levels "${TPIC_DEFAULT_STORAGE_PRECISION}" TPIC_DEFAULT_STORAGE_PRECISION_ID)
list(FIND _tpic_precision_levels "${TPIC_DEFAULT_COMPUTE_PRECISION}" TPIC_DEFAULT_COMPUTE_PRECISION_ID)
if(TPIC_DEFAULT_STORAGE_PRECISION_ID EQUAL -1 OR TPIC_DEFAULT_COMPUTE_PRECISION_ID EQUAL -1)
  message(FATAL_ERROR "TPIC_DEFAULT_STORAGE_PRECISION and TPIC_DEFAULT_COMPUTE_PRECISION must be one of ${_tpic_precision_levels}")
endif()
math(EXPR _tpic_default_pair_bit
     "(${TPIC_PRECISION_PAIR_MASK} >> (${TPIC_DEFAULT_STORAGE_PRECISION_ID} * ${_tpic_precision_level_count} + ${TPIC_DEFAULT_COMPUTE_PRECISION_ID})) & 1")
if(NOT _tpic_default_pair_bit)
  message(FATAL_ERROR "The default precision pair ${TPIC_DEFAULT_STORAGE_PRECISION}/${TPIC_DEFAULT_COMPUTE_PRECISION} is not in TPIC_PRECISION_PAIRS")
endif()
list(LENGTH _tpic_precision_pairs _tpic_precision_pair_count)
message(STATUS "TPIC precision pairs compiled: ${_tpic_precision_pair_count} (default ${TPIC_DEFAULT_STORAGE_PRECISION}/${TPIC_DEFAULT_COMPUTE_PRECISION})")
if(TPIC_BUILD_DEVICE AND NOT LPFX_DEVICE_BACKEND MATCHES "^(CUDA|HIP)$")
  message(FATAL_ERROR
    "TPIC requires -DLPFX_DEVICE_BACKEND=CUDA or HIP and explicit "
    "-DLPFX_DEVICE_ARCHITECTURES=<target>. See README.md.")
endif()
if(NOT LPFX_ENABLE_MPI)
  message(FATAL_ERROR "TPIC requires -DLPFX_ENABLE_MPI=ON for LPFx host/control communication and the shared field files")
endif()
if(TPIC_BUILD_DEVICE)
  if(TPIC_COMMUNICATION_BACKEND STREQUAL "MPI" AND NOT LPFX_ENABLE_DEVICE_AWARE_MPI)
    message(FATAL_ERROR
      "TPIC_COMMUNICATION_BACKEND=MPI requires -DLPFX_ENABLE_DEVICE_AWARE_MPI=ON. "
      "This asserts deployment support for device pointers; compilation does not prove it.")
  elseif(TPIC_COMMUNICATION_BACKEND STREQUAL "NCCL" AND
         (NOT LPFX_DEVICE_BACKEND STREQUAL "CUDA" OR NOT LPFX_ENABLE_NCCL))
    message(FATAL_ERROR "TPIC_COMMUNICATION_BACKEND=NCCL requires LPFX_DEVICE_BACKEND=CUDA and LPFX_ENABLE_NCCL=ON")
  elseif(TPIC_COMMUNICATION_BACKEND STREQUAL "RCCL" AND
         (NOT LPFX_DEVICE_BACKEND STREQUAL "HIP" OR NOT LPFX_ENABLE_RCCL))
    message(FATAL_ERROR "TPIC_COMMUNICATION_BACKEND=RCCL requires LPFX_DEVICE_BACKEND=HIP and LPFX_ENABLE_RCCL=ON")
  endif()
  message(STATUS "TPIC LPFx payload backend: ${TPIC_COMMUNICATION_BACKEND} (MPI host/control)")
endif()
