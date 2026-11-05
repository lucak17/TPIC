# Find AMD ROCm Communication Collectives Library (RCCL).
#
# The ROCm package commonly exports roc::rccl.  This module normalizes that
# target, or a header/library installation, to RCCL::RCCL so LPFx build-tree
# and installed-package consumers have one stable dependency name.
#
# Optional hints:
#   RCCL_ROOT                 installation prefix
#   environment variable RCCL_ROOT
#
# Result variables:
#   RCCL_FOUND
#   RCCL_VERSION
#   RCCL_INCLUDE_DIRS
#   RCCL_LIBRARIES
#
# Imported target:
#   RCCL::RCCL

find_package(rccl CONFIG QUIET)

set(_rccl_config_target "")
foreach(_rccl_candidate IN ITEMS RCCL::RCCL roc::rccl rccl)
  if(TARGET ${_rccl_candidate})
    set(_rccl_config_target "${_rccl_candidate}")
    break()
  endif()
endforeach()

if(_rccl_config_target)
  if(DEFINED rccl_VERSION AND NOT RCCL_VERSION)
    set(RCCL_VERSION "${rccl_VERSION}")
  endif()
  get_target_property(_rccl_target_includes "${_rccl_config_target}"
    INTERFACE_INCLUDE_DIRECTORIES)
  if(_rccl_target_includes)
    foreach(_rccl_target_include IN LISTS _rccl_target_includes)
      if(EXISTS "${_rccl_target_include}/rccl/rccl.h")
        set(RCCL_INCLUDE_DIR "${_rccl_target_include}")
        break()
      endif()
    endforeach()
  endif()
  if(NOT RCCL_VERSION AND RCCL_INCLUDE_DIR)
    file(READ "${RCCL_INCLUDE_DIR}/rccl/rccl.h" _rccl_header)
    string(REGEX MATCH
      "#[ \t]*define[ \t]+NCCL_MAJOR[ \t]+([0-9]+)"
      _rccl_major_match "${_rccl_header}")
    set(_rccl_major "${CMAKE_MATCH_1}")
    string(REGEX MATCH
      "#[ \t]*define[ \t]+NCCL_MINOR[ \t]+([0-9]+)"
      _rccl_minor_match "${_rccl_header}")
    set(_rccl_minor "${CMAKE_MATCH_1}")
    string(REGEX MATCH
      "#[ \t]*define[ \t]+NCCL_PATCH[ \t]+([0-9]+)"
      _rccl_patch_match "${_rccl_header}")
    set(_rccl_patch "${CMAKE_MATCH_1}")
    if(_rccl_major_match AND _rccl_minor_match AND _rccl_patch_match)
      set(RCCL_VERSION "${_rccl_major}.${_rccl_minor}.${_rccl_patch}")
    endif()
  endif()
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(RCCL
    REQUIRED_VARS _rccl_config_target
    VERSION_VAR RCCL_VERSION)
  if(RCCL_FOUND AND NOT TARGET RCCL::RCCL)
    add_library(RCCL::RCCL INTERFACE IMPORTED)
    set_property(TARGET RCCL::RCCL PROPERTY
      INTERFACE_LINK_LIBRARIES "${_rccl_config_target}")
  endif()
  if(RCCL_FOUND)
    set(RCCL_INCLUDE_DIRS "${RCCL_INCLUDE_DIR}")
    set(RCCL_LIBRARIES "RCCL::RCCL")
  endif()
else()
  find_path(RCCL_INCLUDE_DIR
    NAMES rccl/rccl.h
    HINTS "${RCCL_ROOT}" ENV RCCL_ROOT
    PATH_SUFFIXES include)

  find_library(RCCL_LIBRARY
    NAMES rccl
    HINTS "${RCCL_ROOT}" ENV RCCL_ROOT
    PATH_SUFFIXES lib lib64)

  if(RCCL_INCLUDE_DIR AND EXISTS "${RCCL_INCLUDE_DIR}/rccl/rccl.h")
    file(READ "${RCCL_INCLUDE_DIR}/rccl/rccl.h" _rccl_header)
    string(REGEX MATCH
      "#[ \t]*define[ \t]+NCCL_MAJOR[ \t]+([0-9]+)"
      _rccl_major_match "${_rccl_header}")
    set(_rccl_major "${CMAKE_MATCH_1}")
    string(REGEX MATCH
      "#[ \t]*define[ \t]+NCCL_MINOR[ \t]+([0-9]+)"
      _rccl_minor_match "${_rccl_header}")
    set(_rccl_minor "${CMAKE_MATCH_1}")
    string(REGEX MATCH
      "#[ \t]*define[ \t]+NCCL_PATCH[ \t]+([0-9]+)"
      _rccl_patch_match "${_rccl_header}")
    set(_rccl_patch "${CMAKE_MATCH_1}")
    if(_rccl_major_match AND _rccl_minor_match AND _rccl_patch_match)
      set(RCCL_VERSION "${_rccl_major}.${_rccl_minor}.${_rccl_patch}")
    endif()
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(RCCL
    REQUIRED_VARS RCCL_INCLUDE_DIR RCCL_LIBRARY
    VERSION_VAR RCCL_VERSION)

  if(RCCL_FOUND)
    set(RCCL_INCLUDE_DIRS "${RCCL_INCLUDE_DIR}")
    set(RCCL_LIBRARIES "${RCCL_LIBRARY}")
    if(NOT TARGET RCCL::RCCL)
      add_library(RCCL::RCCL UNKNOWN IMPORTED)
      set_target_properties(RCCL::RCCL PROPERTIES
        IMPORTED_LOCATION "${RCCL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${RCCL_INCLUDE_DIR}")
    endif()
  endif()

  mark_as_advanced(RCCL_INCLUDE_DIR RCCL_LIBRARY)
endif()

unset(_rccl_config_target)
unset(_rccl_candidate)
unset(_rccl_target_includes)
unset(_rccl_target_include)
unset(_rccl_header)
unset(_rccl_major)
unset(_rccl_minor)
unset(_rccl_patch)
unset(_rccl_major_match)
unset(_rccl_minor_match)
unset(_rccl_patch_match)
