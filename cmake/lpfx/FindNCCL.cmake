# Find NVIDIA Collective Communications Library (NCCL).
#
# Optional hints:
#   NCCL_ROOT                 installation prefix
#   environment variable NCCL_ROOT
#
# Result variables:
#   NCCL_FOUND
#   NCCL_VERSION
#   NCCL_INCLUDE_DIRS
#   NCCL_LIBRARIES
#
# An already-defined NCCL::NCCL target is accepted and normalized through the
# same header-version check.  This supports parent projects that discover NCCL
# before LPFx without attempting to redefine their imported target.
#
# Imported target:
#   NCCL::NCCL

set(_nccl_existing_target "")
if(TARGET NCCL::NCCL)
  set(_nccl_existing_target "NCCL::NCCL")
  get_target_property(_nccl_target_includes NCCL::NCCL
    INTERFACE_INCLUDE_DIRECTORIES)
  if(_nccl_target_includes)
    foreach(_nccl_target_include IN LISTS _nccl_target_includes)
      if(EXISTS "${_nccl_target_include}/nccl.h")
        set(NCCL_INCLUDE_DIR "${_nccl_target_include}")
        break()
      endif()
    endforeach()
  endif()
endif()

find_path(NCCL_INCLUDE_DIR
  NAMES nccl.h
  HINTS "${NCCL_ROOT}" ENV NCCL_ROOT
  PATH_SUFFIXES include)

find_library(NCCL_LIBRARY
  NAMES nccl
  HINTS "${NCCL_ROOT}" ENV NCCL_ROOT
  PATH_SUFFIXES lib lib64)

if(NCCL_INCLUDE_DIR AND EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
  file(READ "${NCCL_INCLUDE_DIR}/nccl.h" _nccl_header)
  string(REGEX MATCH
    "#[ \t]*define[ \t]+NCCL_MAJOR[ \t]+([0-9]+)"
    _nccl_major_match "${_nccl_header}")
  set(_nccl_major "${CMAKE_MATCH_1}")
  string(REGEX MATCH
    "#[ \t]*define[ \t]+NCCL_MINOR[ \t]+([0-9]+)"
    _nccl_minor_match "${_nccl_header}")
  set(_nccl_minor "${CMAKE_MATCH_1}")
  string(REGEX MATCH
    "#[ \t]*define[ \t]+NCCL_PATCH[ \t]+([0-9]+)"
    _nccl_patch_match "${_nccl_header}")
  set(_nccl_patch "${CMAKE_MATCH_1}")
  if(_nccl_major_match AND _nccl_minor_match AND _nccl_patch_match)
    set(NCCL_VERSION "${_nccl_major}.${_nccl_minor}.${_nccl_patch}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
if(_nccl_existing_target)
  find_package_handle_standard_args(NCCL
    REQUIRED_VARS _nccl_existing_target
    VERSION_VAR NCCL_VERSION)
else()
  find_package_handle_standard_args(NCCL
    REQUIRED_VARS NCCL_INCLUDE_DIR NCCL_LIBRARY
    VERSION_VAR NCCL_VERSION)
endif()

if(NCCL_FOUND)
  set(NCCL_INCLUDE_DIRS "${NCCL_INCLUDE_DIR}")
  if(_nccl_existing_target)
    set(NCCL_LIBRARIES "${_nccl_existing_target}")
  else()
    set(NCCL_LIBRARIES "${NCCL_LIBRARY}")
  endif()
  if(NOT TARGET NCCL::NCCL)
    add_library(NCCL::NCCL UNKNOWN IMPORTED)
    set_target_properties(NCCL::NCCL PROPERTIES
      IMPORTED_LOCATION "${NCCL_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}")
  endif()
endif()

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)

unset(_nccl_header)
unset(_nccl_major)
unset(_nccl_minor)
unset(_nccl_patch)
unset(_nccl_major_match)
unset(_nccl_minor_match)
unset(_nccl_patch_match)
unset(_nccl_existing_target)
unset(_nccl_target_includes)
unset(_nccl_target_include)
