# CrayPat's API header and the API stub library that its perftools-preload link
# mode uses. The stub's PAT_record does nothing; under pat_run, the preloaded
# CrayPat runtime (lib_pat_base.so) supplies the PAT_record that runs.
# perftools-base sets CRAYPAT_ROOT.
find_path(LPFxCrayPat_INCLUDE_DIR NAMES pat_api.h
  HINTS ENV CRAYPAT_ROOT PATH_SUFFIXES include)
find_library(LPFxCrayPat_LIBRARY NAMES _pat_stubs_api
  HINTS ENV CRAYPAT_ROOT PATH_SUFFIXES lib64)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LPFxCrayPat
  REQUIRED_VARS LPFxCrayPat_INCLUDE_DIR LPFxCrayPat_LIBRARY)
if(LPFxCrayPat_FOUND AND NOT TARGET LPFxCrayPat::API)
  add_library(LPFxCrayPat::API UNKNOWN IMPORTED)
  set_target_properties(LPFxCrayPat::API PROPERTIES
    IMPORTED_LOCATION "${LPFxCrayPat_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${LPFxCrayPat_INCLUDE_DIR}")
endif()
mark_as_advanced(LPFxCrayPat_INCLUDE_DIR LPFxCrayPat_LIBRARY)
