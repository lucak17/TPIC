# Preserve every run under a new directory, so repeat CTest runs stay reviewable.
string(TIMESTAMP stamp "%Y%m%dT%H%M%S" UTC)
string(RANDOM LENGTH 8 ALPHABET 0123456789abcdef suffix)
get_filename_component(suite_name "${TPIC_SUITE}" NAME_WE)
set(work_dir "${TPIC_WORK_ROOT}/${suite_name}-${stamp}-${suffix}")
set(arguments --executable "${TPIC_EXECUTABLE}" --work-dir "${work_dir}"
  --launcher "${TPIC_LAUNCHER}" --timeout "${TPIC_JOB_TIMEOUT}" --ranks ${TPIC_RANKS})
if(TPIC_BASELINE)
  list(APPEND arguments --baseline "${TPIC_BASELINE}")
endif()
execute_process(COMMAND "${TPIC_PYTHON}" "${TPIC_SUITE}" ${arguments}
  RESULT_VARIABLE result)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Integration suite failed (${result}); results: ${work_dir}")
endif()
