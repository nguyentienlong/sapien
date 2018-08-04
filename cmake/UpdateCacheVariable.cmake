# By default, there is no easy way in CMake to set the value of a cache
# variable without reinitialising it, which involves resetting its
# associated help string.  This is particularly annoying for CMake options
# where they need to programmatically updated.
#
# This function automates this process by getting the current help string
# for the cache variable to update, then reinitialising it with the new
# value, but with the original help string.
function(UPDATE_CACHE_VARIABLE VAR_NAME VALUE)
  get_property(IS_DEFINED_IN_CACHE CACHE ${VAR_NAME} PROPERTY VALUE SET)
  if (NOT IS_DEFINED_IN_CACHE)
    message(FATAL_ERROR "Specified variable to update in cache: "
      "${VAR_NAME} has not been set in the cache.")
  endif()
  get_property(HELP_STRING CACHE ${VAR_NAME} PROPERTY HELPSTRING)
  get_property(VAR_TYPE CACHE ${VAR_NAME} PROPERTY TYPE)
  set(${VAR_NAME} ${VALUE} CACHE ${VAR_TYPE} "${HELP_STRING}" FORCE)
endfunction()