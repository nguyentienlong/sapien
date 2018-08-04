# The config file
set(SAPIEN_CONFIG_IN_FILE "${CMAKE_CURRENT_LIST_DIR}/config.h.in")

# CreateSapienConfig.cmake - Create the config.h for Sapien
#
# This function configures the Sapien config.h file based on the current
# compile options and copies it into the specified location. It should be
# called before Sapien is built so that the correct config.h is used when
# Sapien is compiled.
#
# INPUTS:
#   CURRENT_SAPIEN_COMPILE_OPTIONS: List of currently enabled Sapien compile
#                                    options. These are compared against the
#                                    full list of valid options, which are
#                                    read from config.h.in. Any options
#                                    which are not part of the valid set will
#                                    invoke an error. Any valid option present
#                                    will be enabled in the resulting config.h
#                                    all other options will be disabled.
#
#   SAPIEN_CONFIG_OUTPUT_DIRECTORY: Path to output directory in which to save
#                                    the configured config.h file. Typically,
#                                    this will be <src>/include/sapiens/internal.

function(CREATE_SAPIEN_CONFIG
  CURRENT_SAPIEN_COMPILE_OPTIONS
  SAPIEN_CONFIG_OUTPUT_DIRECTORY
)
  # Create the specified outout directory if it does not exist.
  if (NOT EXISTS "${SAPIEN_CONFIG_OUTPUT_DIRECTORY}")
    message(STATUS "Creating configured Sapien config.h output directory: "
      "${SAPIEN_CONFIG_OUTPUT_DIRECTORY}")
    file(MAKE_DIRECTORY "${SAPIEN_CONFIG_OUTPUT_DIRECTORY}")
  endif()
  if (EXISTS "${SAPIEN_CONFIG_OUTPUT_DIRECTORY}" AND
      NOT IS_DIRECTORY "${SAPIEN_CONFIG_OUTPUT_DIRECTORY}")
    message(FATAL_ERROR "Sapien Bug: Specified "
      "SAPIEN_CONFIG_OUTPUT_DIRECTORY: "
      "${SAPIEN_CONFIG_OUTPUT_DIRECTORY} exists, but it is not a directory.")
  endif()

  # Read all possible configurable compile options from config.h.in, this
  # avoids us having to hard-code in this file what the valid options are.
  file(READ ${SAPIEN_CONFIG_IN_FILE} SAPIEN_CONFIG_IN_CONTENTS)
  string(REGEX MATCHALL "@[^@ $]+@"
    ALL_CONFIGURABLE_SAPIEN_OPTIONS "${SAPIEN_CONFIG_IN_CONTENTS}")
  # Removing @ symbol at the beginning and end of each option.
  string(REPLACE "@" "" ALL_CONFIGURABLE_SAPIEN_OPTIONS
    "${ALL_CONFIGURABLE_SAPIEN_OPTIONS}")

  # Ensure that there are no repetitions in the current compile options.
  list(REMOVE_DUPLICATES CURRENT_SAPIEN_COMPILE_OPTIONS)

  foreach (SAPIEN_OPTION ${ALL_CONFIGURABLE_SAPIEN_OPTIONS})
    # Try and find the option in the list of current compile options,
    # if it is present, then the option is enabled, otherwise it is disabled.
    list(FIND CURRENT_SAPIEN_COMPILE_OPTIONS ${SAPIEN_OPTION}
      OPTION_ENABLED)

    # list(FIND ..) returns -1 if the element was not in the list, but CMake
    # interprets if (VAR) to be true if VAR is any none-zero number, even
    # negative ones, hence we have to explicitly check for >= 0.
    if (OPTION_ENABLED GREATER -1)
      message(STATUS "Enabling ${SAPIEN_OPTION} in Sapien config.h")
      set(${SAPIEN_OPTION} "#define ${SAPIEN_OPTION}")

      # Remove the item from the list of current options so that we can
      # identify any options that were in CURRENT_SAPIEN_COMPILE_OPTIONS,
      # but not in ALL_CONFIGURABLE_SAPIEN_OPTIONS (which is an error).
      list(REMOVE_ITEM CURRENT_SAPIEN_COMPILE_OPTIONS ${SAPIEN_OPTION})
    else()
      set(${SAPIEN_OPTION} "// #define ${SAPIEN_OPTION}")
    endif()
  endforeach()

  # CURRENT_SAPIEN_COMPILE_OPTIONS should now be an empty list, any elements
  # remaining were not present in ALL_CONFIGURABLE_SAPIEN_OPTIONS read from
  # config.h.in
  if (CURRENT_SAPIEN_COMPILE_OPTIONS)
    message(FATAL_ERROR "Sapien Bug: CURRENT_SAPIEN_COMPILE_OPTIONS "
      "contained the following options which were not present in "
      "config.h.in: ${CURRENT_SAPIEN_COMPILE_OPTIONS}")
  endif()

  configure_file(${SAPIEN_CONFIG_IN_FILE}
    "${SAPIEN_CONFIG_OUTPUT_DIRECTORY}/config.h" @ONLY)
endfunction()
    