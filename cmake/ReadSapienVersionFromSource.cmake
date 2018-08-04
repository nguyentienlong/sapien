# Extract Sapien version from <Sapien_SOURCE_DIR>/include/sapien/version.h
macro(read_sapien_version_from_source SAPIEN_SOURCE_ROOT)
  set(SAPIEN_VERSION_FILE
    ${SAPIEN_SOURCE_ROOT}/include/sapien/version.h)
  if (NOT EXISTS ${SAPIEN_VERSION_FILE})
    message(FATAL_ERROR "Cannot find Sapien version.h file in specified "
      " Sapien source directory: ${SAPIEN_SOURCE_ROOT}")
  endif()

  file(READ ${SAPIEN_VERSION_FILE} SAPIEN_VERSION_FILE_CONTENTS)

  # Extract major version.
  string(REGEX MATCH "#define SAPIEN_VERSION_MAJOR [0-9]+"
    SAPIEN_VERSION_MAJOR "${SAPIEN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "#define SAPIEN_VERSION_MAJOR ([0-9]+)" "\\1"
    SAPIEN_VERSION_MAJOR "${SAPIEN_VERSION_MAJOR}")
  if ("${SAPIEN_VERSION_MAJOR}" STREQUAL "")
    message(FATAL_ERROR "Failed to extract Sapien major version from "
      "${SAPIEN_VERSION_FILE}")
  endif()

  # Extract minor version.
  string(REGEX MATCH "#define SAPIEN_VERSION_MINOR [0-9]+"
    SAPIEN_VERSION_MINOR "${SAPIEN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "#define SAPIEN_VERSION_MINOR ([0-9]+)" "\\1"
    SAPIEN_VERSION_MINOR "${SAPIEN_VERSION_MINOR}")
  if ("${SAPIEN_VERSION_MINOR}" STREQUAL "")
    message("Failed to extract Sapien minor version from "
      "${SAPIEN_VERSION_FILE}")
  endif()

  # Extract patch version.
  string(REGEX MATCH "#define SAPIEN_VERSION_REVISION [0-9]+"
    SAPIEN_VERSION_PATCH "${SAPIEN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "#define SAPIEN_VERSION_REVISION ([0-9]+)" "\\1"
    SAPIEN_VERSION_PATCH "${SAPIEN_VERSION_PATCH}")
  if ("${SAPIEN_VERSION_PATCH}" STREQUAL "")
    message("Failed to extract Sapien patch version from "
      "${SAPIEN_VERSION_FILE}")
  endif()

  # The full version x.x.x
  set(SAPIEN_VERSION "${SAPIEN_VERSION_MAJOR}.${SAPIEN_VERSION_MINOR}.${SAPIEN_VERSION_PATCH}")

  # Report
  message(STATUS "Detected Sapien version: ${SAPIEN_VERSION} from "
    "${SAPIEN_VERSION_FILE}")
endmacro()    