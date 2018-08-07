# List all internal source files. Do NOT use file(GLOB *) to find source!
set(SAPIEN_INTERNAL_SRC
  math/sapien_math.cc

  # metrics
  metrics/accuracy_score.cc
  metrics/r2_score.cc

  # utility
  utility/weight_vector.cc
  utility/sequential_dataset.cc
  utility/stringprintf.cc
  utility/wall_time.cc

  # sgd
  sgd/loss.cc
  sgd/base.cc
  sgd/types.cc
  sgd/default_options.cc
  sgd/sgd_classifier.cc
  sgd/sgd_regressor.cc

  # dataset
  dataset/load_mnist.cc
)

# Also depends on the internal header files so that they appear in IDES.
set(SAPIEN_INTERNAL_HDRS  "")
file(GLOB SAPIEN_INTERNAL_HDRS
  utility/*.h
  sgd/*.h
)

# Depend also on public headers so they appear in IDEs.
file(GLOB SAPIEN_PUBLIC_HDRS
  ${Sapien_SOURCE_DIR}/include/sapien/*.h
  ${Sapien_SOURCE_DIR}/include/sapien/sgd/*.h)
file(GLOB SAPIEN_PUBLIC_INTERNAL_HDRS
  ${Sapien_SOURCE_DIR}/include/sapien/internal/*.h
  ${Sapien_SOURCE_DIR}/include/sapien/internal/type_traits/*.h)

set(SAPIEN_LIBRARY_SOURCE
  ${SAPIEN_INTERNAL_SRC}
  ${SAPIEN_INTERNAL_HDRS}
  ${SAPIEN_PUBLIC_HDRS}
  ${SAPIEN_PUBLIC_INTERNAL_HDRS})

add_library(sapien ${SAPIEN_LIBRARY_SOURCE})
set_target_properties(sapien PROPERTIES
  VERSION ${SAPIEN_VERSION}
  SOVERSION ${SAPIEN_VERSION_MAJOR})

# The ability to specify a minimum language version via cxx_std_[11,14,17]
# requires CMake >= 3.8.  Prior to that we have to specify the compiler features
# we require.
if (CMAKE_VERSION VERSION_LESS 3.8)
  set(REQUIRED_PUBLIC_CXX_FEATURES cxx_deleted_functions)
else()
  # Forward whatever C++ version Ceres was compiled with as our requirement
  # for downstream clients.
  set(REQUIRED_PUBLIC_CXX_FEATURES cxx_std_${CMAKE_CXX_STANDARD})
endif()
target_compile_features(sapien PUBLIC ${REQUIRED_PUBLIC_CXX_FEATURES})


# Always build position-independent code (PIC), even when Building Sapien
# as a static library so that shared library can link against it.
if (NOT WIN32 AND NOT BUILD_SHARED_LIBS)
  set_target_properties(sapien PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# Sapien dependencies.
set(SAPIEN_LIBRARY_PUBLIC_DEPENDENCIES "")
set(SAPIEN_LIBRARY_PRIVATE_DEPENDENCIES "")

list(APPEND SAPIEN_LIBRARY_PUBLIC_DEPENDENCIES ${GLOG_LIBRARIES})
list(APPEND SAPIEN_LIBRARY_PRIVATE_DEPENDENCIES ${BLAS_LIBRARIES})

if (BUILD_SHARED_LIBS)
  # When building a shared library, mark all external libraries as PRIVATE
  # so they don't show up as a dependency.
  target_link_libraries(sapien
    LINK_PUBLIC ${SAPIEN_LIBRARY_PUBLIC_DEPENDENCIES}
    LINK_PRIVATE ${SAPIEN_LIBRARY_PRIVATE_DEPENDENCIES})
else()
  # When building a static library, all external libraries are PUBLIC(default)
  set(SAPIEN_LIBRARY_DEPENDENCIES
    ${SAPIEN_LIBRARY_PUBLIC_DEPENDENCIES}
    ${SAPIEN_LIBRARY_PRIVATE_DEPENDENCIES})
  target_link_libraries(sapien ${SAPIEN_LIBRARY_DEPENDENCIES})
endif()

# Add the Sapien headers to its target.
#
# Force the location containing the configured config.h to the front
# of the include_directories list (by default it is appended to the back)
# to ensure that if the user has an installed version of Sapien in the same
# location as one of the dependencies (e.g. usr/local) that we find the
# config.h we just configured, not the (older) installed config.h
target_include_directories(sapien BEFORE PUBLIC
  $<BUILD_INTERFACE:${Sapien_BINARY_DIR}/config>)
target_include_directories(sapien PRIVATE ${Sapien_SOURCE_DIR}/internal)
target_include_directories(sapien PUBLIC
  $<BUILD_INTERFACE:${Sapien_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Gather the list of public & private include locations for all enabled
# optional dependencies to be added to the Sapien target.
set(SAPIEN_LIBRARY_PRIVATE_DEPENDENCIES_INCLUDE_DIRS "")
set(SAPIEN_LIBRARY_PUBLIC_DEPENDENCIES_INCLUDE_DIRS "")

# NOTE: The variable FOUND_INSTALLED_GLOG_CMAKE_CONFIGURATION is defined
#       in FindGlog.cmake module.
if (NOT FOUND_INSTALLED_GLOG_CMAKE_CONFIGURATION)
  # Only append glog include directories if the glog found was not a CMake
  # exported target that alredy includes them.
  list(APPEND SAPIEN_LIBRARY_PUBLIC_DEPENDENCIES_INCLUDE_DIRS
    ${GLOG_INCLUDE_DIRS})
endif()

list(APPEND SAPIEN_LIBRARY_PRIVATE_DEPENDENCIES_INCLUDE_DIRS
  ${BLAS_INCLUDE_DIRS})

list(REMOVE_DUPLICATES SAPIEN_LIBRARY_PRIVATE_DEPENDENCIES_INCLUDE_DIRS)
foreach(INC_DIR ${SAPIEN_LIBRARY_PRIVATE_DEPENDENCIES_INCLUDE_DIRS})
  target_include_directories(sapien PRIVATE ${INC_DIR})
endforeach()
list(REMOVE_DUPLICATES SAPIEN_LIBRARY_PUBLIC_DEPENDENCIES_INCLUDE_DIRS)
foreach(INC_DIR ${SAPIEN_LIBRARY_PUBLIC_DEPENDENCIES_INCLUDE_DIRS})
  target_include_directories(sapien PUBLIC ${INC_DIR})
endforeach()

install(TARGETS sapien
        EXPORT  SapienExport
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib${LIB_SUFFIX}
	ARCHIVE DESTINATION lib${LIB_SUFFIX})


include(AppendTargetProperty)

if (BUILD_TESTING)
  add_library(gtest
    ${Sapien_SOURCE_DIR}/internal/gtest/gmock_gtest_all.cc
    ${Sapien_SOURCE_DIR}/internal/gtest/gmock_main.cc)
  target_include_directories(gtest PUBLIC
    ${Sapien_SOURCE_DIR}/internal/gtest)
  if (BUILD_SHARED_LIBS)
    # Define gtest-specific shared library flags for compilation.
    append_target_property(gtest COMPILE_DEFINITIONS
      GTEST_CREATE_SHARED_LIBRARY)
  endif()

  target_link_libraries(gtest ${GLOG_LIBRARIES})

  macro (SAPIEN_TEST NAME)
    add_executable(${NAME}_test ${NAME}_test.cc)
    # Pull in local headers from the generated test directories when
    # sapien_test() is invoked there, as well as the private headers in
    # this directory which may be referenced without the 'sapien' path
    # prefix and all private dependencies that may be directly referenced.
    target_include_directories(${NAME}_test
      PUBLIC ${CMAKE_CURRENT_LIST_DIR}
             ${Sapien_SOURCE_DIR}/internal
	     ${SAPIEN_LIBRARY_PRIVATE_DEPENDENCIES_INCLUDE_DIRS})

    target_link_libraries(${NAME}_test sapien gtest)

    if (BUILD_SHARED_LIBS)
      # Define gtest-specific shared library flags for linking.
      append_target_property(${NAME}_test COMPILE_DEFINITIONS
        GTEST_LINKED_AS_SHARED_LIBRARY)
    endif()

    add_test(NAME ${NAME}_test
             COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}_test
	     --test_srcdir
	     ${Sapien_SOURCE_DIR}/data)
  endmacro (SAPIEN_TEST)

  add_subdirectory(math/test)
  add_subdirectory(metrics/test)
  add_subdirectory(utility/test)
  add_subdirectory(sgd/test)
endif (BUILD_TESTING)
