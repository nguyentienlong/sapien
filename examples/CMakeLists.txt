# Only Sapiens inself should be compiled with SAPIENS_BUILDING_SHARED_LIBRARY
# defined, any users of Sapiens will have SAPIENS_USING_SHARED_LIBRARY defined
# for them in Sapiens' config.h if appropriate
if (BUILD_SHARED_LIBS)
  remove_definitions(-DSAPIENS_BUILDING_SHARED_LIBRARY)
endif()

add_executable(sgd_mnist sgd_mnist.cc)
target_link_libraries(sgd_mnist sapien)

add_executable(sgd_regressor sgd_regressor.cc)
target_link_libraries(sgd_regressor sapien)

add_executable(line_search_minimizer line_search_minimizer.cc)
target_link_libraries(line_search_minimizer sapien)
