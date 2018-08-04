// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com

#include "sapien/utility/wall_time.h"

#ifdef SAPIEN_USE_OPENMP
#include <omp.h>
#else
#include <ctime>
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

namespace sapien {
namespace internal {

double WallTimeInSeconds() {
#ifdef SAPIEN_USE_OPENMP
  return omp_get_wtime();
#else
#ifdef _WIN32
  LARGE_INTEGER count;
  LARGE_INTEGER frequency;
  QueryPerformanceCounter(&count);
  QueryPerformanceFrequency(&frequency);
  return static_cast<double>(count.QuadPart) /
      static_cast<double>(frequency.QuadPart);
#else
  timeval time_val;
  gettimeofday(&time_val, NULL);
  return (time_val.tv_sec + time_val.tv_usec * 1e-6);
#endif
#endif
}

}  // namespace internal
}  // namespace sapien
