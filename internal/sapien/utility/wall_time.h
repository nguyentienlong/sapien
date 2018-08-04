// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// Time utilities.

#ifndef INTERNAL_SAPIEN_UTILITY_WALL_TIME_H_
#define INTERNAL_SAPIEN_UTILITY_WALL_TIME_H_

#include "sapien/internal/port.h"

namespace sapien {
namespace internal {

// Returns time, in seconds, from some arbitrary point.
double WallTimeInSeconds();
}  // namespace internal
}  // namespace sapien
#endif  // INTERNAL_SAPIEN_UTILITY_WALL_TIME_H_
