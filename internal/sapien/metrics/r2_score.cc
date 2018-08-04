// Copyright 2018.

#include <stdlib.h>  /* malloc */
#include <limits>

#include "sapien/metrics.h"
#include "sapien/internal/sapien_math.h"

namespace sapien {
namespace metrics {

using ::sapien::internal::sapien_dot;
using ::sapien::internal::sapien_xmy;
using ::sapien::internal::sapien_xpa;
using ::sapien::internal::sapien_vmean;

// TODO(Linh): Is it overkilled? How about just using plain for loop and
// no blas at all.
template<typename TargetType>
TargetType R2Score(const size_t N, const TargetType* y_true,
                   const TargetType* y_pred) {
  TargetType *res;
  res = reinterpret_cast<TargetType*>(malloc(sizeof(TargetType) * N));
  sapien_xmy(N, y_true, y_pred, res);
  TargetType SS_res = sapien_dot(N, res, res);

  sapien_xpa(N, -sapien_vmean(N, y_true), y_true, res);
  TargetType SS_tot = sapien_dot(N, res, res);

  if (SS_tot == 0.0) {
    return std::numeric_limits<TargetType>::infinity();
  }

  free(res);

  return 1.0 - (SS_res / SS_tot);
}

template float R2Score(const size_t N, const float* y_true,
                       const float* y_pred);
template double R2Score(const size_t N, const double* y_true,
                        const double* y_pred);
}  // namespace metrics
}  // namespace sapien
