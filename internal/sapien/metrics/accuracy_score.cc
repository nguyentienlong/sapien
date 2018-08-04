// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com

#include "sapien/internal/port.h"
#include "sapien/metrics.h"
#include "glog/logging.h"

namespace sapien {
namespace metrics {

#define ACCURACY_SCORE(t) template \
  double AccuracyScore<t>(const size_t N, const t* y_true, const t* y_pred)

template<typename LabelType>
double AccuracyScore(const size_t N, const LabelType* y_true,
                     const LabelType* y_pred) {
  size_t count = 0;
  size_t i, n;
  n = N;
  i = n >> 4;  // Multiple of 16 (1 << 4)
  if (i) {
    n -= (i << 4);
    do {
      if (*y_true == *y_pred) ++count;
      if (y_true[1] == y_pred[1]) ++count;
      if (y_true[2] == y_pred[2]) ++count;
      if (y_true[3] == y_pred[3]) ++count;
      if (y_true[4] == y_pred[4]) ++count;
      if (y_true[5] == y_pred[5]) ++count;
      if (y_true[6] == y_pred[6]) ++count;
      if (y_true[7] == y_pred[7]) ++count;
      if (y_true[8] == y_pred[8]) ++count;
      if (y_true[9] == y_pred[9]) ++count;
      if (y_true[10] == y_pred[10]) ++count;
      if (y_true[11] == y_pred[11]) ++count;
      if (y_true[12] == y_pred[12]) ++count;
      if (y_true[13] == y_pred[13]) ++count;
      if (y_true[14] == y_pred[14]) ++count;
      if (y_true[15] == y_pred[15]) ++count;
      y_true += 16;
      y_pred += 16;
    } while (--i);
  }

  if (n >> 3) {  // >= 8
    if (*y_true == *y_pred) ++count;
    if (y_true[1] == y_pred[1]) ++count;
    if (y_true[2] == y_pred[2]) ++count;
    if (y_true[3] == y_pred[3]) ++count;
    if (y_true[4] == y_pred[4]) ++count;
    if (y_true[5] == y_pred[5]) ++count;
    if (y_true[6] == y_pred[6]) ++count;
    if (y_true[7] == y_pred[7]) ++count;
    y_true += 8;
    y_pred += 8;
    n -= 8;
  }

  switch (n) {  // left over
    case 1:
      if (*y_true == *y_pred) ++count;
      break;
    case 2:
      if (*y_true == *y_pred) ++count;
      if (y_true[1] == y_pred[1]) ++count;
      break;
    case 3:
      if (*y_true == *y_pred) ++count;
      if (y_true[1] == y_pred[1]) ++count;
      if (y_true[2] == y_pred[2]) ++count;
      break;
    case 4:
      if (*y_true == *y_pred) ++count;
      if (y_true[1] == y_pred[1]) ++count;
      if (y_true[2] == y_pred[2]) ++count;
      if (y_true[3] == y_pred[3]) ++count;
      break;
    case 5:
      if (*y_true == *y_pred) ++count;
      if (y_true[1] == y_pred[1]) ++count;
      if (y_true[2] == y_pred[2]) ++count;
      if (y_true[3] == y_pred[3]) ++count;
      if (y_true[4] == y_pred[4]) ++count;
      break;
    case 6:
      if (*y_true == *y_pred) ++count;
      if (y_true[1] == y_pred[1]) ++count;
      if (y_true[2] == y_pred[2]) ++count;
      if (y_true[3] == y_pred[3]) ++count;
      if (y_true[4] == y_pred[4]) ++count;
      if (y_true[5] == y_pred[5]) ++count;
      break;
    case 7:
      if (*y_true == *y_pred) ++count;
      if (y_true[1] == y_pred[1]) ++count;
      if (y_true[2] == y_pred[2]) ++count;
      if (y_true[3] == y_pred[3]) ++count;
      if (y_true[4] == y_pred[4]) ++count;
      if (y_true[5] == y_pred[5]) ++count;
      if (y_true[6] == y_pred[6]) ++count;
      break;
    default:
      break;
  }

  return static_cast<double>(count)/static_cast<double>(N);
}

ACCURACY_SCORE(int8_t);
ACCURACY_SCORE(uint8_t);
ACCURACY_SCORE(int16_t);
ACCURACY_SCORE(uint16_t);
ACCURACY_SCORE(int32_t);
ACCURACY_SCORE(uint32_t);
ACCURACY_SCORE(int64_t);
ACCURACY_SCORE(uint64_t);

#undef ACCURACY_SCORE
}  // namespace metrics
}  // namespace sapien
