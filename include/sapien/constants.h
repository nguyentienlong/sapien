// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com
//
// Defines common used math constants.

#ifndef INCLUDE_SAPIEN_CONSTANTS_H_
#define INCLUDE_SAPIEN_CONSTANTS_H_

#include <limits>

#include "sapien/internal/type_traits.h"

namespace sapien {

// Constant helper
namespace internal {

class ConstantHelper {
 public:
  template<typename T>
  static
  typename enable_if<is_float<T>::value, T>::type
  NaN() {
    return (std::numeric_limits<T>::has_quiet_NaN) ?
        std::numeric_limits<T>::quiet_NaN() : T(0);
  }

  template<typename T>
  static
  typename enable_if<is_integral<T>::value, T>::type
  NaN() { return T(0); }

  template<typename T>
  static
  typename enable_if<is_float<T>::value, T>::type
  Inf() {
    return (std::numeric_limits<T>::has_infinity) ?
        std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();
  }

  template<typename T>
  static
  typename enable_if<is_integral<T>::value, T>::type
  Inf() { return std::numeric_limits<T>::max(); }
};
}  // namespace internal

// Constants
template<typename T>
class Constant {
 public:
  static const T nan;
  static const T inf;
};

template<typename T>
const T Constant<T>::nan = internal::ConstantHelper::NaN<T>();

template<typename T>
const T Constant<T>::inf = internal::ConstantHelper::Inf<T>();
}  // namespace sapien
#endif  // INCLUDE_SAPIEN_CONSTANTS_H_
