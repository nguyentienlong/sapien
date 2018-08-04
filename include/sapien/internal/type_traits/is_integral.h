// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// Is type T an integral type.

#ifndef INCLUDE_SAPIEN_INTERNAL_TYPE_TRAITS_IS_INTEGRAL_H_
#define INCLUDE_SAPIEN_INTERNAL_TYPE_TRAITS_IS_INTEGRAL_H_

#include "sapien/internal/port.h"

namespace sapien {
namespace internal {

#define IT_IS_INTEGRAL(type) template<> struct is_integral<type> {\
    static const bool value = true; }

template<typename T> struct is_integral {
  static const bool value = false;
};

template<typename T> struct is_integral<const T>
    : public is_integral<T> {};

template<typename T> struct is_integral<volatile const T>
    : public is_integral<T> {};

template<typename T> struct is_integral<volatile T>
    : public is_integral<T> {};

IT_IS_INTEGRAL(int8_t);
IT_IS_INTEGRAL(uint8_t);
IT_IS_INTEGRAL(int16_t);
IT_IS_INTEGRAL(uint16_t);
IT_IS_INTEGRAL(int32_t);
IT_IS_INTEGRAL(uint32_t);
IT_IS_INTEGRAL(int64_t);
IT_IS_INTEGRAL(uint64_t);

#undef IT_IS_INTEGRAL

}  // namespace internal
}  // namespace sapien
#endif  // INCLUDE_SAPIEN_INTERNAL_TYPE_TRAITS_IS_INTEGRAL_H_
