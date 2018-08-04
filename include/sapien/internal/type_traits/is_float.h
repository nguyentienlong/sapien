// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com
//
// Is type T a floating-point type?

#ifndef INCLUDE_SAPIEN_INTERNAL_TYPE_TRAITS_IS_FLOAT_H_
#define INCLUDE_SAPIEN_INTERNAL_TYPE_TRAITS_IS_FLOAT_H_

namespace sapien {
namespace internal {

template<typename T> struct is_float {
  static const bool value = false;
};

template<typename T> struct is_float<const T>
    : public is_float<T>{};

template<typename T> struct is_float<volatile const T>
    : public is_float<T>{};

template <typename T> struct is_float<volatile T> :
      public is_float<T>{};

template<> struct is_float<float> {
  static const bool value = true;
};

template<> struct is_float<double> {
  static const bool value = true;
};

template<> struct is_float<long double> {
  static const bool value = true;
};

}  // namespace internal
}  // namespace sapien
#endif  // INCLUDE_SAPIEN_INTERNAL_TYPE_TRAITS_IS_FLOAT_H_
