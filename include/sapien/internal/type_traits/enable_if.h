// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com
//
// Alternative to C++11 std::enable_if (just in case we do not have C++11).

#ifndef INCLUDE_SAPIEN_INTERNAL_TYPE_TRAITS_ENABLE_IF_H_
#define INCLUDE_SAPIEN_INTERNAL_TYPE_TRAITS_ENABLE_IF_H_

namespace sapien {
namespace internal {

template<bool, typename T> struct enable_if {};
template<typename T> struct enable_if<true, T> {
  typedef T type;
};
}  // namespace internal
}  // namespace sapien
#endif  // INCLUDE_SAPIEN_INTERNAL_TYPE_TRAITS_ENABLE_IF_H_

