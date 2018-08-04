// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com
//
// Is type T a pointer type.

#ifndef INCLUDE_SAPIEN_INTERNAL_TYPE_TRAITS_IS_POINTER_H_
#define INCLUDE_SAPIEN_INTERNAL_TYPE_TRAITS_IS_POINTER_H_

namespace sapien {
namespace internal {

template<typename T> struct is_pointer {
  static const bool value = false;
};

template<typename T> struct is_pointer<T*> {
  static const bool value = true;
};

template<typename T> struct is_pointer<T*const> {
  static const bool value = true;
};

template<typename T> struct is_pointer<T*const volatile> {
  static const bool value = true;
};

template<typename T> struct is_pointer<T*volatile> {
  static const bool value = true;
};

}  // namespace internal
}  // namespace sapien
#endif  // INCLUDE_SAPIEN_INTERNAL_TYPE_TRAITS_IS_POINTER_H_

