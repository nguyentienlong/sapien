// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com

#ifndef INTERNAL_SAPIEN_UTILITY_UNIQUE_H_
#define INTERNAL_SAPIEN_UTILITY_UNIQUE_H_

#include <cstddef>  // size_t
#include <cstring>
#include <memory>
#include <algorithm>

namespace sapien {
namespace internal {

// Default unique comparator
template<typename T>
struct default_unique_comparator {
  bool operator()(const T& a, const T& b) const {
    return (a < b);
  }
};

// Returns a unique_ptr to an array of unique elements from the input
// array v sorted in ascending order, and set unique_count to the number
// of unique elements from v.
template<typename T, typename Comparator>
std::unique_ptr<T[]> UniqueElements(const size_t n,
                                    const T* v,
                                    size_t* unique_count,
                                    const Comparator& comp =
                                    default_unique_comparator()) {
  // Since we donot allow to modify v, we need to make a copy of v.
  std::unique_ptr<T[]> copied_v(new T[n]);
  std::memcpy(copied_v.get(), v, n * sizeof(T));

  std::sort(copied_v.get(), copied_v.get() + n, comp);

  // Number of unique elements.

  *unique_count = 1;
  for (size_t i = 1; i < n; ++i) {
    const T a = copied_v[i - 1];
    const T b = copied_v[i];
    const T diff = a - b;
    if (diff != T(0)) { ++(*unique_count); }
  }

  // Extract unique elements
  std::unique_ptr<T[]> ret(new T[*unique_count]);
  T* ptr = ret.get();
  *ptr = copied_v[0];
  ptr++;

  for (size_t i = 1; i < n; ++i) {
    const T a = copied_v[i - 1];
    const T b = copied_v[i];
    const T diff = a - b;
    if (diff != T(0)) {
      *ptr = b;
      ptr++;
    }
  }

  return ret;
}
}  // namespace internal
}  // namespace sapien
#endif  // INTERNAL_SAPIEN_UTILITY_UNIQUE_H_
