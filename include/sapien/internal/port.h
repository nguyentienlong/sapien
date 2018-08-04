// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com

#ifndef INCLUDE_SAPIEN_INTERNAL_PORT_H_
#define INCLUDE_SAPIEN_INTERNAL_PORT_H_

#include "sapien/internal/config.h"

#if defined(_MSC_VER) && defined(SAPIEN_BUILDING_SHARED_LIBRARY)
# define SAPIEN_EXPORT __declspec(dllexport)
#elif defined(_MSC_VER) && defined(SAPIEN_USING_SHARED_LIBRARY)
# define SAPIEN_EXPORT __declspec(dllimport)
#else
# define SAPIEN_EXPORT
#endif

// Portable integral types -------------------------------------------------
#if defined(__cplusplus)   /* C++ */
#  if defined(_MSC_VER) && _MSC_VER < 1600 /* msvc 2010 */
namespace sapien {
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef signed short int16_t;      // NOLINT
typedef unsigned short uint16_t;   // NOLINT
typedef signed int int32_t;
typedef unsigned int uint32_t;
typedef signed __int64 int64_t;
typedef unsigned __int64 uint64_t;
}
#  elif defined(_MSC_VER) || __cplusplus >= 201103L
#include <cstdint>
namespace sapien {
using std::int8_t;
using std::uint8_t;
using std::int16_t;
using std::uint16_t;
using std::int32_t;
using std::uint32_t;
using std::int64_t;
using std::uint64_t;
}
#  else
#include <stdint.h>
namespace sapien {
typedef ::int8_t int8_t;
typedef ::uint8_t uint8_t;
typedef ::int16_t int16_t;
typedef ::uint16_t uint16_t;
typedef ::int32_t int32_t;
typedef ::uint32_t uint32_t;
typedef ::int64_t int64_t;
typedef ::uint64_t uint64_t;
}
#  endif
#else    /* pure C */
#include <stdint.h>
#endif

#endif  // INCLUDE_SAPIEN_INTERNAL_PORT_H_
