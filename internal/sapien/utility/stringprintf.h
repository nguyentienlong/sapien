// Copyright 2018.
//
// Author: Sanjay Ghemawat (Google), see [1] for more details.
//
// Printf variants that place thier output in a C++ string.
//
// Usage:
//      string result = StringPrintf("%d %s\n", 10, "hello");
//      SStringPrintf(&result, "%d %s\n", 10, "hello");
//      StringAppendF(&result, "%d %s\n", 20, "there");
//
// [1] - https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/stringprintf.h.

#ifndef INTERNAL_SAPIEN_UTILITY_STRINGPRINTF_H_
#define INTERNAL_SAPIEN_UTILITY_STRINGPRINTF_H_

#include <cstdarg>
#include <string>

#include "sapien/internal/port.h"

namespace sapien {
namespace internal {

#if (defined(__GNUC__) || defined(__clang__))
// Tell the compiler to do printf format string checking if the compiler
// supports it; see the 'format' attribute in
// <http://gcc.gnu.org/onlinedocs/gcc-4.3.0/gcc/Function-Attributes.html>.
//
// N.B.: As the GCC manual states, "[s]ince non-static C++ methods
// have an implicit 'this' argument, the arguments of such methods
// should be counted from two, not one."
#define SAPIEN_PRINTF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__printf__, string_index, first_to_check)))

#define SAPIEN_SCANF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__scanf__, string_index, first_to_check)))
#else
#define SAPIEN_PRINTF_ATTRIBUTE(string_index, first_to_check)
#endif

// Return a C++ string.
extern std::string StringPrintf(const char* format, ...)
    // Tell the compiler to do printf format string checking.
    SAPIEN_PRINTF_ATTRIBUTE(1, 2);

// Store result into a supplied string and return it.
extern const std::string& SStringPrintf(std::string* dst, const char* format,
                                        ...)
    // Tell the compiler to do printf format string checking.
    SAPIEN_PRINTF_ATTRIBUTE(2, 3);

// Append result to a supplied string.
extern void StringAppendF(std::string* dst, const char* format, ...)
    // Tell the compiler to do printf format string checking.
    SAPIEN_PRINTF_ATTRIBUTE(2, 3);

// Lower-level routine that takes a va_list and appends to a specified string.
// All other routines are just convenience wrappers around it.
extern void StringAppendV(std::string* dst, const char* format, va_list ap);

#undef SAPIEN_PRINTF_ATTRIBUTE

}  // namespace internal
}  // namespace sapien
#endif  // INTERNAL_SAPIEN_UTILITY_STRINGPRINTF_H_

