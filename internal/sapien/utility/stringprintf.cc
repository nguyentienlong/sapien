// Copyright 2018.
//
// Author: Sanjay Ghemawat.

#include <cerrno>
#include <cstdarg>  // For va_list and related operations
#include <cstdio>   // MSVC requires this for _vsnprintf
#include <string>
#include <vector>

#include "sapien/utility/stringprintf.h"

namespace sapien {
namespace internal {

using std::string;

// va_copy() was defined in the C99 standard.  However, it did not appear in the
// C++ standard until C++11.  This means that if Ceres is being compiled with a
// strict pre-C++11 standard (e.g. -std=c++03), va_copy() will NOT be defined,
// as we are using the C++ compiler (it would however be defined if we were
// using the C compiler).  Note however that both GCC & Clang will in fact
// define va_copy() when compiling for C++ if the C++ standard is not explicitly
// specified (i.e. no -std=c++<XX> arg), even though it should not strictly be
// defined unless -std=c++11 (or greater) was passed.
#if !defined(va_copy)
#if defined (__GNUC__)
// On GCC/Clang, if va_copy() is not defined (C++ standard < C++11 explicitly
// specified), use the internal __va_copy() version, which should be present
// in even very old GCC versions.
#define va_copy(d, s) __va_copy(d, s)
#else
// Some older versions of MSVC do not have va_copy(), in which case define it.
// Although this is required for older MSVC versions, it should also work for
// other non-GCC/Clang compilers which also do not defined va_copy().
#define va_copy(d, s) ((d) = (s))
#endif  // defined (__GNUC__)
#endif  // !defined(va_copy)

void StringAppendV(string* dst, const char* format, va_list ap) {
  // First try with a small fixed size buffer
  char space[1024];

  // It's possible for methods that use a va_list to invalidate
  // the data in it upon use.  The fix is to make a copy
  // of the structure before using it and use that copy instead.
  va_list backup_ap;
  va_copy(backup_ap, ap);
  int result = vsnprintf(space, sizeof(space), format, backup_ap);
  va_end(backup_ap);

  if (result < sizeof(space)) {
    if (result >= 0) {
      // Normal case -- everything fit.
      dst->append(space, result);
      return;
    }

#if defined (_MSC_VER)
    // Error or MSVC running out of space.  MSVC 8.0 and higher
    // can be asked about space needed with the special idiom below:
    va_copy(backup_ap, ap);
    result = vsnprintf(NULL, 0, format, backup_ap);
    va_end(backup_ap);
#endif

    if (result < 0) {
      // Just an error.
      return;
    }
  }

  // Increase the buffer size to the size requested by vsnprintf,
  // plus one for the closing \0.
  int length = result+1;
  char* buf = new char[length];

  // Restore the va_list before we use it again
  va_copy(backup_ap, ap);
  result = vsnprintf(buf, length, format, backup_ap);
  va_end(backup_ap);

  if (result >= 0 && result < length) {
    // It fit
    dst->append(buf, result);
  }
  delete[] buf;
}


string StringPrintf(const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  string result;
  StringAppendV(&result, format, ap);
  va_end(ap);
  return result;
}

const string& SStringPrintf(string* dst, const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  dst->clear();
  StringAppendV(dst, format, ap);
  va_end(ap);
  return *dst;
}

void StringAppendF(string* dst, const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  StringAppendV(dst, format, ap);
  va_end(ap);
}

}  // namespace internal
}  // namespace sapien
