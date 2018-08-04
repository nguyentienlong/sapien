// Copyright 2018.

#include <string>

#include "sapien/utility/stringprintf.h"
#include "gtest/gtest.h"

namespace sapien {
namespace internal {

TEST(StringPrintf, Test1) {
  std::string result = StringPrintf("%d %s", 10, "hello");
  EXPECT_EQ(result, "10 hello");
}
}  // namespace internal
}  // namespace sapien

