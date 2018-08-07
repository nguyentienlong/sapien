// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com

#include <memory>

#include "sapien/sgd/types.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace sapien {
namespace sgd {

using ::testing::StrEq;

TEST(SGDTypes, ConvertPenaltyTypeToString) {
  EXPECT_THAT(PenaltyTypeToString(NO_PENALTY), StrEq("NO_PENALTY"));
  EXPECT_THAT(PenaltyTypeToString(L1_PENALTY), StrEq("L1_PENALTY"));
  EXPECT_THAT(PenaltyTypeToString(L2_PENALTY), StrEq("L2_PENALTY"));
  EXPECT_THAT(PenaltyTypeToString(ELASTIC_NET_PENALTY),
              StrEq("ELASTIC_NET_PENALTY"));
}

TEST(SGDTypes, ConvertStringToPenaltyType) {
  PenaltyType type;

  EXPECT_TRUE(StringToPenaltyType("no_penalty", &type));
  EXPECT_EQ(type, NO_PENALTY);

  EXPECT_TRUE(StringToPenaltyType("l1_penalty", &type));
  EXPECT_EQ(type, L1_PENALTY);

  EXPECT_TRUE(StringToPenaltyType("l2_penalty", &type));
  EXPECT_EQ(type, L2_PENALTY);

  EXPECT_TRUE(StringToPenaltyType("elastic_net_penalty", &type));
  EXPECT_EQ(type, ELASTIC_NET_PENALTY);

  EXPECT_FALSE(StringToPenaltyType("not_exist", &type));
}

}  // namespace sgd
}  // namespace sapien
