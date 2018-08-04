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

TEST(SGDTypes, ConvertLossTypeToString) {
  EXPECT_THAT(LossTypeToString(MODIFIED_HUBER_LOSS),
              StrEq("MODIFIED_HUBER_LOSS"));
  EXPECT_THAT(LossTypeToString(HINGE_LOSS), StrEq("HINGE_LOSS"));
  EXPECT_THAT(LossTypeToString(SQUARED_HINGE_LOSS),
              StrEq("SQUARED_HINGE_LOSS"));
  EXPECT_THAT(LossTypeToString(PERCEPTRON_LOSS), StrEq("PERCEPTRON_LOSS"));
  EXPECT_THAT(LossTypeToString(LOG_LOSS), StrEq("LOG_LOSS"));
  EXPECT_THAT(LossTypeToString(SQUARED_LOSS), StrEq("SQUARED_LOSS"));
  EXPECT_THAT(LossTypeToString(HUBER_LOSS), StrEq("HUBER_LOSS"));
  EXPECT_THAT(LossTypeToString(EPSILON_INSENSITIVE_LOSS),
              StrEq("EPSILON_INSENSITIVE_LOSS"));
  EXPECT_THAT(LossTypeToString(SQUARED_EPSILON_INSENSITIVE_LOSS),
              StrEq("SQUARED_EPSILON_INSENSITIVE_LOSS"));
}

TEST(SGDTypes, ConvertStringToLossType) {
  LossType type;

  StringToLossType("modified_huber_loss", &type);
  EXPECT_EQ(type, MODIFIED_HUBER_LOSS);

  StringToLossType("hinge_loss", &type);
  EXPECT_EQ(type, HINGE_LOSS);

  StringToLossType("squared_hinge_loss", &type);
  EXPECT_EQ(type, SQUARED_HINGE_LOSS);

  EXPECT_FALSE(StringToLossType("abc", &type));
}

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

TEST(SGDTypes, CheckIfValidLossType) {
  EXPECT_TRUE(IsValidLossType(CLASSIFICATION_MODEL, MODIFIED_HUBER_LOSS));
  EXPECT_TRUE(IsValidLossType(CLASSIFICATION_MODEL, HINGE_LOSS));
  EXPECT_TRUE(IsValidLossType(CLASSIFICATION_MODEL, SQUARED_HINGE_LOSS));
  EXPECT_TRUE(IsValidLossType(CLASSIFICATION_MODEL, PERCEPTRON_LOSS));
  EXPECT_TRUE(IsValidLossType(CLASSIFICATION_MODEL, LOG_LOSS));

  EXPECT_FALSE(IsValidLossType(CLASSIFICATION_MODEL, SQUARED_LOSS));
  EXPECT_FALSE(IsValidLossType(CLASSIFICATION_MODEL, HUBER_LOSS));
  EXPECT_FALSE(IsValidLossType(CLASSIFICATION_MODEL,
                               EPSILON_INSENSITIVE_LOSS));
  EXPECT_FALSE(IsValidLossType(CLASSIFICATION_MODEL,
                               SQUARED_EPSILON_INSENSITIVE_LOSS));

  EXPECT_TRUE(IsValidLossType(REGRESSION_MODEL, SQUARED_LOSS));
  EXPECT_TRUE(IsValidLossType(REGRESSION_MODEL, HUBER_LOSS));
  EXPECT_TRUE(IsValidLossType(REGRESSION_MODEL,
                               EPSILON_INSENSITIVE_LOSS));
  EXPECT_TRUE(IsValidLossType(REGRESSION_MODEL,
                               SQUARED_EPSILON_INSENSITIVE_LOSS));
}

// Loss function test

using ::testing::DoubleEq;

TEST(SGDTypes, CheckHingeLoss) {
  std::shared_ptr<LossFunctor> loss;
  loss = LossTypeToLossFunctor(HINGE_LOSS, 1.0);

  double y = -1;
  double p = -0.23273782;

  double true_l = 1.0 - p * y;
  double true_dl = -y;

  EXPECT_THAT(loss->Loss(p, y), DoubleEq(true_l));
  EXPECT_THAT(loss->DLoss(p, y), DoubleEq(true_dl));
}

TEST(SGDTypes, CheckModifiedHuberLoss) {
  std::shared_ptr<LossFunctor> loss;
  loss = LossTypeToLossFunctor(MODIFIED_HUBER_LOSS);

  double y = -1;
  double p = 0.25;
  double z = p * y;
  double true_l = (1.0 - z) * (1.0 - z);
  double true_dl = 2.0 * (1.0 - z) * (-y);

  EXPECT_THAT(loss->Loss(p, y), DoubleEq(true_l));
  EXPECT_THAT(loss->DLoss(p, y), DoubleEq(true_dl));

  y = -1;
  p = 5.4372382834981;
  z = p * y;

  true_l = -4.0 * z;
  true_dl = -4.0 * y;

  EXPECT_THAT(loss->Loss(p, y), DoubleEq(true_l));
  EXPECT_THAT(loss->DLoss(p, y), DoubleEq(true_dl));

  y = 1.0;
  p = 1.25;

  EXPECT_THAT(loss->Loss(p, y), DoubleEq(0.0));
  EXPECT_THAT(loss->DLoss(p, y), DoubleEq(0.0));
}

}  // namespace sgd
}  // namespace sapien
