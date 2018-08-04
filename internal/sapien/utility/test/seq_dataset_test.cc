// Copyright 2018.

#include "sapien/utility/seq_dataset.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace sapien {
namespace internal {

using ::testing::ElementsAre;
using ::testing::DoubleEq;
using ::testing::FloatEq;

TEST(SeqDataset, InitializeDataset) {
  const int M = 3;  // n_samples
  const int N = 2;  // n_features
  double matrix[M*N] = {1, 2, 3, 4, 5, 6};
  double targets[M] = {-1, 1, -1};

  SeqDataset<double> dataset(M, N, matrix, targets);

  EXPECT_EQ(dataset.n_features, 2);
  EXPECT_EQ(dataset.n_samples, 3);
}

TEST(SeqDataset, AccessIndividualSampleWithDefaultWeights) {
  typedef SeqDataset<double>::Sample SampleType;

  const int M = 3;
  const int N = 2;
  double matrix[M*N] = {1, 2, 3, 4, 5, 6};
  double targets[M] = {-1, 1, -1};

  SeqDataset<double> dataset(M, N, matrix, targets);

  SampleType sample0 = dataset[0];
  SampleType sample1 = dataset[1];
  SampleType sample2 = dataset[2];

  // sample0 = (1, 2), -1, 1.0
  EXPECT_THAT(sample0.x[0], DoubleEq(1));
  EXPECT_THAT(sample0.x[1], DoubleEq(2));
  EXPECT_THAT(sample0.target, DoubleEq(-1));
  EXPECT_THAT(sample0.weight, DoubleEq(1.0));

  // sample1 = (3, 4), 1, 1.0
  EXPECT_THAT(sample1.x[0], DoubleEq(3));
  EXPECT_THAT(sample1.x[1], DoubleEq(4));
  EXPECT_THAT(sample1.target, DoubleEq(1));
  EXPECT_THAT(sample1.weight, DoubleEq(1.0));

  // sample2 = (5, 6), -1, 1.0
  EXPECT_THAT(sample2.x[0], DoubleEq(5));
  EXPECT_THAT(sample2.x[1], DoubleEq(6));
  EXPECT_THAT(sample2.target, DoubleEq(-1));
  EXPECT_THAT(sample2.weight, DoubleEq(1.0));
}

TEST(SeqDataset, AccessIndividualSampleWithCustomWeights) {
  const int M = 3;
  const int N = 2;
  float matrix[M*N] = {1, 2, 3, 4, 5, 6};
  float targets[M] = {-1, 1, -1};
  float weights[M] = {0.1, 0.2, 0.3};

  typedef SeqDataset<float>::Sample SampleType;
  SeqDataset<float> dataset(M, N, matrix, targets, weights);

    SampleType sample0 = dataset[0];
  SampleType sample1 = dataset[1];
  SampleType sample2 = dataset[2];

  // sample0 = (1, 2), -1, 1.0
  EXPECT_THAT(sample0.x[0], FloatEq(1));
  EXPECT_THAT(sample0.x[1], FloatEq(2));
  EXPECT_THAT(sample0.target, FloatEq(-1));
  EXPECT_THAT(sample0.weight, FloatEq(0.1));

  // sample1 = (3, 4), 1, 1.0
  EXPECT_THAT(sample1.x[0], FloatEq(3));
  EXPECT_THAT(sample1.x[1], FloatEq(4));
  EXPECT_THAT(sample1.target, FloatEq(1));
  EXPECT_THAT(sample1.weight, FloatEq(0.2));

  // sample2 = (5, 6), -1, 1.0
  EXPECT_THAT(sample2.x[0], FloatEq(5));
  EXPECT_THAT(sample2.x[1], FloatEq(6));
  EXPECT_THAT(sample2.target, FloatEq(-1));
  EXPECT_THAT(sample2.weight, FloatEq(0.3));
}
}  // namespace internal
}  // namespace sapien
