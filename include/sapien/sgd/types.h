// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com

#ifndef INCLUDE_SAPIEN_SGD_TYPES_H_
#define INCLUDE_SAPIEN_SGD_TYPES_H_

#include <string>

#include "sapien/internal/port.h"

namespace sapien {
namespace sgd {
// Logging options.
// SILENT: No logging at all.
// PER_EPOCH: Log the output of each epoch (each passing over
// the dataset) to either STDERR or STDOUT depending on the options
// provided by user.
enum LoggingType {
  SILENT,
  PER_EPOCH
};

// SGD model type
enum ModelType {
  UNDEFINED,
  CLASSIFICATION_MODEL,
  REGRESSION_MODEL
};

// SGD penalty type.
//
// 1. NO_PENALTY: No penalty (a.k.a regularization) at all.
//
// 2. L1_PENALTY: the L1 regularization.
//
// 3. L2_PENALTY: the L2 regularization.
//
// 4. ELASTIC_NET_PENALTY: linearly combined the L1 and L2 penalties
//    (see [1]).
//
// [1] - https://en.wikipedia.org/wiki/Elastic_net_regularization.
enum PenaltyType {
  NO_PENALTY,
  L1_PENALTY,
  L2_PENALTY,
  ELASTIC_NET_PENALTY
};

// SGD learning rate type.
//
// 1. LEARNING_RATE_CONSTANT:
//      eta = eta0 for all samples, all epochs.
//
// 2. LEARNING_RATE_OPTIMAL:
//      eta = 1.0 / (alpha * (t + t0)), in which:
//
//        - alpha is the regularization strength.
//        - t0 is the optimal initial value.
//        - t is the iteration count.
//
// 3. LEARNING_RATE_INVERSE_SCALING:
//      eta = eta0 / pow(t, inverse_scaling_exp), where:
//
//        - inverse_scaling_exp is some scalar.
//        - t is the iteration count.
//
// 4. PASSIVE_AGRESSIVE_1, PASSIVE_AGRESSIVE_2, PASSIVE_AGRESSIVE_3:
//    Technically speaking, when learning rate type is set to one of these
//    values, the optimization algorithm is no longer the plain gradient
//    descent (which is typically used to train SGD), but instead the
//    technique called "Oline Passive-Agressive Algorithm" is used to
//    update the weight vector.
//
//    See [1] for more details about this technique.
//
// [1] - http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf.
enum LearningRateType {
  LEARNING_RATE_CONSTANT,
  LEARNING_RATE_OPTIMAL,
  LEARNING_RATE_INVERSE_SCALING,
  PASSIVE_AGRESSIVE,
  PASSIVE_AGRESSIVE_1,
  PASSIVE_AGRESSIVE_2
};

// Utility functions & constants -------------------------------------------

const double kMaxDerivativeLoss = 1e12;

SAPIEN_EXPORT
const char* PenaltyTypeToString(const PenaltyType penalty_type);

SAPIEN_EXPORT
bool StringToPenaltyType(const std::string& penalty_string,
                         PenaltyType* penalty_type);

SAPIEN_EXPORT
const char* LearningRateTypeToString(const LearningRateType learning_rate_type);

SAPIEN_EXPORT
bool StringToLearningRateType(const std::string& learning_rate_string,
                              LearningRateType* learning_rate);
}  // namespace sgd
}  // namespace sapien
#endif  // INCLUDE_SAPIEN_SGD_TYPES_H_

