// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com

#ifndef INCLUDE_SAPIEN_SGD_TYPES_H_
#define INCLUDE_SAPIEN_SGD_TYPES_H_

#include <string>
#include <memory>

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

// SGD loss type.
//
// In the following, we'll use these 'variables' y, p, w to denote the
// ground truth outcome/target, the predicted value and the weight vector,
// respectively. Note that the predicted value w.r.t the weight vector and
// sample vector x is computed as p = w.dot(x) + intercept.
//
// 1. MODIFIED_HUBER: Modified Huber loss for binary classification with
//    y in {-1, 1}.
//
//                  |max(0, 1- yp)^2 for yp >= -1
//    loss(p, y) =  |
//                  |-4yp otherwise
//
//    See [1] for more details.
//
// 2. HINGE: Hinge loss for binary classification with y in {-1, 1}.
//
//    loss(p, y) = max(0, threshold - p * y).
//
//    Two specicial cases for the threshold value:
//      - threshold = 1.0: SVM loss.
//      - threshold = 0.0: Perceptron loss.
//
//    See [2] for more details.
//
// 3. SQUARED_HINGE: the squared version of hinge loss used for binary
//    classification with y in {-1, 1}.
//
//    loss(p, y) = max(0, param - py)^2.
//
//    See [2] for more details.
//
// 4. PERCEPTRON: equivalent to hinge(0.0), i.e:
//
//    loss(p, y) = max(0, -p*y).
//
//    This loss function is used by the Perceptron algorithm (see [3]).
//
// 5. LOG_LOSS: logistic regression loss for binary classification with
//    y in {-1, 1}.
//
//    loss(p, y) = ln(1 + e^(-py)).
//
// 6. SQUARED_LOSS: squared loss used in linear regression
//
//    loss(p, y) = 0.5 * (p - y)^2 for a single sample x.
//
// 7. HUBER: Huber loss for regression model (see [1]).
//
// 8. EPSILON_INSENSITIVE: epsilon insensitive loss for regression model.
//
//    loss(p, y) = max(0, |y - p| - epsilon).
//
// 9. SQUARED_EPSILON_INSENSITIVE: squared version of epsilone insensitive
//    loss.
//
//    loss(p, y) = max(0, |y - p| - epsilon)^2.
//
// [1] - https://en.wikipedia.org/wiki/Huber_loss.
// [2] - https://en.wikipedia.org/wiki/Hinge_loss.
// [3] - https://en.wikipedia.org/wiki/Perceptron.
enum LossType {
  // Classification loss type
  MODIFIED_HUBER_LOSS,
  HINGE_LOSS,
  SQUARED_HINGE_LOSS,
  PERCEPTRON_LOSS,
  LOG_LOSS,

  // Regression loss type
  SQUARED_LOSS,
  HUBER_LOSS,
  EPSILON_INSENSITIVE_LOSS,
  SQUARED_EPSILON_INSENSITIVE_LOSS
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

// Class weight type, i.e how to set weight associated with each class
// in classification model?
//
// 1. CLASS_WEIGHT_UNIFORM: all weights are set to 1.0.
// 2. CLASS_WEIGHT_BALANCED: the weight of class i is computed as
//    weight(i) = (n_samples) / (n_classes * (n_class_i in y)).
// 3. CLASS_WEIGHT_MAP: User provided a std::map of the form:
//    {class: weight}.
enum ClassWeightType {
  CLASS_WEIGHT_UNIFORM,
  CLASS_WEIGHT_BALANCED,
  CLASS_WEIGHT_MAP
};

// Abstract base class for all loss functions
struct SAPIEN_EXPORT LossFunctor {
  // Evaluate the loss at the prediction p w.r.t the ground truth y.
  virtual double Loss(const double p, const double y) const = 0;

  // Evaluate the first derivative of the loss function w.r.t
  // the prediction p.
  virtual double DLoss(const double p, const double y) const = 0;

  virtual ~LossFunctor() {}
};

// Concrete loss functor --------------------------------------------------

// Moified huber for binary classification with y in {-1, 1}.
//               |max(0, 1- yp)^2 for yp >= -1
// loss(p, y) =  |
//               |-4yp otherwise
struct SAPIEN_EXPORT ModifiedHuberLoss : public LossFunctor {
  // Evaluate the loss of the prediction p w.r.t the ground truth y
  double Loss(const double p, const double y) const;

  // Evaluate the first derivative of the loss function w.r.t
  // the prediction p.
  double DLoss(const double p, const double y) const;
};

// Hinge loss for binary classification tasks with y in {-1, 1}
// loss(p, y) = max(0, param - p * y).
struct SAPIEN_EXPORT HingeLoss : public LossFunctor {
  const double param;

  HingeLoss();
  explicit HingeLoss(const double param);

  double Loss(const double p, const double y) const;
  double DLoss(const double p, const double y) const;
};

// Squared Hinge loss for binary classification tasks with y in {-1, 1}
// loss(p, y) = max(0, param - py)^2
struct SAPIEN_EXPORT SquaredHingeLoss : public LossFunctor {
  const double param;

  SquaredHingeLoss();
  explicit SquaredHingeLoss(const double param);

  double Loss(const double p, const double y) const;
  double DLoss(const double p, const double y) const;
};

// Logistic regression loss for binary classification with y in {-1, 1}.
// loss(p, y) = ln(1 + e^(-py)).
struct SAPIEN_EXPORT LogLoss : public LossFunctor {
  double Loss(const double p, const double y) const;
  double DLoss(const double p, const double y) const;
};

// Squared loss traditional used in linear regression
// loss(p, y) = 0.5 * (p - y)^2 for a single example.
struct SAPIEN_EXPORT SquaredLoss : public LossFunctor {
  double Loss(const double p, const double y) const;
  double DLoss(const double p, const double y) const;
};

// Huber regression loss.
struct SAPIEN_EXPORT HuberLoss : public LossFunctor {
  const double param;

  HuberLoss();
  explicit HuberLoss(const double param);

  double Loss(const double p, const double y) const;
  double DLoss(const double p, const double y) const;
};

// Epsilon insensitive for regression
// loss = max(0, |y - p| - epsilon)
struct SAPIEN_EXPORT EpsilonInsensitiveLoss : public LossFunctor {
  const double epsilon;

  EpsilonInsensitiveLoss();
  explicit EpsilonInsensitiveLoss(const double epsilon);

  double Loss(const double p, const double y) const;
  double DLoss(const double p, const double y) const;
};

// Squared epsilon insensitive for regression
// loss = max(0, |y - p| - epsilon)^2.
struct SquaredEpsilonInsensitiveLoss : public LossFunctor {
  const double epsilon;

  SquaredEpsilonInsensitiveLoss();
  explicit SquaredEpsilonInsensitiveLoss(const double epsilon);

  double Loss(const double p, const double y) const;
  double DLoss(const double p, const double y) const;
};


// Utility functions & constants -------------------------------------------

const double kMaxDerivativeLoss = 1e12;

SAPIEN_EXPORT
const char* LossTypeToString(const LossType loss_type);

SAPIEN_EXPORT
bool StringToLossType(const std::string& loss_string,
                      LossType* loss_type);

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

SAPIEN_EXPORT
bool IsValidLossType(const ModelType model_type, const LossType loss_type);

SAPIEN_EXPORT
std::shared_ptr<LossFunctor>
LossTypeToLossFunctor(const LossType type, const double param = 1.0);
}  // namespace sgd
}  // namespace sapien
#endif  // INCLUDE_SAPIEN_SGD_TYPES_H_

