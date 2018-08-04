// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com

#include <cmath>
#include <algorithm>

#include "sapien/sgd/types.h"

namespace sapien {
namespace sgd {

#define CASESTR(x) case x: return #x
#define STRENUM(x) if (value == #x) {*type = x; return true; }

static void UpperCase(std::string* input) {
  std::transform(input->begin(), input->end(), input->begin(), std::toupper);
}

// Concrete loss function ------------------------------------------------

// Modified huber loss

double ModifiedHuberLoss::Loss(const double p, const double y) const {
  double z = p * y;
  if (z >= 1.0) {
    return 0.0;
  } else if (z >= -1.0) {
    return (1 - z) * (1 - z);
  } else {
    return -4.0 * z;
  }
}

double ModifiedHuberLoss::DLoss(const double p, const double y) const {
  double z = p * y;
  if (z >= 1.0) {
    return 0.0;
  } else if (z >= -1.0) {
    return -2.0 * y * (1 - z);
  } else {
    return -4.0 * y;
  }
}

// Hinge loss

HingeLoss::HingeLoss() : param(1.0) {}
HingeLoss::HingeLoss(const double param) : param(param) {}

double HingeLoss::Loss(const double p, const double y) const {
  double z = p * y;
  if (z <= param) {
    return param - z;
  }
  return 0.0;
}

double HingeLoss::DLoss(const double p, const double y) const {
  double z = p * y;
  if (z <= param) {
    return -y;
  }
  return 0.0;
}

// Squared hinge loss

SquaredHingeLoss::SquaredHingeLoss() : param(1.0) {}
SquaredHingeLoss::SquaredHingeLoss(const double param) : param(param) {}

double SquaredHingeLoss::Loss(const double p, const double y) const {
  double z = param - p * y;
  if (z >= 0.0) {
    return z * z;
  }
  return 0.0;
}

double SquaredHingeLoss::DLoss(const double p, const double y) const {
  double z = param - p * y;
  if (z >= 0.0) {
    return 2.0 * (-y) * z;
  }
  return 0.0;
}

// Logistic regression loss

double LogLoss::Loss(const double p, const double y) const {
  double z = p * y;
  if (z > 18) {
    return std::exp(-z);
  } else if (z < -18) {
    return -z;
  } else {
    return std::log(1.0 + std::exp(-z));
  }
}

double LogLoss::DLoss(const double p, const double y) const {
  double z = p * y;
  if (z > 18.0) {
    return (-y) * std::exp(-z);
  } else if (z < -18.0) {
    return -y;
  } else {
    return (-y) / ( 1.0 + std::exp(z));
  }
}

// Squared loss

double SquaredLoss::Loss(const double p, const double y) const {
  return 0.5 * (p - y) * (p - y);
}

double SquaredLoss::DLoss(const double p, const double y) const {
  return (p - y);
}

// Huber loss

HuberLoss::HuberLoss() : param(0.1) {}
HuberLoss::HuberLoss(const double param) : param(param) {}

double HuberLoss::Loss(const double p, const double y) const {
  double r = p - y;
  double abs_r = std::fabs(r);
  if (abs_r <= param) {
    return 0.5 * r * r;
  } else {
    return (param * abs_r) - (0.5 * param * param);
  }
}

double HuberLoss::DLoss(const double p, const double y) const {
  double r = p - y;
  double abs_r = std::fabs(r);
  if (abs_r <= param) {
    return r;
  } else if (r >= 0.0) {
    return param;
  } else {
    return -param;
  }
}

// Epsilon insensitive loss

EpsilonInsensitiveLoss::EpsilonInsensitiveLoss() : epsilon(0.0) {}
EpsilonInsensitiveLoss::EpsilonInsensitiveLoss(const double epsilon)
    : epsilon(epsilon) {}

double EpsilonInsensitiveLoss::Loss(const double p, const double y) const {
  double ret = std::fabs(y - p) - epsilon;
  return (ret > 0.0) ? ret : 0.0;
}

double EpsilonInsensitiveLoss::DLoss(const double p, const double y) const {
  if ((y - p) > epsilon) {
    return -1.0;
  } else if ((p - y) > epsilon) {
    return 1.0;
  } else {
    return 0.0;
  }
}

// Squared epsilon insensitive loss

SquaredEpsilonInsensitiveLoss::
SquaredEpsilonInsensitiveLoss() : epsilon(0.0) {}
SquaredEpsilonInsensitiveLoss::
SquaredEpsilonInsensitiveLoss(const double epsilon) : epsilon(epsilon) {}

double
SquaredEpsilonInsensitiveLoss::Loss(const double p, const double y) const {
  double ret = std::fabs(y - p) - epsilon;
  return (ret > 0.0) ? ret * ret : 0.0;
}

double
SquaredEpsilonInsensitiveLoss::DLoss(const double p, const double y) const {
  double z = y - p;
  if (z > epsilon) {
    return -2.0 * (z - epsilon);
  } else if (z < -epsilon) {
    return -2.0 * (z + epsilon);
  } else {
    return 0.0;
  }
}

// Utility functions -------------------------------------------------------

const char* LossTypeToString(const LossType type) {
  switch (type) {
    CASESTR(MODIFIED_HUBER_LOSS);
    CASESTR(HINGE_LOSS);
    CASESTR(SQUARED_HINGE_LOSS);
    CASESTR(PERCEPTRON_LOSS);
    CASESTR(LOG_LOSS);
    CASESTR(SQUARED_LOSS);
    CASESTR(HUBER_LOSS);
    CASESTR(EPSILON_INSENSITIVE_LOSS);
    CASESTR(SQUARED_EPSILON_INSENSITIVE_LOSS);
    default:
      return "UNKNOWN";
  }
}

bool StringToLossType(const std::string& s , LossType* type) {
  std::string value(s);
  UpperCase(&value);
  STRENUM(MODIFIED_HUBER_LOSS);
  STRENUM(HINGE_LOSS);
  STRENUM(SQUARED_HINGE_LOSS);
  STRENUM(PERCEPTRON_LOSS);
  STRENUM(LOG_LOSS);
  STRENUM(SQUARED_LOSS);
  STRENUM(HUBER_LOSS);
  STRENUM(EPSILON_INSENSITIVE_LOSS);
  STRENUM(SQUARED_EPSILON_INSENSITIVE_LOSS);
  return false;
}

const char* PenaltyTypeToString(const PenaltyType type) {
  switch (type) {
    CASESTR(NO_PENALTY);
    CASESTR(L1_PENALTY);
    CASESTR(L2_PENALTY);
    CASESTR(ELASTIC_NET_PENALTY);
    default:
      return "UNKNOWN";
  }
}

bool StringToPenaltyType(const std::string& s, PenaltyType* type) {
  std::string value(s);
  UpperCase(&value);
  STRENUM(NO_PENALTY);
  STRENUM(L1_PENALTY);
  STRENUM(L2_PENALTY);
  STRENUM(ELASTIC_NET_PENALTY);
  return false;
}

const char* LearningRateTypeToString(const LearningRateType type) {
  switch (type) {
    CASESTR(LEARNING_RATE_CONSTANT);
    CASESTR(LEARNING_RATE_OPTIMAL);
    CASESTR(LEARNING_RATE_INVERSE_SCALING);
    CASESTR(PASSIVE_AGRESSIVE);
    CASESTR(PASSIVE_AGRESSIVE_1);
    CASESTR(PASSIVE_AGRESSIVE_2);
    default:
      return "UNKNOWN";
  }
}

bool StringToLearningRateType(const std::string& s, LearningRateType* type) {
  std::string value(s);
  UpperCase(&value);
  STRENUM(LEARNING_RATE_CONSTANT);
  STRENUM(LEARNING_RATE_OPTIMAL);
  STRENUM(LEARNING_RATE_INVERSE_SCALING);
  STRENUM(PASSIVE_AGRESSIVE);
  STRENUM(PASSIVE_AGRESSIVE_1);
  STRENUM(PASSIVE_AGRESSIVE_2);
  return false;
}

bool IsValidLossType(const ModelType model_type, const LossType loss_type) {
  switch (model_type) {
    case CLASSIFICATION_MODEL:
      return (loss_type == MODIFIED_HUBER_LOSS ||
              loss_type == HINGE_LOSS ||
              loss_type == SQUARED_HINGE_LOSS ||
              loss_type == PERCEPTRON_LOSS ||
              loss_type == LOG_LOSS);
    case REGRESSION_MODEL:
      return (loss_type == SQUARED_LOSS ||
              loss_type == HUBER_LOSS ||
              loss_type == EPSILON_INSENSITIVE_LOSS ||
              loss_type == SQUARED_EPSILON_INSENSITIVE_LOSS);
    default:
      return false;
  }
}

std::shared_ptr<LossFunctor>
LossTypeToLossFunctor(const LossType type, const double param) {
  switch (type) {
    case MODIFIED_HUBER_LOSS:
      return std::make_shared<ModifiedHuberLoss>();
    case HINGE_LOSS:
      return std::make_shared<HingeLoss>(param);
    case SQUARED_HINGE_LOSS:
      return std::make_shared<SquaredHingeLoss>(param);
    case PERCEPTRON_LOSS:
      return std::make_shared<HingeLoss>(0.0);
    case LOG_LOSS:
      return std::make_shared<LogLoss>();
    case SQUARED_LOSS:
      return std::make_shared<SquaredLoss>();
    case HUBER_LOSS:
      return std::make_shared<HuberLoss>(param);
    case EPSILON_INSENSITIVE_LOSS:
      return std::make_shared<EpsilonInsensitiveLoss>(param);
    case SQUARED_EPSILON_INSENSITIVE_LOSS:
      return std::make_shared<SquaredEpsilonInsensitiveLoss>(param);
    default:
      return nullptr;
  }
}

}  // namespace sgd
}  // namespace sapien
