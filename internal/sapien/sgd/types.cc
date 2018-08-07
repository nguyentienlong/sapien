// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com

#include <algorithm>

#include "sapien/sgd/types.h"

namespace sapien {
namespace sgd {

#define CASESTR(x) case x: return #x
#define STRENUM(x) if (value == #x) {*type = x; return true; }

static void UpperCase(std::string* input) {
  std::transform(input->begin(), input->end(), input->begin(), std::toupper);
}

// Utility functions -------------------------------------------------------

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

}  // namespace sgd
}  // namespace sapien
