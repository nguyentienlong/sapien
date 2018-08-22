// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// A collection of test problems for testing different kinds of
// solvers/optimizers. See [1] for more details
//
// [1] - https://en.wikipedia.org/wiki/Test_functions_for_optimization

#ifndef EXAMPLES_SOLVER_TEST_PROBLEMS_H_
#define EXAMPLES_SOLVER_TEST_PROBLEMS_H_

#include "sapien/solver/types.h"

namespace test_prob {

// N-dimensional Rosenbrock function
struct Rosenbrock : sapien::FirstOrderFunction {
  // Default is 3-D Rosenbrock
  Rosenbrock() : N_(2) {}
  explicit Rosenbrock(const int N) : N_(N) {}

  // Returns number of variables (i.e dimension)
  int n_variables() const { return N_; }

  // Evaluates the value of Rosenbrock at a given position
  //  position = [x1, x2, ..., xn]
  double operator()(const double* position) const;

  // Evaluates the gradient of Rosenbrock at a given position
  //  position = [x1, x2, ..., xn]
  void Gradient(const double* position, double* gradient) const;

 private:
  int N_;  // dimension
};
}  // namespace test_prob
#endif  // EXAMPLES_SOLVER_TEST_PROBLEMS_H_
