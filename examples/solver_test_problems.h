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
//
//  f(x) = sum 100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2
//  (sum taken from i = 1 upto N - 1)
//
// N-dimensional Rosenbrock has a unique global minimizer at (1, 1, .., 1)'
// and the minimum is 0.0
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

// Beale function
//
//  f(x, y) = (1.5 - x + xy )^2 + (2.25 - x + x * y^2)^2
//                              + (2.625 - x + x * y^3)^2
//
//  -4.5 <= x, y <= 4.5
//
// The minimizer is [3, 0.5]' and f(3, 0.5) = 0
struct Beale : public sapien::FirstOrderFunction {
  // Returns number of variables (which is 2)
  int n_variables() const { return 2; }

  // Evaluates the value of this function at a given position
  double operator()(const double* position) const;

  // Evaluates the gradient of this function at a given position
  void Gradient(const double* position, double* gradient) const;
};

// Himmelblau function
// https://en.wikipedia.org/wiki/Himmelblau%27s_function
struct Himmelblau : public sapien::FirstOrderFunction {
  int n_variables() const { return 2; }
  double operator()(const double* position) const;
  void Gradient(const double* position, double* gradient) const;
};
}  // namespace test_prob
#endif  // EXAMPLES_SOLVER_TEST_PROBLEMS_H_
