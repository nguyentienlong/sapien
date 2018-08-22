// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com

#include "solver_test_problems.h"  // NOLINT

namespace test_prob {

// N-dimensional Rosenbrock function ---------------------------------------

// Evaluate Rosenbrock at given x = [x1, x2, ..., xn]
//
//  f(x) = sum [100(x_{i+1} - x_i*x_i)^2 + (1 - x_i)^2]
//  (sum taken from i = 1 upto n-1)
double Rosenbrock::operator()(const double* x) const {
  double ret = 0.0;
  double xi;
  double xi1;

  for (int i = 0; i < N_ - 1; ++i) {
    xi = x[i];
    xi1 = x[i+1];

    ret += 100.0 * (xi1 - xi * xi) * (xi1 - xi * xi) +
        (1.0 - xi) * (1.0 - xi);
  }

  return ret;
}

// Evaluate gradient of Rosenbrock at a given position = [x1, x2, ..,xn]
//
//  f(x) = sum [100(x_{i+1} - x_i*x_i)^2 + (1 - x_i)^2]
//       = sum t_i for i = 1 upto n-1
//
// We have
//
//  dt_i
//  ---- = -400 * xi * (xi1 - xi^2) - 2 * (1 - xi)
//  dxi
//
//  dt_i
//  ---- = 200 * (xi1 - xi^2)
//  dxi1
void Rosenbrock::Gradient(const double* position, double* gradient) const {
  double xi;
  double xi1;  // x_{i+1}
  double xi2;  // x_{i+2}

  double ti_xi = 0.0;  // gradient of the term t_i w.r.t x_i
  double ti_xi1 = 0.0;  // gradient of the term t_i w.r.t x_{i+1}

  double previous = 0.0;

  int i;
  for (i = 0; i < N_ - 1; ++i) {
    xi = position[i];
    xi1 = position[i+1];

    ti_xi = -400.0 * xi* (xi1 - xi * xi) - 2.0 * (1 - xi);
    ti_xi1 = 200.0 * (xi1 - xi * xi);

    gradient[i] = previous + ti_xi;
    previous = ti_xi1;
  }
  gradient[i] = previous;
}
}  // namespace test_prob

