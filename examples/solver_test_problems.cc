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

// Beale function ---------------------------------------------------------

// Evaluates the value of Beale function at a given position
//
//  f(x, y) = (1.5 - x + xy^2)^2 + (2.25 - x + xy^2)^2
//            + (2.625 - x + xy^3)^2
double Beale::operator()(const double* position) const {
  const double x = position[0];
  const double y = position[1];

  double t1, t2, t3;

  t1 = (1.5 - x + x * y);
  t2 = (2.25 - x + x * y * y);
  t3 = (2.625 - x + x * y * y * y);

  return (t1 * t1 + t2 * t2 + t3 * t3);
}

// Evaluates the gradient of Beale function at a given postion
void Beale::Gradient(const double* position, double* gradient) const {
  const double x = position[0];
  const double y = position[1];

  double t1, t2, t3;

  t1 = (1.5 - x + x * y);
  t2 = (2.25 - x + x * y * y);
  t3 = (2.625 - x + x * y * y * y);

  double t1x, t1y, t2x, t2y, t3x, t3y;

  t1x = 2.0 * (y - 1.0) * t1;
  t1y = 2.0 * x * t1;

  t2x = 2.0 * (y * y - 1.0) * t2;
  t2y = 2.0 * (2.0 * x * y) * t2;

  t3x = 2.0 * (y * y * y - 1.0) * t3;
  t3y = 2.0 * (3.0 * x * y * y) * t3;

  gradient[0] = t1x + t2x + t3x;
  gradient[1] = t1y + t2y + t3y;
}

// Himmelblau function
// https://en.wikipedia.org/wiki/Himmelblau%27s_function
// ------------------------------------------------------------------------

// f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
double Himmelblau::operator()(const double* position) const {
  const double x = position[0];
  const double y = position[1];
  double t1, t2;
  t1 = (x * x + y - 11);
  t2 = (x + y * y - 7);
  return (t1 * t1 + t2 * t2);
}

void Himmelblau::Gradient(const double* position, double* gradient) const {
  const double x = position[0];
  const double y = position[1];
  double t1, t2;
  t1 = (x * x + y - 11);
  t2 = (x + y * y - 7);

  gradient[0] = 2.0 * (2.0 * x) * t1 + 2.0 * t2;
  gradient[1] = 2.0 * t1 + 2.0 * (2.0 * y) * t2;
}
}  // namespace test_prob

