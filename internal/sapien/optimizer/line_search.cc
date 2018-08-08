// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com

#include <cmath>
#include <memory>  // unique_ptr

#include "sapien/optimizer/line_search.h"
#include "sapien/internal/sapien_math.h"
#include "glog/logging.h"

namespace sapien {
namespace internal {

LineSearch::LineSearch(const LineSearch::Options& options)
    : options_(options) {}

ArmijoLineSearch::ArmijoLineSearch(const LineSearch::Options& options)
    : LineSearch(options) {}

double ArmijoLineSearch::Search(FirstOrderFunction* func,
                                const double* position,
                                const double* direction) const {
  CHECK_NOTNULL(func);
  CHECK_NOTNULL(position);
  CHECK_NOTNULL(direction);

  const int n_variables = func->n_variables();

  // Compute the gradient of func(position + step_size * direction)
  // w.r.t step_size at 0.0, i.e to compute func'(position) * direction.
  std::unique_ptr<double[]> tmp(new double[n_variables]);
  func->Gradient(tmp.get(), position);
  const double g0 = sapien_dot(n_variables, tmp.get(), direction);
  // Since Armijo line search and all line search algorithms in general
  // requires that the search direction is a descent direction, we need
  // to check to make sure that the given search direction is indeed
  // a descent direction.
  CHECK_LT(g0, 0.0);

  // f0 = func(position),
  // f = func(next_position) = func(position + step_size * direction)
  const double f0 = (*func)(position);
  double f;
  double previous_f;

  const double initial_step_size = this->options().initial_step_size;
  const double initial_step_size_squared = initial_step_size *
      initial_step_size;  // Use for initial quadratic interpolation.

  double step_size = initial_step_size;
  double previous_step_size = 0.0;
  double step_size_iter;
  const double min_step_size = this->options().min_step_size;

  // next_position = position + step_size * direction
  std::unique_ptr<double[]> next_position(new double[n_variables]);
  std::memcpy(next_position.get(), position, n_variables * sizeof(double));

  // decrease = f0 + sufficient_decrease * step_size * g0.
  // The sufficient decrease condition satisfied iff f <= decrease.
  double decrease;

  // Sufficient decrease condition constant
  const double sufficient_decrease = this->options().sufficient_decrease;

  // Contractions.
  const double max_step_contraction = this->options().max_step_contraction;
  const double min_step_contraction = this->options().min_step_contraction;

  const int max_iter = this->options().max_iter;
  int iter = 0;

  while (iter < max_iter && step_size > min_step_size) {
    // Compute the next_position = position + step_size * direction.
    sapien_axpy(n_variables, step_size - previous_step_size, direction,
                next_position.get());

    // Compute f = func(next_position)
    f = (*func)(next_position.get());

    // Compute decrease = f0 + sufficient_decrese * step_size * g0
    decrease = f0 + sufficient_decrease * step_size * g0;

    if (f <= decrease) {  // sufficient decrease condition satisfies
      return step_size;
    } else if (iter == 0) {
      // Find step_size_iter using parabolic interpolation.
      //
      // We approximate func(position + x * direction) by quadratic
      // function: a * x^2 + b * x + c using informations:
      // func(0) = f0, func'(0) = g0, and func(initial_step_size) = f.
      //
      // Using these informations, we can easily derive:
      //
      //   c = f0  (derived from the fact that func(0) = f0)
      //   b = g0  (derived from the fact that func'(0) = g0)
      //   a = f - (c + initial_step_size * b) / (initial_step_size_squared)
      //
      // This quadratic function has the minimum at:
      //
      //   x = -b/2a = -g0 / 2.0 * a = g0 / 2.0 * (-a)
      //
      // Therefore:
      double a = (f0 + initial_step_size * g0)/initial_step_size_squared - f;
      CHECK_NE(a, 0.0);
      step_size_iter = g0 / (2.0 * a);
    } else {
      // Cubic interpolation (f0, g0, f, step_size, previous_f,
      //                      previous_step_size)
      step_size_iter = Cubic(f0, g0, f, step_size, previous_f,
                             previous_step_size);
    }

    // Store the previous values for interpolation.
    previous_step_size = step_size;
    previous_f = f;

    // Update step_size based on max_step_contraction and
    // min_step_contraction
    double lo = max_step_contraction * step_size;
    double hi = min_step_contraction * step_size;
    if (step_size_iter < lo) {
      step_size = lo;
    } else if (step_size_iter > hi) {
      step_size = hi;
    } else {
      step_size = step_size_iter;
    }

    iter++;
  }

  // Search failed
  return min_step_size;
}

// Utilities functions ---------------------------------------------------

// Quadratic interpolation
//
// We want to find the value of x that minimizes this quadratic function:
//
//  f(x) = a * x^3 + b * x^2 + c * x + d.
//
// In which a, b, c, d are coefficients such that:
//
//  f(0)  = f0
//  f'(0) = g0
//  f(x1) = f1
//  f(x2) = f2
double Cubic(const double f0, const double g0,
             const double f1, const double x1,
             const double f2, const double x2) {
  // Denote:
  //
  //   x1s = x1 * x1
  //   x2s = x2 * x2;
  //   x1c = x1 * x1 * x1
  //   x2c = x2 * x2 * x2
  //
  // We evaluate:
  //
  // | a |            1             | x2s  -x1s |   | f1 - f0 - x1 * g0 |
  // |   | = -------------------- * |           | * |                   |
  // | b |   x1s * x2s * (x1 - x2)  | -x2c  x1c |   | f2 - f0 - x2 * g0 |
  const double x1s = x1 * x1;
  const double x2s = x2 * x2;
  const double x1c = x1s * x1;
  const double x2c = x2s * x2;
  const double tmp1 = f1 - f0 - x1 * g0;
  const double tmp2 = f2 - f0 - x2 * g0;
  const double tmp = 1.0 / (x1s * x2s * (x1 - x2));

  double a, b, d;

  a = tmp * (x2s * tmp1 - x1s * tmp2);
  b = tmp * (x1c * tmp2 - x2c * tmp1);

  if (a == 0.0) {  // cubic is quadratic
    return -g0 / (2.0 * b);
  } else {
    d = b * b - 3.0 * a * g0;   // discriminant
    return (std::sqrt(d) - b)/(3.0 * a);
  }
}
}  // namespace internal
}  // namespace sapien
