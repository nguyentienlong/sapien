// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com
//
// Interface for Armijo and Wolfe line search algorithms.

#ifndef INTERNAL_SAPIEN_OPTIMIZER_LINE_SEARCH_H_
#define INTERNAL_SAPIEN_OPTIMIZER_LINE_SEARCH_H_

#include "sapien/optimizer/objective_functions.h"

namespace sapien {
namespace internal {

// One way to estimate the global minimizer of a continuously differentiable
// Lipschitz function f is to use iterative method, i.e method that, starting
// from a point x_0, build a sequence {x_k} such that f(x_(k+1)) <= f(x_k).
//
// The general iterative scheme is as follows:
//
// 1. Calculate a search direction p_k from current position x_k, and
//    ensure that p_k is a descent direction, i.e: f'(x_k) * p_k < 0,
//    whenever f'(x_k) != 0. (Note that the product f'(x_k) * p_k is
//    the dot product of two vectors in multidimensional space).
//
// 2. Use line search to calculate a suitable step-size step_size > 0 so that
//    f(x_k + step_size * p_k) < f(x_k).
//
// 3. Update th point: x_(k+1) = x_k + step_size * p_k.
//
// Step 2 above is equivalent to find step_size to minimize the function f
// along the search direction p_k, i.e to find the minimizer of the univariate
// function g(s) = f(x_k + s * p_k) (s > 0). While finding the exact
// minimum of this function is hard, expensive, and sometimes unnecessary,
// in practice we often use some iterative algorithms to approximate
// this minimizer, most well-known algorithms are: Armijo and Wolfe.
class LineSearch {
 public:
  struct Options {
    // Armijo and Wolfe line search parameters.

    // Initial step_size.
    double initial_step_size = 1.0;

    // We want to find a step_size which results in sufficient decrease of
    // the objective function f along the search direction p_k. More
    // precisely, we are looking for a step size s.t
    //
    // g(step_size) <= g(0) + sufficient_decrease * step_size * g'(0)
    double sufficient_decrease = 1e-4;

    // Say, at the current iteration of Armijo / Wolfe line search we found
    // a step_size that satifies either the sufficient decrease condition
    // (Armijo) or the sufficient decrease condition and the curvature
    // condition (Wolfe), then the next_step_size is determined as follows:
    //
    // if step_size <= max_step_contraction * previous_step_size:
    //    next_step_size = max_step_contraction * previous_step_size
    // else if step_size >= min_step_contraction * previous_step_size:
    //    next_step_size = min_step_contraction * previous_step_size
    // else
    //    next_step_size = step_size.
    //
    // Note that:
    //  0 < max_step_contraction < min_step_contraction < 1
    double max_step_contraction = 1e-3;
    double min_step_contraction = 0.9;

    // If during the line search, the step_size falls below this value,
    // it is set to this value and the line search terminates.
    double min_step_size = 1e-9;

    // Maximum number of trial step size iterations during each line search,
    // If a step size satisfying the search coditions connot be found
    // within this number of trials, the line search will terminate.
    int max_iter = 20;

    // Wolfe-specific line search parameters.

    // The Wolfe conditions consist of the Armijo sufficient decrease
    // condition, and an additional requirement that the step_size be chosen
    // s.t:
    //
    //  g'(step_size) >= sufficient_curvature_decrease * g'(0).
    //
    //  where:
    //
    //  g(step_size) = f(x_k + step_size * d_k).
    //
    // Note that: We only implement the Wolfe conditions NOT the strong
    // Wolfe conditions.
    double sufficient_curvature_decrease = 0.9;
  };

  explicit LineSearch(const LineSearch::Options& options);
  virtual ~LineSearch() {}

  // Perform the line search.
  // Given the first order (differentiable) function func, a position, and
  // a direction, returns a step_size.
  // Note that, it is the caller's resposibility to make sure that the size
  // of position as well as direction is the same as func->n_variables().
  virtual double Search(FirstOrderFunction* func,
                        const double* position,
                        const double* direction) const = 0;

 protected:
  const LineSearch::Options& options() const { return options_; }

 private:
  LineSearch::Options options_;
};

// Armijo line search.
// Idea:
//
//     1. We start out with step_size = initial_step_size (typically 1.0).
//
//     2. Initially we know three things: g(0), g'(0), and
//        g(initial_step_size) where g(s) = func(position + s * direction).
//
//        We'll use parabolic interpolation to estimate the minimum.
//
//     3. In the successive iteration, we'll use cubic interpolation to
//        estimate the minimum. The informations used by cubic interpolation
//        are: g(0), g'(0), g(current_step_size), current_step_size,
//        g(previous_step_size), and previous_step_size.
class ArmijoLineSearch : public LineSearch {
 public:
  explicit ArmijoLineSearch(const LineSearch::Options& options);
  virtual ~ArmijoLineSearch() {}

  // Perform line search.
  // Given the first order (differentiable) function func, a position,
  // and a direction, returns a step_size.
  // Note that, it is the caller's resposibility to make sure that the size
  // of position as well as direction is the same as func->n_variables().
  virtual double Search(FirstOrderFunction* func,
                        const double* position,
                        const double* direction) const;
};

// Utilities functions used in both Armijo line search and Wolfe line search

// Quadratic interpolation.
//
// Returns the value of x that minimizes this quadratic function:
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
             const double f2, const double x2);
}  // namespace internal
}  // namespace sapien
#endif  // INTERNAL_SAPIEN_OPTIMIZER_LINE_SEARCH_H_
