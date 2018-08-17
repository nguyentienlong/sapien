// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// LineSearchMinimizer minimizes a continuously differentiable Lipschitz
// function f using the following paradigm:
//
// Given input function f, a starting point x, a maximum number of
// iterations max_iter, and a tollerance epsilon.
//
// 1. Setup:
//
//    - iter <- 0
//    - initial_gradient <- compute gradient of f at initial point x.
//    - Initialize search_direction
//
// 2. while iter < max_iter and |gradient| > epsilon * |initial_gradient|:
//
//    - Compute a descent search_direction. Choices are: STEEPEST_DESCENT,
//      NONLINEAR_CONJUGATE_GRADIENT, and LBFGS.
//
//    - Use line search (ARMIJO or WOLFE) to find a step_size
//
//    - Update:
//      x <- x + step_size * search_direction
//      iter <- iter + 1
//      gradient <- f'(x)
//      update search_direction based on STEEPEST_DESCENT,
//      NONLINEAR_CONJUGATE_GRADIENT, or LBFGS
//
// Example: Estimating the global minimizer of Rosenbrock function:
//
//  f(x1, x2) = 100(x2 - x1^2)^2 + (x1 - 1)^2
//
// This function has a unique global minimum at x = [1, 1]
//
// # Create Rosenbrock function by implementing the
//   LineSearchObjectiveFunctor interface
//
// struct Rosenbrock : public LineSearchObjectiveFunctor {
//  int n_variables() const { return 2; }
//
//  double operator()(const double* x) const {
//   const double x1 = x[0];
//   const double x2 = x[1];
//   const double tmp1 = x2 - x1 * x1;
//   const double tmp2 = x1 - 1.0;
//   return 100.0 * tmp1 * tmp1 + tmp2 * tmp2
//  }
//
//  void Gradient(double* gradient, const double *x) const {
//   const double x1 = x[0];
//   const double x2 = x[1];
//   const double tmp = x2 - x1 * x1
//   gradient[0] = 200.0 * tmp * (-2.0 * x1) + 2.0 * (x1 - 1.0);
//   gradient[1] = 200.0 * tmp;
//  }
// };
//
//
// LineSearchMinimizer minimizer;
// Rosenbrock rosen = new RosenBrock();
//
// double solution[2] = {0.0, 0.0};  // starting point
// minimizer.Minimize(rosen, solution)
// delete rosen;


#ifndef INCLUDE_SAPIEN_OPTIMIZER_LINE_SEARCH_MINIMIZER_H_
#define INCLUDE_SAPIEN_OPTIMIZER_LINE_SEARCH_MINIMIZER_H_

#include <cstddef>

#include "sapien/internal/port.h"

namespace sapien {

// The interface for function that can be optimized using line search
// paradigm.
struct SAPIEN_EXPORT LineSearchObjectiveFunctor {
  LineSearchObjectiveFunctor() {}
  virtual ~LineSearchObjectiveFunctor() {}

  // Returns the number of variables
  virtual int n_variables() const = 0;

  // Evaluates the value of this function at the given point x.
  // Note that, it is the caller's responsibility to make sure that the
  // number of elements in array x is equal to n_variables
  virtual double operator()(const double* x) const = 0;

  // Evaluates the gradient of this function at the given point x.
  // Note that, it is the caller's responsibility to make sure that the
  // number of elements in array x as well as the size of gradient array
  // are equal to n_variables.
  virtual void Gradient(double* gradient, const double* x) const = 0;
};

enum LineSearchType {
  ARMIJO,
  WOLFE
};

enum LineSearchDirectionType {
  STEEPEST_DESCENT,
  NONLINEAR_CONJUGATE_GRADIENT,
  LBFGS
};


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
// along the search direction p_k, i.e to estimate the minimizer of the
// univariate function
//
//  phi(step_size) = f(x_k + step_size * p_k) (step_size > 0).
//
// Note that, by chain rule we have:
//
//  phi'(step_size) = f'(x_k + step_size * p_k) * p_k
//
// While finding the exact minimum of this function is hard, expensive,
// and sometimes unnecessary, in practice we often use some iterative
// algorithms to approximate this minimizer, most well-known algorithms
// are: Armijo and Wolfe.
class SAPIEN_EXPORT LineSearchMinimizer {
 public:
  struct SAPIEN_EXPORT Options {
    // Line search type to compute step_size at each iteration
    // By default Armijo line search will be used.
    LineSearchType line_search_type = ARMIJO;

    // By default Polack and Ribie're Nonlinear conjugate gradient
    // will be used
    LineSearchDirectionType line_search_direction_type =
        NONLINEAR_CONJUGATE_GRADIENT;

    // LineSearchMinimizer terminates if:
    //
    //  |current_gradient| <= tolerance * |initial_gradient|
    double tolerance = 1e-4;

    // LineSearchMinimizer minimizes an objective function f by generating
    // a sequence of points {x_k} such that
    //  f(x_{k+1}) <= f(x_k), for all k >=0.
    //
    // This parameter controls the maximum number of points generated.
    size_t max_num_iterations = 50;

    // line search specific parameters ------------------------------------

    // Armijo and Wolfe line search parameters.

    // Initial step_size
    double initial_step_size = 1.0;

    // We want to find a step_size which results in sufficient decrease of
    // the objective function f along the search direction p_k. More
    // precisely, we are looking for a step size s.t
    //
    // f(x_k + step_size * p_k) <= f(x_k) +
    //                   sufficient_decrease * step_size * f'(x_k) * p_k
    // Or equivalently:
    //
    // phi(step_size) <= phi(0) + sufficient_decrease * step_size * phi'(0)
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
    double max_step_contraction = 0.1;  // 1e-3;
    double min_step_contraction = 0.5;  // 0.9;

    // If during the line search, the step_size falls below this value,
    // it is set to this value and the line search terminates.
    double min_step_size = 1e-9;

    // Maximum number of trial step size iterations during each line search,
    // If a step size satisfying the search coditions connot be found
    // within this number of trials, the line search will terminate.
    size_t max_num_step_size_trials = 20;

    // Wolfe-specific line search parameters.

    // The Wolfe conditions consist of the Armijo sufficient decrease
    // condition, and an additional requirement that the step_size be chosen
    // s.t:
    //
    //  phi'(step_size) >= sufficient_curvature_decrease * phi'(0)
    //
    // Note that: We only implement the Wolfe conditions NOT the strong
    // Wolfe conditions.
    double sufficient_curvature_decrease = 0.9;

    // The Wolfe line search algorithm is similar to that of the Armijo
    // line search algorithm until it found a step size armijo_step_size
    // satisfying the sufficient decrease condition. At this point the
    // Armijo line search terminates while the Wolfe line search continues
    // the search in the interval [armijo_step_size, max_step_size]
    // (the ZOOM stage) until it found a step_size which satifies the Wolfe
    // condition.
    //
    // Note that, according to [1], the interval
    // [armijo_step_size, max_step_size] contains a step_size satisfying
    // the Wolfe condition.
    //
    // [1] Nocedal J., Wright S., Numerical Optimization, 2nd Ed., Springer, 1999.  // NOLINT
    double max_step_size = 4.0;

    // At each iteration in the ZOOM stage of the Wolfe line search, we
    // enlarge the current step_size by multiplying it with
    // max_step_expansion, so that we have
    //
    //  next_step_size = step_size * max_step_expansion
    //
    // If this next_step_size violates the sufficient decrease condition
    // we go to the REFINE stage (see below), if it meets the Wolfe
    // condition we return it, otherwise keep expanding step size.
    double max_step_expansion = 2.0;

    // We only reach the REFINE stage if in the ZOOM stage the step
    // expansion causes the next_step_size to violates the sufficient
    // decrease condition, once we enter this stage we have this interval:
    //
    //  [lo, hi]
    //
    // in which lo satisfies the sufficient decrease condition wile
    // the hi doesn't.
    //
    // At each iteration we'll use quadratic interpolation to generate
    // our next trial step_size (within this interval). If this step_size
    // statifies the Wolfe condition we're done, othersie we update
    // the search interval (by replacing either left endpoint or
    // or right endpoint) until its length <=
    // min_step_size_search_interval_length
    double min_step_size_search_interval_length = 1e-3;
  };

  LineSearchMinimizer() : options_(Options()) {}
  explicit LineSearchMinimizer(const Options& options) : options_(options) {}

  LineSearchMinimizer(const LineSearchMinimizer& src) = delete;
  LineSearchMinimizer& operator=(const LineSearchMinimizer& rhs) = delete;

  void Minimize(const LineSearchObjectiveFunctor* obj_functor,
                double* solution);

 protected:
  const Options& options() const { return options_; }

 private:
  Options options_;
};
}  // namespace sapien
#endif  // INCLUDE_SAPIEN_OPTIMIZER_LINE_SEARCH_MINIMIZER_H_
