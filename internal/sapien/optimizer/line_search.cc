// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com

#include <cmath>
#include <memory>  // unique_ptr

#include "sapien/optimizer/line_search.h"
#include "sapien/internal/sapien_math.h"
#include "sapien/utility/stringprintf.h"
#include "glog/logging.h"

namespace sapien {
namespace internal {

LineSearch::LineSearch(const LineSearch::Options& options)
    : options_(options) {
}

// Phi function ----------------------------------------------------------

PhiFunction::PhiFunction(const FirstOrderFunction* func,
                         const double* position,
                         const double* direction)
     : func_(func),
       direction_(direction),
       current_step_size_(0.0),
       phi0(*(func)(position)),
       gradient0(0.0) {
  // Current position. We need to keep tract of this value as well as
  // current step size in order to quickly compute the value and
  // the derivative of phi at a given step_size
  current_position_ =
      std::unique_ptr<double[]>(new double[func->n_variables()]);
  std::memcpy(current_position_.get(), position,
              func->n_variables() * sizeof(double));

  // Current gradient of func_.
  current_func_gradient_ =
      std::unique_ptr<double[]>(new double[func->n_variables()]);
  func->Gradient(current_func_gradient_.get(), position);

  // The derivative of Phi at step_size = 0.0
  const_cast<double&>(gradient0) = sapien_dot(func->n_variables(),
                                              current_func_gradient_.get(),
                                              direction);
}

// Evaluate the value of Phi at step_size
// This method updates current_step_size_ and current_positon_
double PhiFunction::operator()(const double step_size) const {
  // Update the current_position_
  sapien_axpy(func_->n_variables(),
              step_size - current_step_size_,
              direction_,
              current_position_.get());
  current_step_size_ = step_size;
  return (*func_)(current_position_.get());
}

// Returns the derivative of Phi at current_step_size_
double PhiFunction::Derivative() const {
  // The derivative of Phi at current_step_size_ is simply the dot product
  // of the gradient of func_ at current_position_ and direction_
  func_->FirstDerivative(current_func_gradient_.get(),
                         current_position_.get());
  return sapien_dot(func->n_variables(),
                    current_func_gradient_.get(),
                    direction_);
}

// Returns the derivative of Phi at step_size.
// This methods update current_step_size_ and current_position_
double PhiFunction::Derivative(const double step_size) const {
  // Update the current_position_
  sapien_axpy(func_->n_variables(),
              step_size - current_step_size_,
              direction_,
              current_position_.get());

  // And current_step_size_
  current_step_size_ = step_size;

  // Evaluate the gradient of func_ at the current_position_
  func_->FirstDerivative(current_func_gradient_.get(),
                         current_position_.get());

  // Then the derivative of Phi at step_size is simply the dot product
  // of current_func_gradient_ and direction_
  return sapien_dot(func_->n_variables(),
                    current_func_gradient_.get(),
                    direction_);
}

// Armijo line search -----------------------------------------------------

ArmijoLineSearch::ArmijoLineSearch(const LineSearch::Options& options)
    : LineSearch(options) {}

double ArmijoLineSearch::Search(FirstOrderFunction* func,
                                const double* position,
                                const double* direction,
                                LineSearch::Summary* summary) const {
  // CHECK parameters
  CHECK_NOTNULL(func);
  CHECK_NOTNULL(position);
  CHECK_NOTNULL(direction);

  // CHECK options.
  CHECK_GT(options().sufficient_decrease, 0.0);
  CHECK_LT(options().sufficient_decrease, 1.0);
  CHECK_GT(options().max_step_contraction, 0.0);
  CHECK_LT(options().max_step_contraction, options().min_step_contraction);
  CHECK_LT(options().min_step_contraction, 1.0);
  CHECK_GT(options().min_step_size, 0.0);
  CHECK_GE(options().initial_step_size, options().min_step_size);
  CHECK_GT(options().max_iter, 0);

  // Construct Phi function
  PhiFunction phi_function(func, position, direction);

  double previous_step_size = 0.0;
  double current_step_size = options().initial_step_size;
  double interpolated_step_size;

  double previous_phi;
  double current_phi;

  double decrease;

  const int max_iter = options().max_iter;
  int iter = 0;

  while (iter < max_iter && current_step_size > options().min_step_size) {
    current_phi = phi_function(current_step_size);
    decrease = phi_function.phi0 + options().sufficient_decrease *
        current_step_size * phi_function.gradient0;

    if (current_phi <= decrease) {  // sufficient decrease condition met
      if (summary != nullptr) {
        summary->search_failed = false;
        summary->message = StringPrintf("%s\n", "SUCCESS");
      }
      return current_step_size;  // success
    } else if (iter == 0) {
      // Use Quadratic interpolation to guess next trial
      interpolated_step_size =
          QuadraticInterpolate(phi_function.phi0,
                               phi_function.gradient0,
                               current_step_size,
                               current_phi);
    } else {
      // Use Cubic interpolation to guess next trial
      interpolated_step_size =
          CubicInterpolate(phi_function.phi0,
                           phi_function.gradient0,
                           current_phi,
                           current_step_size,
                           previous_phi,
                           previous_step_size);
    }

    // Store the previous values for interpolation.
    previous_step_size = current_step_size;
    previous_phi = current_phi;

    // On on hand, we want our next trial step size to be less than the
    // previous one (so that we have a monotonically decreasing sequence
    // of step sizes). On the other hand, we don't want our next trial step
    // size to be too far less than the previous one. So we need to
    // contract our interpolated_step_size so that it always lies between
    // max_step_contraction * previous_step_size and
    // min_step_contraction * previous_step_size.
    current_step_size =
        ContractStep(interpolated_step_size,
                     options().max_step_contraction * previous_step_size,
                     options().min_step_contraction * previous_step_size);
    iter++;
  }

  if (summary != nullptr) {
    summary->search_failed = true;
    summary->message = StringPrintf("%s\n",
                                    "FAILURE: max_iter was reached, "
                                    "return min_step_size as a fallback");
  }
  return options().min_step_size;
}

// Wolfe line search ----------------------------------------------------

WolfeLineSearch::WolfeLineSearch(const LineSearch::Options& options)
    : LineSearch(options) {}

double WolfeLineSearch::Search(FirstOrderFunction* func,
                               const double* position,
                               const double* direction,
                               LineSearch::Summary* summary) const {
  // CHECK parameters
  CHECK_NOTNULL(func);
  CHECK_NOTNULL(position);
  CHECK_NOTNULL(direction);

  // CHECK options
  CHECK_LT(options().sufficient_decrease,
           options().sufficient_curvature_decrease);
  CHECK_GT(options().sufficient_decrease, 0.0);
  CHECK_LT(options().sufficient_curvature_decrease, 1.0);
  CHECK_LT(options().max_step_contraction, options().min_step_contraction);
  CHECK_GT(options().max_step_contraction, 0.0);
  CHECK_LT(options().min_step_contraction, 1.0);
  CHECK_GT(options().min_step_size, 0.0);
  CHECK_LT(options().min_step_size, options().initial_step_size);
  CHECK_LT(options().initial_step_size, options().max_step_size);
  CHECK_GT(options().max_step_expansion, 1.0);
  CHECK_GT(options().epsilon, 0.0);
  CHECK_GT(options().max_iter, 0);

  // The Wolfe line search algorithm is similar to that of the Armijo
  // line search until it found a step size armijo_step_size satisfying
  // the sufficient decrease condition. At this point the Armijo terminates
  // while the Wolfe continues the search within the interval
  //  [armijo_step_size, max_step_size] until it finds a step size satisfying
  // the Wolfe condition.

  // First stage: find armijo_step_size

  ArmijoLineSearch armijo(options());
  LineSearch::Summary armijo_summary;
  double armijo_step_size;
  armijo_step_size = armijo.Search(func, position, direction,
                                   &armijo_summary);

  // If Armijo stage failed, we return immediately
  if (armijo_summary.search_failed) {
    if (summary != nullptr) {
      summary->search_failed = true;
      summary->message = armijo_summary.message;
    }
    return armijo_step_size;
  }

  if (armijo_step_size >= options().max_step_size) {
    if (summary != nullptr) {
      summary->search_failed = true;
      summary->message = StringPrintf("%s\n",
                                      "FAILURE: the result of Armijo "
                                      "stage was larger than the given "
                                      "max_step_size. Returned "
                                      "max_step_size as a fallback");
    }
    return options().max_step_size;
  }

  // Zoom stage

  // construct Phi function
  PhiFunction phi_function(func, position, direction);

  const double sufficient_curvature =
      options().sufficient_curvature_decrease * phi_function.gradient0;
  double phi_gradient = phi_function.Derivative(armijo_step_size);

  if (phi_gradient >= sufficient_curvature) {
    if (summary != nullptr) {
      summary->search_failed = false;
      summary->message = StringPrintf("SUCCESS: %s\n",
                                      "Found Wolfe point right after "
                                      "Armijo stage");
    }
    return armijo_step_size;
  }

  double previous_step_size;
  double current_step_size = armijo_step_size;

  double previous_phi;
  double current_phi = phi_function(current_step_size);

  double sufficient_decrease;

  while (current_step_size < options().max_step_size) {
    // Save values
    previous_step_size = current_step_size;
    previous_phi = current_phi;

    // Enlarge step size
    current_step_size = options().max_step_expansion * current_step_size;
    current_phi = phi_function(current_step_size);
    sufficient_decrease = phi_function.phi0 +
        options().sufficient_decrease * current_step_size *
        phi_function.gradient0;

    if (current_phi > sufficient_decrease) {
      // The current_step_size violates the sufficient decrease condition.
      // So we know from [1] that the solution must lie in the interval
      // [previous_step_size, current_step_size].
      return Refine(&phi_function,
                    previous_step_size,
                    previous_phi,
                    current_step_size,
                    current_phi,
                    summary);
    }

    // Do NOT pass in the current_step_size beacause if we did that,
    // it would update the current_position_ (expensive), which is
    // unnecessary because we have already computed current_position_
    // as a side-product of updating current_phi.
    phi_gradient = phi_function.Derivative();

    if (phi_gradient >= sufficient_curvature) {
      // Found Wolfe point.
      if (summary != nullptr) {
        summary->search_failed = false;
        summary->message = StringPrintf("%s\n", "SUCCESS");
      }
      return current_step_size;
    }
  }

  // Search failed.
  // Returns max_step_size as a fallback.
  if (summary != nullptr) {
    summary->search_failed = true;
    summary->message = StringPrintf("FAILURE: %s\n",
                                    "The zoom stage was failed. "
                                    "Returned max_step_size as a fallback");
  }
  return options().max_step_size;
}

// Wolfe refine stage
double WolfeLineSearch::Refine(PhiFunction* phi_function,
                               double lo, double phi_lo,
                               double hi, double phi_hi,
                               LineSearch::Summary* summary) const {
  double phi_gradient_lo = phi_function->Derivative(lo);

  double current_step_size = lo;
  double current_phi = phi_lo;
  double phi_gradient = phi_gradient_lo;

  double delta = hi - lo;
  double delta_step_size;

  const double sufficient_curvature =
      options().sufficient_curvature_decrease * phi_function->gradient0;
  double sufficient_decrease;

  while (delta > options().epsilon) {
    // Compute the delta step size.
    delta_step_size = (delta * delta * phi_gradient_lo) /
        (2.0 * (phi_lo + delta * phi_gradient_lo - phi_hi));
    delta_step_size = ContractStep(delta_step_size,
                                   0.2 * delta,
                                   0.8 * delta);

    // Update
    current_step_size = lo + delta_step_size;
    current_phi = (*phi_function)(current_step_size);
    sufficient_decrease = phi_function->phi0 +
        options().sufficient_decrease * current_step_size *
        phi_function->gradient0;

    if (current_phi <= sufficient_decrease) {
      // current_step_size satisfies the sufficient decrease condition.
      // Evaluate the derivative of Phi at current_step_size. Again
      // do NOT pass in the current_step_size.
      phi_gradient = phi_function->Derivative();

      if (phi_gradient >= sufficient_curvature) {
        // Found Wolfe point
        if (summary != nullptr) {
          summary->search_failed = false;
          summary->message = StringPrintf("SUCCESS: %s\n",
                                          "at REFINE stage");
        }
        return current_step_size;
      }

      // Replace lo endpoint
      lo = current_step_size;
      phi_lo = current_phi;
      phi_gradient_lo = phi_gradient;
      delta -= delta_step_size;
    } else {
      // current_step_size violates the sufficient decrease condition,
      // we replace hi endpoint by current_step_size.
      hi = current_step_size;
      phi_hi = current_phi;
      delta = delta_step_size;
    }
  }

  // Search failed
  // Return whaever the current value of current_step_size as a fallback.
  // TODO(Linh): Is it a good idea to return current_step_size here?
  if (summary != nullptr) {
    summary->search_failed = true;
    summary->message = StringPrintf("FAILURE: %s\n",
                                    "failed at the REFINE stage");
  }
  return current_step_size;
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
double CubicInterpolate(const double f0, const double g0,
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

// Quadratic interpolation
//
// Returns the value of x that minimizes this quadratic function:
//
//  f(x) = a * x^2 + b * x + c
//
// In which a, b, c are interpolated so that:
//
//  f(0) = f0
//  f'(0) = g0
//  f(x) = fx.
double QuadraticInterpolate(const double f0, const double g0,
                            const double x, const double fx) {
  // Given that f(x) is the quadratic function:
  //
  //  f(x)  = a * x^2 + b * x + c
  //
  // and
  //
  //  f(0) = f0
  //  f'(0) = g0
  //  f(x) = fx
  //
  // We have:
  //
  //  c = f0
  //  b = g0
  //  a = (fx - b * x - c) / x^2 = (fx - g0 * x - f0) / x^2.
  //
  // This quadratic function has the minimum at
  //
  //  x* = -b / 2a (where the gradient vanishes)
  //              -g0 * x^2
  //     = -----------------------
  //       2.0 * (fx - g0 *x - f0)
  return g0 * x * x / (2.0 * (g0 * x + f0 - fx));
}

// Contract step size between [lo, hi]
double ContractStep(const double step_size,
                    const double lo,
                    const double hi) {
  if (step_size < lo) {
    return lo;
  } else if (step_size > hi) {
    return hi;
  } else {
    return step_size;
  }
}
}  // namespace internal
}  // namespace sapien
