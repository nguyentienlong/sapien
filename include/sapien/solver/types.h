// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// Interface for various kinds of objective functions

#ifndef INCLUDE_SAPIEN_SOLVER_TYPES_H_
#define INCLUDE_SAPIEN_SOLVER_TYPES_H_

#include "sapien/internal/port.h"

namespace sapien {

// Interface for differentiable function f
class SAPIEN_EXPORT FirstOrderFunction {
 public:
  FirstOrderFunction() {}
  virtual ~FirstOrderFunction() {}

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
  virtual void Gradient(const double* x, double* gradient) const = 0;
};

// Interface for twice differentiable function
class SAPIEN_EXPORT SecondOrderFunction : public FirstOrderFunction {
 public:
  SecondOrderFunction() {}
  virtual ~SecondOrderFunction() {}

  // Compute the dot product of the Hessian matrix and a vector x, and
  // stores the result in result.
  virtual void HessianDot(const double* x, double* result) const = 0;
};

enum LineSearchDirectionType {
  // Search direction is the negative of gradient, i.e
  //
  //  p_k = -f'(x_k)
  STEEPEST_DESCENT,

  // Search direction is initialized to the gradient at the starting point,
  // i.e:
  //
  //  p_0 = -f'(x_0)
  //
  // Subsequently, it is updated as follows:
  //
  //  p_k = -f'(x_k) + beta * p_{k-1}
  //
  // in which beta is the Polak-Ribiere parameter and is computed as:
  //
  //         f'(x_k) . [f'(x_k) - f'(x_{k-1})]
  //  beta = ---------------------------------
  //               |f'(x_{k-1})|^2
  NONLINEAR_CONJUGATE_GRADIENT,

  // Search direction at iteration k is:
  //
  //  p_k = -H_k . f'(x_k)
  //
  // in which H_k is the inverse Hessian (f''(x_k)) approximation and
  // is updated using the limited memory BFGS formula
  // See Jorge Nocedal, Stephen J. Wright Numerical Optimization (2rd edition)
  LBFGS
};

enum LineSearchType {
  // Armijo line search,
  ARMIJO,

  // Strong Wolfe line search
  WOLFE
};

}  // namespace sapien
#endif  // INCLUDE_SAPIEN_SOLVER_TYPES_H_

