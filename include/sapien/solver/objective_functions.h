// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// Interface for various kinds of objective functions

#ifndef INCLUDE_SAPIEN_SOLVER_OBJECTIVE_FUNCTIONS_H_
#define INCLUDE_SAPIEN_SOLVER_OBJECTIVE_FUNCTIONS_H_

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
  virtual void Gradient(double* gradient, const double* x) const = 0;
};

// Interface for twice differentiable function
class SAPIEN_EXPORT SecondOrderFunction : public FirstOrderFunction {
 public:
  SecondOrderFunction() {}
  virtual ~SecondOrderFunction() {}

  // Compute the dot product of the Hessian matrix and a vector x, and
  // stores the result in result.
  virtual void HessianDot(const double* x, double* result) const = 0;

  // Return the diagonal of the Hessian matrix at current point x.
  virtual void HessianDiag(const double* x, double* hessian_diag) const = 0;
};
}  // namespace sapien
#endif  // INCLUDE_SAPIEN_SOLVER_OBJECTIVE_FUNCTIONS_H_

