// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com liblinear (see [2])
//
// Trust Region method for minimizing twice continuously differentiable
// function f.
//
// Here is the idea:
//
// TrustRegionMinimizer minimizes a twice continuously differentiable function
// f by generating a sequence of points {x_k} such that
// f(x_{k+1}) <= f(x_k) for all k >= 0.
// At each iteration TrustRegionMinimizer tries to:
//
// 1. Approximate f using second order Taylor expansion within a ball
//    centered at the current iterate x_k with some radius
//    trust_region_radius, i.e:
//
//    f(s + x_k) ~ f(x_k) + f'(x_k) * s + 1/2 s^T * f''(x_k) * s
//               ~ q(s)
//    for all s such that:
//    |s| <= trust_region_radius
//
//    (f'(x_k), f''(x_k) is the gradient, the Hessian matrix of f at
//     the current iterate x_k).
//
// 2. Solve the trust region subproblem:
//
//    s* <-- argmin(q(s)) w.r.t |s| <= trust_region_radius
//
// 3. Update the iterate and trust_region_radius based on the radio of
//    two importance values:
//
//    predicted_reduction = q(x_k + s*) - q(x_k)
//    actual_reduction = f(x_k + s*) - f(x_k)
//    r = actual_reduction / predicted_reduction
//
//    - If the ratio r is between 0 < eta1 < r < eta2 < 1 we accept the step
//      s* and do not modify trust_region_radius:
//       x_{k+1} <-- x_k + s*
//
//    - If the ratio r is small r <= eta1 we reject the step s* and shrink
//      the trust_region_radius by a factor gamma1 < 1:
//       trust_region_radius <-- trust_region_radius * gamma1
//
//    - If the ratio is large r >= eta2 we accept the step s* and enlarge
//      the trust_region_radius by a factor gamma2 > 1.
//       x_{k+1} <-- x_k + s*
//       trust_region_radius <-- trust_region_radius * gamma2
//
// 4. If the L2 norm of current gradient f'(x_k) <= epsilon then terminate
//    otherwise, go back to step 1.
//
// The implementation details directly follows [1] and [2].
//
// Example: Estimating the global minimizer of Rosenbrock function:
//
//  f(x1, x2) = 100(x2 - x1^2)^2 + (x1 - 1)^2
//
// This function has a unique global minimum at x = [1, 1]
//
// # Create Rosenbrock function by implementing the SecondOrderFunction
//   interface
//
// struct Rosenbrock : public SecondOrderFunction {
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
//
//  void HessianDot(const double* x, double* Hx) const {
//   double H11 = 400.0 * (3.0 * x[0] * x[0] - x[1]) + 2.0;
//   double H12 = -400.0 * x[0];
//   double H21 = -400.0 * x[0];
//   double H22 = 200.0;
//  }
//
//  void HessianDiag(const double* x, double* hessian_diag,
//                   const double inc = 0.0) const {
//   hessian_diag[0] = 400.0 * (3.0 * x[0] * x[0] - x[1]) + 2.0;
//   hessian_diag[1] = 200.0;
//  }
// };
//
// Rosenbrock rosen = new Rosenbrock();
// TrustRegionMinimizer trust_region;
// double solution[0] = {0.0, 0.0}
// trust_region.Minimize(rosen, solution);
//
// [1] - https://www.csie.ntu.edu.tw/~cjlin/papers/logistic.pdf
// [2] - https://github.com/cjlin1/liblinear

#ifndef INCLUDE_SAPIEN_SOLVER_TRUST_REGION_MINIMIZER_H_
#define INCLUDE_SAPIEN_SOLVER_TRUST_REGION_MINIMIZER_H_

#include <cstddef>

#include "sapien/internal/port.h"
#include "sapien/solver/objective_functions.h"

namespace sapien {

class SAPIEN_EXPORT TrustRegionMinimizer {
 public:
  struct SAPIEN_EXPORT Options {
    // TrustRegionMinimizer terminates if
    //
    //  |gradient| <= tolerance * |initial_gradient|
    double tolerance = 1e-4;

    // At each iteration TrustRegionMinimizer solves the constrained
    // minimization problem:
    //
    //  s* <-- argmin(q(s)) w.r.t |s| <= trust_region_radius
    //
    // using preconditioned nonlinear conjugate gradient algorithm.
    // So this parameter is the stopping criterion for nonlinear
    // conjugate gradient subproblem.
    double subproblem_tolerance = 1e-4;

    // TrustRegionMinimizer minimizes an objective function f by generating
    // a sequence of points {x_k} such that
    //  f(x_{k+1}) <= f(x_k), for all k >=0.
    //
    // This parameter controls the maximum number of points generated.
    size_t max_num_iterations = 50;

    // Parameters for updating trust_region_step
    // See [1] & [2] for more details
    double eta0 = 1e-4;
    double eta1 = 0.25;
    double eta2 = 0.75;

    // Parameters for updating trust_region_radius
    double sigma1 = 0.25;
    double sigma2 = 0.5;
    double sigma3 = 4.0;
  };

  // Construct a TrustRegionMinimizer with default options
  TrustRegionMinimizer();

  // Construct a TrustRegionMinimizer with provided custom objects.
  explicit TrustRegionMinimizer(const TrustRegionMinimizer::Options&
                                options);

  // Estimates the global minimizer of an obj_function of type
  // SecondOrderFunction and stores the result in solution.
  void Minimize(const SecondOrderFunction* obj_function, double* solution);

 protected:
  const TrustRegionMinimizer::Options& options() const { return options_; }

 private:
  TrustRegionMinimizer::Options options_;
  void SolveSubproblem(const SecondOrderFunction* obj_function,
                       const double* gradient,
                       const double* approximate_hessian,
                       const double trust_region_radius,
                       double* trust_region_step,
                       double* residual,
                       bool* reach_trust_region_boundary) const;
};
}  // namespace sapien
#endif  // INCLUDE_SAPIEN_SOLVER_TRUST_REGION_MINIMIZER_H_

