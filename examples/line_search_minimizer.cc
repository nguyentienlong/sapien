// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// An example of using LineSearchMinimizer to minimize a two dimensional
// Rosenbrock function
//
//  f(x1, x2) = 100(x2 - x1^2)^2 + (x1 - 1)^2
#include <cstring>

#include "sapien/solver/types.h"
#include "sapien/solver/line_search_minimizer.h"
#include "glog/logging.h"

using sapien::FirstOrderFunction;
using sapien::LineSearchMinimizer;
using sapien::Preconditioner;

// Rosenbrock function
//
//  f(x1, x2) = 100(x2 - x1^2)^2 + (x1 - 1)^2.
//
// This function has a unique global minimum at x = [1, 1]
struct Rosenbrock : public FirstOrderFunction {
  // Returns the number of variables/dimensions
  int n_variables() const { return 2; }

  // Evaluate this function at the given point x.
  double operator()(const double* x) const {
    const double x1 = x[0];
    const double x2 = x[1];
    const double tmp1 = x2 - x1 * x1;
    const double tmp2 = x1 - 1.0;
    return 100.0 * tmp1 * tmp1 + tmp2 * tmp2;
  }

  // Evaluate the gradient at the given point x.
  void Gradient(const double* x, double* gradient) const {
    const double x1 = x[0];
    const double x2 = x[1];
    const double tmp = x2 - x1 * x1;

    gradient[0] = 200.0 * tmp * (-2.0 * x1) + 2.0 * (x1 - 1.0);
    gradient[1] = 200.0 * tmp;
  }
};

// A trivial identity preconditioner
struct IdentityPreconditioner : public Preconditioner {
  // Mx = b => x = M`b = b
  void InverseDot(const size_t n, const double* b, double* x) const {
    std::memcpy(x, b, n * sizeof(double));
  }

  // Update preconditioner: does nothing!
  void Update(const size_t n, const double* x) {
  }
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  // Default minimizer using Polack and Ribie're nonlinear conjugate
  // gradient
  LineSearchMinimizer m1;

  // Custom minimizer using preconditioned nonlinear conjugate gradients
  // with Polak-Ribiere parameter and identity preconditioner
  Preconditioner* M = new IdentityPreconditioner();
  LineSearchMinimizer::Options options;
  options.line_search_direction_type = sapien::NONLINEAR_CONJUGATE_GRADIENT;
  options.preconditioner = M;
  LineSearchMinimizer m2(options);

  Rosenbrock rosen;
  double m1_result[2] = {0.0, 0.0};
  double m2_result[2] = {0.0, 0.0};

  m1.Minimize(rosen, m1_result);
  m2.Minimize(rosen, m2_result);

  LOG(INFO) << "True minimizer: x* = [1.0, 1.0]";
  LOG(INFO) << "Default result: x = [" << m1_result[0] << ", "
            << m1_result[1] << "]";
  LOG(INFO) << "Identity preconditioner: x = [" << m2_result[0] << ", "
            << m2_result[1] << "]";

  return 0;
}
