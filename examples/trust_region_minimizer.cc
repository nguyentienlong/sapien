// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// An example of using TrustRegionMinimizer to minimize a two dimensional
// Rosenbrock function
//
//  f(x1, x2) = 100(x2 - x1^2)^2 + (x1 - 1)^2

#include "sapien/solver/trust_region_minimizer.h"
#include "glog/logging.h"

using sapien::SecondOrderFunction;
using sapien::TrustRegionMinimizer;

// Create Rosenbrock function by implementing the SecondOrderFunction
// interface
struct Rosenbrock : public SecondOrderFunction {
  int n_variables() const { return 2; }

  double operator()(const double* x) const {
    const double x1 = x[0];
    const double x2 = x[1];
    const double tmp1 = x2 - x1 * x1;
    const double tmp2 = x1 - 1.0;
    return 100.0 * tmp1 * tmp1 + tmp2 * tmp2;
  }

  void Gradient(double* gradient, const double *x) const {
    const double x1 = x[0];
    const double x2 = x[1];
    const double tmp = x2 - x1 * x1;
    gradient[0] = 200.0 * tmp * (-2.0 * x1) + 2.0 * (x1 - 1.0);
    gradient[1] = 200.0 * tmp;
  }

  void HessianDot(const double* x, double* Hx) const {
    const double x1 = x[0];
    const double x2 = x[1];

    double H11 = 400.0 * (3.0 * x1 * x1 - x2) + 2.0;
    double H12 = -400.0 * x1;
    double H21 = -400.0 * x1;
    double H22 = 200.0;

    Hx[0] = H11 * x1 + H12 * x2;
    Hx[1] = H21 * x1 + H22 * x2;
  }

  void HessianDiag(const double* x, double* hessian_diag) const {
    hessian_diag[0] = 400.0 * (3.0 * x[0] * x[0] - x[1]) + 2.0;
    hessian_diag[1] = 200.0;
  }
};


//! f(x1, x2) = (x1 - 2*x2)^2 + (x2 - 1)^2 + 1000
struct FooFunction : public SecondOrderFunction {
  int n_variables() const {
    return 2;
  }

  double operator()(const double* w) const {
    double x1 = w[0];
    double x2 = w[1];
    return (x1 - 2.0*x2) * (x1 - 2.0*x2) + (x2 - 1.0) * (x2 - 1.0) + 1000.0;
  }

  void Gradient(double *g, const double* w) const {
    double x1 = w[0];
    double x2 = w[1];
    g[0] = 2.0 * x1 - 4.0*x2;
    g[1] = -4.0 * x1 + 10.0 * x2 - 2.0;
  }

  void HessianDot(const double* s, double* Hs) const {
    Hs[0] = 2.0 * s[0] - 4.0 * s[1];
    Hs[1] = -4.0 * s[0] + 10.0 * s[1];
  }

  void HessianDiag(const double* x, double* ret) const {
    ret[0] = 2.0;
    ret[1] = 10.0;
  }
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  Rosenbrock* rosen = new Rosenbrock();
  TrustRegionMinimizer::Options options;
  options.max_num_iterations = 1000;
  TrustRegionMinimizer trust_region;
  double solution[2] = {0.0, 0.0};
  trust_region.Minimize(rosen, solution);

  LOG(INFO) << "Trust region solution: x = [" << solution[0] << ", "
            << solution[1] << "]";
  // LOG(INFO) << "Function value: " << (*foo)(solution);

  delete rosen;
  return 0;
}

