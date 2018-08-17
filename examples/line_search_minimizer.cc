// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// An example of using LineSearchMinimizer to minimize a two dimensional
// Rosenbrock function
//
//  f(x1, x2) = 100(x2 - x1^2)^2 + (x1 - 1)^2

#include "sapien/optimizer/line_search_minimizer.h"
#include "glog/logging.h"

using sapien::LineSearchObjectiveFunctor;
using sapien::LineSearchMinimizer;

// Rosenbrock function
//
//  f(x1, x2) = 100(x2 - x1^2)^2 + (x1 - 1)^2.
//
// This function has a unique global minimum at x = [1, 1]
struct Rosenbrock : public LineSearchObjectiveFunctor {
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
  void Gradient(double* gradient, const double* x) const {
    const double x1 = x[0];
    const double x2 = x[1];
    const double tmp = x2 - x1 * x1;

    gradient[0] = 200.0 * tmp * (-2.0 * x1) + 2.0 * (x1 - 1.0);
    gradient[1] = 200.0 * tmp;
  }
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  LineSearchMinimizer::Options options;
  options.line_search_direction_type = sapien::STEEPEST_DESCENT;
  options.max_num_steepest_descent_iterations = 500;
  options.steepest_descent_tolerance = 1e-4;
  LineSearchMinimizer minimizer;
  Rosenbrock* rosen = new Rosenbrock();
  double solution[2] = {0.0, 0.0};
  minimizer.Minimize(rosen, solution);

  LOG(INFO) << "True minimizer: x* = [1.0, 1.0]";
  LOG(INFO) << "Estimated minimizer: x = [" << solution[0]
            << ", " << solution[1] << "]";
  return 0;
}
