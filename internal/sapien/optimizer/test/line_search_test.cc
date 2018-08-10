// Copyright 2018.

#include <memory>

#include "sapien/optimizer/line_search.h"
#include "sapien/internal/sapien_math.h"
#include "gtest/gtest.h"
#include "glog/logging.h"

namespace sapien {
namespace internal {

// f(x) = 0.5 * x^T * A * x - b^T * x + c, where:
//
//  A = I is the identidy matrix.
//  b^T = [2, 1]
//  c = 1.
//
// This quadratic function has a unique global minimum at x = b.
struct ObjectiveFunction : public FirstOrderFunction {
  // Returns the number of variables/dimension
  int n_variables() const { return 2; }

  // Evaluate this function at the given point x.
  double operator()(const double* x) const {
    const double x1 = x[0];
    const double x2 = x[1];
    return 0.5 * (x1 * x1 + x2 * x2) - (2.0 * x1 + x2) + 1.0;
  }

  // Evaluate the gradient of this function at the given point x.
  void Gradient(double* gradient, const double* x) const {
    gradient[0] = x[0] - 2.0;
    gradient[1] = x[1] - 1.0;
  }
};

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
  void Gradient(double* gradient, const double* x) const {
    const double x1 = x[0];
    const double x2 = x[1];
    const double tmp = x2 - x1 * x1;

    gradient[0] = 200.0 * tmp * (-2.0 * x1) + 2.0 * (x1 - 1.0);
    gradient[1] = 200.0 * tmp;
  }
};


// Beale function
//
//  f(x1, x2) = (1.5 - x1 + x1 * x2)^2 + (2.25 - x1 + x1* x2^2)^2
//              + (2.625 - x1 + x1 * x2^{3})^2
//
// This function has unique global minimum at [3, 0.5] and
//
//  f(3, 0.5) = 0.
//
// Search domian is -4.5 <= x1, x2 <= 4.5
struct Beale : public FirstOrderFunction {
  int n_variables() const { return 2; }

  double operator()(const double* x) const {
    const double x1 = x[0];
    const double x2 = x[1];
    const double tmp1 = 1.5 - x1 + x1 * x2;
    const double tmp2 = 2.25 - x1 + x1 * x2 * x2;
    const double tmp3 = 2.625 - x1 + x1 * x2 * x2 * x2;
    return tmp1 * tmp1 + tmp2 * tmp2 + tmp3 * tmp3;
  }

  void Gradient(double* gradient, const double* x) const {
    const double x1 = x[0];
    const double x2 = x[1];
    const double tmp1 = 1.5 - x1 + x1 * x2;
    const double tmp2 = 2.25 - x1 + x1 * x2 * x2;
    const double tmp3 = 2.625 - x1 + x1 * x2 * x2 * x2;

    gradient[0] = 2.0 * (x2 - 1.0) * tmp1 +
        2.0 * (x2 * x2 - 1.0) * tmp2 +
        2.0 * (x2 * x2 * x2 - 1.0) * tmp3;
    gradient[1] = 2.0 * (x1) * tmp1 +
        2.0 * (2.0 * x1 * x2) * tmp2 +
        2.0 * (3.0 * x1 * x2 * x2) * tmp3;
  }
};

// Simple steepest descent algorithm.
void SteepestDescent(const FirstOrderFunction* func,
                     const double tolerance,
                     const LineSearch* line_search,
                     double* x) {
  CHECK_NOTNULL(func);
  CHECK_NOTNULL(x);

  // Gradient
  std::unique_ptr<double[]> gradient(new double[func->n_variables()]);
  func->Gradient(gradient.get(), x);
  double gradient_norm = sapien_nrm2(func->n_variables(), gradient.get());

  // Search direction: steepest descent direction
  std::unique_ptr<double[]> search_direction(new double[func->n_variables()]);
  for (size_t i = 0; i < func->n_variables(); ++i) {
    search_direction[i] = -gradient[i];
  }

  // Stopping criterion
  const double epsilon = tolerance * gradient_norm;

  double step_size;

  LineSearch::Summary summary;

  while (gradient_norm > epsilon) {
    // Compute step_size using line_search
    step_size = line_search->Search(func, x, search_direction.get(),
                                    &summary);

    // Output summary
    LOG(INFO) << "Line search summary:";
    if (summary.search_failed) {
      LOG(INFO) << "    Termination: FAILURE";
    } else {
      LOG(INFO) << "    Termination: SUCCESS";
    }
    LOG(INFO) << "    Armijo iterations: " << summary.num_armijo_iterations;
    LOG(INFO) << "    Zoom iterations: " << summary.num_zoom_iterations;
    LOG(INFO) << "    Refine iterations: " << summary.num_refine_iterations;
    LOG(INFO) << "    Time elapsed: " << summary.total_time_elapsed;

    // Update
    sapien_axpy(func->n_variables(), step_size, search_direction.get(), x);
    func->Gradient(gradient.get(), x);
    gradient_norm = sapien_nrm2(func->n_variables(), gradient.get());
    for (size_t i = 0; i < func->n_variables(); ++i) {
      search_direction[i] = -gradient[i];
    }
  }
}

TEST(ArmijoLineSearch, FindMinimumOfASimpleQuadraticFunction) {
  const int N = 2;

  ObjectiveFunction func;
  Rosenbrock rosenbrock;
  Beale beale;

  const double rosenbrock_optimizer[N] = {1.0, 1.0};
  const double beale_optimizer[N] = {3, 0.5};

  double x[N] = {0.0, 0.0};

  // LineSearch::Options options;
  // options.sufficient_decrease = 0.5;
  // options.sufficient_curvature_decrease = 0.9;

  ArmijoLineSearch armijo;
  WolfeLineSearch wolfe;

  SteepestDescent(&beale, 1e-4, &wolfe, x);

  LOG(INFO) << "True value of f_min = " << beale(beale_optimizer);
  LOG(INFO) << "Estimated global minimizer x = [" << x[0]
            << ", " << x[1] << "]";
  LOG(INFO) << "Estimated minimum = " << beale(x);
}
}  // namespace internal
}  // namespace sapien

