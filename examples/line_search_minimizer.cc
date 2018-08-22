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
#include "solver_test_problems.h"  // NOLINT
#include "glog/logging.h"

using sapien::LineSearchMinimizer;
using sapien::Preconditioner;

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

  // Init minimizer
  LineSearchMinimizer::Options options;
  options.max_num_iterations = 50;
  LineSearchMinimizer minimizer(options);

  const int N = 10;
  test_prob::Rosenbrock rosen(N);
  double solution[N];
  for (int i = 0; i < N; ++i) { solution[i] = 0.0; }

  LOG(INFO) << "Value of Rosenbrock at starting point: "
            << rosen(solution);

  minimizer.Minimize(rosen, solution);

  LOG(INFO) << "Estimated minimizer of " << N << " dimensional Rosenbrock:";
  for (int i = 0; i < N; ++i) {
    LOG(INFO) << solution[i];
  }

  LOG(INFO) << "Estimated minima:  " << rosen(solution);

  return 0;
}
