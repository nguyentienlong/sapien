// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com

#include <memory>

#include "sapien/optimizer/line_search_algorithm.h"
#include "sapien/internal/sapien_math.h"
#include "glog/logging.h"

namespace sapien {
namespace internal {

// Algorithm base ----------------------------------------------------------

LineSearchAlgorithm::LineSearchAlgorithm(const LineSearchMinimizer::Options&
                                         options)
    : options_(options) {}

LineSearchAlgorithm* LineSearchAlgorithm::
Create(const LineSearchMinimizer::Options& options) {
  switch (options.line_search_direction_type) {
    case sapien::STEEPEST_DESCENT:
      return new SteepestDescent(options);
    case sapien::NONLINEAR_CONJUGATE_GRADIENT:
      return new NonlinearConjugateGradient(options);
    case sapien::LBFGS:
      return NULL;
    default:
      return NULL;
  }
}

std::shared_ptr<LineSearch> LineSearchAlgorithm::GetLineSearch() const {
  LineSearch::Options ls_options;
  ls_options.initial_step_size = options_.initial_step_size;
  ls_options.sufficient_decrease = options_.sufficient_decrease;
  ls_options.max_step_contraction = options_.max_step_contraction;
  ls_options.min_step_contraction = options_.min_step_contraction;
  ls_options.min_step_size = options_.min_step_size;
  ls_options.max_iter = options_.max_num_step_size_trials;
  ls_options.sufficient_curvature_decrease =
      options_.sufficient_curvature_decrease;
  ls_options.max_step_size = options_.max_step_size;
  ls_options.max_step_expansion = options_.max_step_expansion;
  ls_options.epsilon = options_.min_step_size_search_interval_length;

  switch (options_.line_search_type) {
    case sapien::ARMIJO:
      return std::shared_ptr<LineSearch>(new ArmijoLineSearch(ls_options));
    case sapien::WOLFE:
      return std::shared_ptr<LineSearch>(new WolfeLineSearch(ls_options));
    default:
      return nullptr;
  }
}

void
LineSearchAlgorithm::Minimize(const FirstOrderFunction* obj_function,
                              double* solution) const {
  this->DoMinimize(obj_function, solution);
}

// Steepest descent --------------------------------------------------------

SteepestDescent::
SteepestDescent(const LineSearchMinimizer::Options& options)
    : LineSearchAlgorithm(options) {}

// Steepest descent algorithm
void SteepestDescent::
DoMinimize(const FirstOrderFunction* obj_function,
           double* solution) const {
  CHECK_NOTNULL(obj_function);
  CHECK_NOTNULL(solution);

  const size_t n = obj_function->n_variables();

  // Init gradient
  std::unique_ptr<double[]> gradient(new double[n]);
  obj_function->Gradient(gradient.get(), solution);
  double gradient_norm = sapien_nrm2(n, gradient.get());

  // Init line search
  std::shared_ptr<LineSearch> line_search = GetLineSearch();
  double step_size;

  // The steepest descent algorithm terminates if
  // gradient_norm <= epsilon
  const double epsilon = options().tolerance * gradient_norm;
  size_t iter = 0;

  while (iter < options().max_num_iterations && gradient_norm > epsilon) {
    // We use the -gradient as a search direction.
    step_size = line_search->Search(obj_function, solution, gradient.get(),
                                    -1.0);

    // Update solution
    // solution = step_size * (-gradient) + solution
    sapien_axpy(n, -step_size, gradient.get(), solution);

    // Update gradient
    obj_function->Gradient(gradient.get(), solution);
    gradient_norm = sapien_nrm2(n, gradient.get());

    ++iter;
  }
}

// Polack and Ribie're nonelinear conjugate gradient  --------------------
NonlinearConjugateGradient::
NonlinearConjugateGradient(const LineSearchMinimizer::Options& options)
    : LineSearchAlgorithm(options) {}

void NonlinearConjugateGradient::
DoMinimize(const FirstOrderFunction* obj_function,
           double* solution) const {
  CHECK_NOTNULL(obj_function);
  CHECK_NOTNULL(solution);

  const size_t n = obj_function->n_variables();

  // Init gradient
  std::unique_ptr<double[]> gradient(new double[n]);
  std::unique_ptr<double[]> previous_gradient(new double[n]);
  obj_function->Gradient(gradient.get(), solution);
  double gradient_norm2 = sapien_dot(n, gradient.get(), gradient.get());
  double mid_gradient_norm2, new_gradient_norm2;

  // Init search direction
  std::unique_ptr<double[]> search_direction(new double[n]);
  for (size_t i = 0; i < n; ++i) {
    search_direction[i] = -gradient[i];
  }

  // Init line search
  std::shared_ptr<LineSearch> line_search = GetLineSearch();
  double step_size;

  // Polack and Ribie're residual orthogonaliation
  double residual_orthogonalization;

  // The steepest descent algorithm terminates if
  // gradient_norm2 <= epsilon
  const double epsilon = options().tolerance * options().tolerance *
      gradient_norm2;
  size_t iter = 0;

  while (iter < options().max_num_iterations && gradient_norm2 > epsilon) {
    // Next step_size
    step_size = line_search->Search(obj_function, solution,
                                    search_direction.get());

    // Update solution
    // solution = step_size * search_direction + solution
    sapien_axpy(n, step_size, search_direction.get(), solution);

    // Update gradient, residual orthogonalization and serach_direction

    sapien_copy(n, gradient.get(), previous_gradient.get());
    obj_function->Gradient(gradient.get(), solution);
    new_gradient_norm2 = sapien_dot(n, gradient.get(), gradient.get());
    mid_gradient_norm2 = sapien_dot(n, gradient.get(),
                                    previous_gradient.get());
    residual_orthogonalization =
        (new_gradient_norm2 - mid_gradient_norm2) / gradient_norm2;

    sapien_scal(n, residual_orthogonalization, search_direction.get());
    sapien_axpy(n, -1.0, gradient.get(), search_direction.get());

    // CG is restarted (by setting search_direction = -gradient)
    // whenever a search direction is computed that is not a descent
    // direction.
    if (sapien_dot(n, search_direction.get(), gradient.get()) >= 0) {
      for (size_t i = 0; i < n; ++i) {
        search_direction[i] = -gradient[i];
      }
    }
    // Finally
    gradient_norm2 = new_gradient_norm2;
    ++iter;
  }
}

}  // namespace internal
}  // namespace sapien

