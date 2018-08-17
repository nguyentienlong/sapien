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
      return new LineSearchSteepestDescent(options);
    case sapien::NONLINEAR_CONJUGATE_GRADIENT:
      return NULL;
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
LineSearchAlgorithm::Minimize(const LineSearchObjectiveFunctor* obj_functor,
                              double* solution) const {
  this->DoMinimize(obj_functor, solution);
}

// Steepest descent --------------------------------------------------------

LineSearchSteepestDescent::
LineSearchSteepestDescent(const LineSearchMinimizer::Options& options)
    : LineSearchAlgorithm(options) {}

// Steepest descent algorithm
void LineSearchSteepestDescent::
DoMinimize(const LineSearchObjectiveFunctor* obj_functor,
           double* solution) const {
  CHECK_NOTNULL(obj_functor);
  CHECK_NOTNULL(solution);

  const size_t n = obj_functor->n_variables();

  // Init gradient
  std::unique_ptr<double[]> gradient(new double[n]);
  obj_functor->Gradient(gradient.get(), solution);
  double gradient_norm = sapien_nrm2(n, gradient.get());

  // Init line search
  std::shared_ptr<LineSearch> line_search = GetLineSearch();
  double step_size;

  // The steepest descent algorithm terminates if
  // gradient_norm <= epsilon
  const double epsilon = options().steepest_descent_tolerance *
      gradient_norm;
  size_t iter = 0;

  while (iter < options().max_num_steepest_descent_iterations &&
         gradient_norm > epsilon) {
    // We use the -gradient as a search direction.
    step_size = line_search->Search(obj_functor, solution, gradient.get(),
                                    -1.0);

    // Update solution
    // solution = step_size * (-gradient) + solution
    sapien_axpy(n, -step_size, gradient.get(), solution);

    // Update gradient
    obj_functor->Gradient(gradient.get(), solution);
    gradient_norm = sapien_nrm2(n, gradient.get());

    ++iter;
  }
}

}  // namespace internal
}  // namespace sapien

