// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com

#include <cstring>

#include "sapien/solver/line_search_algorithm.h"
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
    case sapien::FLETCHER_REEVES_CONJUGATE_GRADIENT:
      return new FletcherReevesCG(options);
    case sapien::POLAK_RIBIERE_CONJUGATE_GRADIENT:
      return new PolakRibiereCG(options);
    case sapien::LBFGS:
      // return new LimitedMemoryBFGS(options);
      return nullptr;
    default:
      return nullptr;
  }
}

LineSearch* LineSearchAlgorithm::GetLineSearch() const {
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
      return new ArmijoLineSearch(ls_options);
    case sapien::WOLFE:
      return new WolfeLineSearch(ls_options);
    default:
      return nullptr;
  }
}

void
LineSearchAlgorithm::Minimize(const FirstOrderFunction& obj_function,
                              double* solution) const {
  this->DoMinimize(obj_function, solution);
}

// Steepest descent --------------------------------------------------------

SteepestDescent::
SteepestDescent(const LineSearchMinimizer::Options& options)
    : LineSearchAlgorithm(options) {}

// Steepest descent algorithm
void SteepestDescent::
DoMinimize(const FirstOrderFunction& obj_function,
           double* solution) const {
  CHECK_NOTNULL(solution);

  const size_t n = obj_function.n_variables();

  // Init gradient
  double* gradient = new double[n];
  obj_function.Gradient(solution, gradient);
  double gradient_norm2 = sapien_dot(n, gradient, gradient);

  // Init line search
  LineSearch* line_search = GetLineSearch();
  double step_size;

  // The steepest descent algorithm terminates if
  //
  //  gradient_norm2 <= epsilon
  //
  // or if
  //
  //  iter >= options().max_num_iterations
  const double epsilon = options().tolerance * options().tolerance *
      gradient_norm2;
  size_t iter = 0;

  while (iter < options().max_num_iterations && gradient_norm2 > epsilon) {
    // Compte step size using -gradient as a search diretion, i.e
    //
    //  step_sise <- min obj_function(solution - s * gradient), s > 0
    step_size = line_search->Search(obj_function, solution, gradient, -1.0);

    // Update solution
    // solution = step_size * (-gradient) + solution
    sapien_axpy(n, -step_size, gradient, solution);

    // Update gradient
    obj_function.Gradient(solution, gradient);
    gradient_norm2 = sapien_dot(n, gradient, gradient);

    ++iter;
  }

  delete[] gradient;
  delete line_search;
}

// Preconditioned nonlinear conjugate gradient with Polak-Ribiere parameter
// See https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
// ------------------------------------------------------------------------

PolakRibiereCG::
PolakRibiereCG(const LineSearchMinimizer::Options& options)
    : LineSearchAlgorithm(options) {}

void PolakRibiereCG::DoMinimize(const FirstOrderFunction& function,
                                  double* solution) const {
  CHECK_NOTNULL(solution);

  const size_t n = function.n_variables();

  // Init residual to -gradient
  double* residual = new double[n];
  function.Gradient(solution, residual);
  sapien_scal(n, -1.0, residual);

  // Init preconditioner (M)
  if (options().preconditioner != nullptr) {
    options().preconditioner->Update(n, solution);
  }

  // Cache the term M`r
  double* M_inv_r = new double[n];
  if (options().preconditioner != nullptr) {
    options().preconditioner->InverseDot(n, residual, M_inv_r);
  } else {
    // M is the identity matrix, so that M_inv_r = r
    std::memcpy(M_inv_r, residual, n * sizeof(double));
  }

  // Init search direction
  // We have:
  //  d_hat = E'.d & r_hat = E`.r
  // So that:
  //  E'.d = E`.r <=> d = (E')`.E`.r = (EE')`.r = M`.r
  double* search_direction = new double[n];
  std::memcpy(search_direction, M_inv_r, n * sizeof(double));

  // preconditioned residual norm
  // We have:
  //  r_hat = E`.r => |r_hat|^2 = r_hat'.r_hat
  //                            = (E`.r)'.(E`.r)
  //                            = r'.(EE')`.r
  //                            = r'.M`.r
  double preconditioned_residual_norm2 = sapien_dot(n, residual, M_inv_r);
  double previous_preconditioned_residual_norm2;
  double mid_preconditioned_residual_norm2;

  // Polak-Ribiere parameter
  double polak_ribiere_beta;

  // Init line search
  LineSearch* line_search = GetLineSearch();
  double step_size;

  // The algorithm terminates if
  //  precondition_residual_norm2 <= epsilon
  // or if
  //  iter >= options().max_num_iterations
  const double epsilon = options().tolerance * options().tolerance *
      preconditioned_residual_norm2;
  size_t iter = 0;

  while (iter < options().max_num_iterations &&
         preconditioned_residual_norm2 > epsilon) {
    // Compute step size
    step_size = line_search->Search(function, solution, search_direction);

    // Update solution
    //  solution = step_size * search_direction + solution
    sapien_axpy(n, step_size, search_direction, solution);

    // Update residual to current -gradient
    function.Gradient(solution, residual);
    sapien_scal(n, -1.0, residual);

    // Compute Polak-Ribiere parameter

    previous_preconditioned_residual_norm2 = preconditioned_residual_norm2;
    mid_preconditioned_residual_norm2 = sapien_dot(n, residual, M_inv_r);

    // Update preconditioner
    if (options().preconditioner != nullptr) {
      options().preconditioner->Update(n, solution);
      options().preconditioner->InverseDot(n, residual, M_inv_r);
    } else {
      std::memcpy(M_inv_r, residual, n * sizeof(double));
    }

    preconditioned_residual_norm2 = sapien_dot(n, residual, M_inv_r);
    polak_ribiere_beta =
        (preconditioned_residual_norm2 - mid_preconditioned_residual_norm2) /
        previous_preconditioned_residual_norm2;

    // Update search direction
    sapien_scal(n, polak_ribiere_beta, search_direction);
    sapien_axpy(n, 1.0, M_inv_r, search_direction);

    // Since nonlinear conjugate gradients with Polak-Ribiere parameter
    // doesn't not guarentee to provide a descent search direction, we
    // need to restart it whenever the new computed search_direction is
    // not a descent direction.
    if (sapien_dot(n, search_direction, M_inv_r) <= 0) {
      std::memcpy(search_direction, M_inv_r, n * sizeof(double));
    }

    ++iter;
  }

  delete[] residual;
  delete[] search_direction;
  delete[] M_inv_r;
  delete line_search;
}

// Preconditioned nonlinear conjugate gradients with Fletcher-Reeves
// parameter
// See https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
// -------------------------------------------------------------------------

FletcherReevesCG::
FletcherReevesCG(const LineSearchMinimizer::Options& options)
    : LineSearchAlgorithm(options) {}

void FletcherReevesCG::DoMinimize(const FirstOrderFunction& function,
                                  double* solution) const {
  CHECK_NOTNULL(solution);

  const size_t n = function.n_variables();

  // Init residual to -gradient
  double* residual = new double[n];
  function.Gradient(solution, residual);
  sapien_scal(n, -1.0, residual);

  // Cache the term M`r
  double* M_inv_r = new double[n];
  if (options().preconditioner != nullptr) {
    options().preconditioner->Update(n, solution);
    options().preconditioner->InverseDot(n, residual, M_inv_r);
  } else {
    // M is identity, so that M`r = r
    std::memcpy(M_inv_r, residual, n * sizeof(double));
  }

  // Init search direction
  double* search_direction = new double[n];
  std::memcpy(search_direction, M_inv_r, n * sizeof(double));

  // Preconditioned residual squares norm
  // We have
  //  r_hat = E`.r => |r_hat|^2 = r_hat'.r_hat
  //                            = r'.(EE')`.r
  //                            = r'.M`.r = r`.M_inv_r
  double preconditioned_residual_norm2;
  double previous_preconditioned_residual_norm2;
  preconditioned_residual_norm2 = sapien_dot(n, residual, M_inv_r);

  // Init line search
  LineSearch* line_search = GetLineSearch();
  double step_size;

  // This algorithm terminates if
  //  preconditioned_residual_norm2 <= epsilon
  // or if
  //  iter >= options().max_num_iterations
  const double epsilon = options().tolerance * options().tolerance *
      preconditioned_residual_norm2;
  size_t iter = 0;

  // Fletcher-Reeves parameter
  double fletcher_reeves_beta;

  while (iter < options().max_num_iterations &&
         preconditioned_residual_norm2 > epsilon) {
    // Compute step size
    step_size = line_search->Search(function, solution, search_direction);

    // Update solution & residual
    //  solution <- step_size * search_direction + solution
    //  residual <- -gradient(solution)
    sapien_axpy(n, step_size, search_direction, solution);
    function.Gradient(solution, residual);
    sapien_scal(n, -1.0, residual);

    previous_preconditioned_residual_norm2 = preconditioned_residual_norm2;

    // Update the term M`r
    if (options().preconditioner != nullptr) {
      options().preconditioner->Update(n, solution);
      options().preconditioner->InverseDot(n, residual, M_inv_r);
    } else {
      std::memcpy(M_inv_r, residual, n * sizeof(double));
    }

    // Compute Fletcher-Reeves parameter
    preconditioned_residual_norm2 = sapien_dot(n, residual, M_inv_r);
    fletcher_reeves_beta = preconditioned_residual_norm2 /
        previous_preconditioned_residual_norm2;

    // Update search direction
    //  search_direction <- M_inv_r + beta * search_direction
    sapien_scal(n, fletcher_reeves_beta, search_direction);
    sapien_axpy(n, 1.0, M_inv_r, search_direction);

    // We need to restart CG if the new computed search_direction is not
    // a descent direction
    if (sapien_dot(n, search_direction, M_inv_r) <= 0) {
      std::memcpy(search_direction, M_inv_r, n * sizeof(double));
    }

    ++iter;
  }

  delete[] residual;
  delete[] M_inv_r;
  delete[] search_direction;
  delete line_search;
}
}  // namespace internal
}  // namespace sapien

