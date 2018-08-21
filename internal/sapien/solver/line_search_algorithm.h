// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// Interface for all kinds of line search algorithms

#ifndef INTERNAL_SAPIEN_SOLVER_LINE_SEARCH_ALGORITHM_H_
#define INTERNAL_SAPIEN_SOLVER_LINE_SEARCH_ALGORITHM_H_

#include "sapien/solver/line_search_minimizer.h"
#include "sapien/solver/line_search.h"

namespace sapien {
namespace internal {

using sapien::LineSearchMinimizer;
using sapien::FirstOrderFunction;

// Algorithm factory
class LineSearchAlgorithm {
 public:
  explicit LineSearchAlgorithm(const LineSearchMinimizer::Options& options);

  virtual ~LineSearchAlgorithm() {}

  // Create a concrete algorithm based on the provided
  // line_search_direction_type and options.
  static LineSearchAlgorithm*
  Create(const LineSearchMinimizer::Options& options);

  // Estimiate the global minimizer of an  obj_function of type
  // FirstOrderFunction and stores the result in solution.
  void Minimize(const FirstOrderFunction& obj_function,
                double* solution) const;

 protected:
  const LineSearchMinimizer::Options& options() const { return options_; }
  std::shared_ptr<LineSearch> GetLineSearch() const;

 private:
  const LineSearchMinimizer::Options& options_;
  virtual void DoMinimize(const FirstOrderFunction& obj_function,
                          double* solution) const = 0;
};

// Concrete algorithm ------------------------------------------------------

// Steepest descent
class SteepestDescent : public LineSearchAlgorithm {
 public:
  explicit SteepestDescent(const LineSearchMinimizer::Options& options);

 private:
  virtual void DoMinimize(const FirstOrderFunction& obj_function,
                          double* solution) const;
};

// Polack and Ribie're nonlinear conjugate gradient
class NonlinearConjugateGradient : public LineSearchAlgorithm {
 public:
  explicit NonlinearConjugateGradient(const LineSearchMinimizer::Options&
                                      options);

 private:
  virtual void DoMinimize(const FirstOrderFunction& obj_function,
                          double* solution) const;
};

}  // namespace internal
}  // namespace sapien
#endif  // INTERNAL_SAPIEN_SOLVER_LINE_SEARCH_ALGORITHM_H_
