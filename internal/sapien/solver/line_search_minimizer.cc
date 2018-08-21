// Copyright 2018

#include "sapien/solver/line_search_minimizer.h"
#include "sapien/solver/line_search_algorithm.h"

namespace sapien {

void LineSearchMinimizer::
Minimize(const FirstOrderFunction* obj_function, double* solution) {
  using internal::LineSearchAlgorithm;

  // We delegate work to proper algorithm.
  LineSearchAlgorithm* alg = LineSearchAlgorithm::Create(options());
  alg->Minimize(obj_function, solution);
  delete alg;
}
}  // namespace sapien
