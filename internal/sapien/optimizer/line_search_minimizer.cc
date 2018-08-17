// Copyright 2018

#include "sapien/optimizer/line_search_minimizer.h"
#include "sapien/optimizer/line_search_algorithm.h"

namespace sapien {

void LineSearchMinimizer::
Minimize(const LineSearchObjectiveFunctor* obj_functor, double* solution) {
  using internal::LineSearchAlgorithm;

  // We delegate work to proper algorithm.
  LineSearchAlgorithm* alg = LineSearchAlgorithm::Create(options());
  alg->Minimize(obj_functor, solution);
  delete alg;
}
}  // namespace sapien
