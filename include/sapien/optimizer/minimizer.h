// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// Interface for all kinds of minimizers

#ifndef INCLUDE_SAPIEN_OPTIMIZER_MINIMIZER_H_
#define INCLUDE_SAPIEN_OPTIMIZER_MINIMIZER_H_

namespace sapien {
namespace optimizer {

// Virtual base class for all minimizers
template<typename CostFunctorType>
class Minimizer {
 public:
  // Minimizes cost_functor and stores the minimizer in solution
  virtual void Minimize(const CostFunctorType& cost_functor,
                        double* solution) = 0;
  virtual ~Minimizer() {}
};

}  // namespace optimizer
}  // namespace sapien
#endif  // INCLUDE_SAPIEN_OPTIMIZER_MINIMIZER_H_
