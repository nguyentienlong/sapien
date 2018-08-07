// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com
//
// Base class for SGDClassifier and SGDRegressor.

#ifndef INCLUDE_SAPIEN_SGD_BASE_H_
#define INCLUDE_SAPIEN_SGD_BASE_H_

#include <stddef.h>  // size_t
#include <string>
#include <memory>    // unique_ptr

#include "sapien/internal/port.h"
#include "sapien/sgd/types.h"
#include "sapien/constants.h"
#include "sapien/sgd/loss.h"

namespace sapien {
namespace sgd {

class SAPIEN_EXPORT Base {
 public:
  // Options for both SGDClassifier and SGDRegressor.
  struct SAPIEN_EXPORT Options {
    // Returns true if the options struct has a valid configuration.
    // Returns false otherwise, and fills in *error with a message
    // describing the problem.
    bool IsValid(std::string* error) const;

    // Learning rate type.
    LearningRateType learning_rate_type = LEARNING_RATE_OPTIMAL;

    // Initial learning rate.
    double initial_learning_rate = 1.0;

    // The exponent for inverse scaling learning rate.
    //
    // If the learning rate is set to LEARNING_RATE_INVERSE_SCALING, the
    // learning rate (eta) will be updated after seeing each sample by:
    //   eta = eta0 / pow(t, inverse_scaling_exp), where t is the iteration
    //   count.
    //
    // For other learning rate types, this has no effect.
    double inverse_scaling_exp = 0.5;

    // Agressiveness parameter (C) (see [1]).
    // This field only has effect if learning_rate_type is set to one of
    // PASSIVE_AGRESSIVE_[1,2,3]. For other learning_rate_type it has
    // no effect.
    //
    // [1] - http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf.
    double agressiveness_param = 1e-3;

    // Penalty type
    PenaltyType penalty_type = L2_PENALTY;

    // Penalty strength (a.k.a regularization strength)
    double penalty_strength = 1e-4;

    // l1 ratio to use if the penalty type is set to ELASTIC_NET_PENALTY.
    //
    // If the penalty type is set to ELASTIC_NET_PENALTY, this quantity
    // is added to the loss function:
    //   l1_ratio * L1(w) + (1 - l1_ratio) * L2(w), where L1, L2  are
    //   the L1 penalty, L2 penalty, respectively.
    //
    // The value of l1_ratio must be: 0 <= l1_ratio <=1. When l1_ratio is set
    // to 0.0 meaning L2 regularization will be used, 1.0 L1 regularization,
    // anywhere in between linearly combined L1 and L2 as aforementioned.
    //
    // For other types this field has no effect.
    double l1_ratio = 0.15;

    // If shuffle is set to true, then the dataset is shuffled after each
    // epoch (one epoch = one passing over the entire dataset).
    //
    // Note that, the underlying implementation of shuffle is optimized, so
    // it takes roughly O(m) to shuffle a dataset of m traing examples
    // (only the indices are shuffled).
    bool shuffle = true;

    // Max number of iterations.
    size_t max_iter = 10;

    // The stopping criterion.
    // By default it is set to negative infinity  meaning that the
    // training process will terminate only after max_iter has reached.
    // If it is set to some finite number the iterations will stop when
    // loss > previous_loss - tol.
    double tolerance = -Constant<double>::inf;

    // Whether to fit SGD or ASGD (Averaged SGD) to the dataset.
    //
    // By default average_sgd is set to 0 meaning that SGD will be used.
    // If it is set to a positive number, say 100, then the ASGD will be used
    // instead, and the averaging process will begin after seeing 100
    // samples. For more information about ASGD, please refer to [1].
    //
    // [1] - http://research.microsoft.com/pubs/192769/tricks-2012.pdf.
    size_t average_sgd = 0;

    // Whether or not to fit the intercept (bias) term. Is this is set to
    // false, the dataset is expected to be centered around 0.
    bool fit_intercept = true;

    // Logging options ---------------------------------------------------

    LoggingType logging_type = SILENT;

    // By default the traing progress is logged to VLOG(1), which is sent
    // to STDERR depending on the vlog level. If this flag is set to true,
    // and logging_type is not SILENT, the logging ouput is sent to
    // STDOUT.
    bool log_to_stdout = false;
  };

  Base();
  explicit Base(const Base::Options& options);

  // Set loss functor.
  //
  // User could choose one of the concrete loss functor defined in loss.h
  // or implement the LossFunctor interface (also defined in loss.h)
  void loss_functor(LossFunctor<double>* loss);

  // Returns loss functor
  const LossFunctor<double>* loss_functor() const {
    return loss_functor_.get();
  }

 protected:
  const Base::Options& options() const { return options_; }

  void TrainOne(const size_t n_samples,
                const size_t n_features,
                const double* X,
                const double* y,
                const double* sample_weight,
                double* weight,
                double* intercept,
                double* average_weight,
                double* average_intercept,
                const double weight_pos,
                const double weight_neg) const;

 private:
  Base::Options options_;
  std::unique_ptr< LossFunctor<double> > loss_functor_;
};
}  // namespace sgd
}  // namespace sapien
#endif  // INCLUDE_SAPIEN_SGD_BASE_H_

