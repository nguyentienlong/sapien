// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com

#include <iostream>
#include <cmath>
#include <string>
#include <memory>

#include "sapien/sgd/base.h"
#include "sapien/utility/stringprintf.h"
#include "sapien/utility/weight_vector.h"
#include "sapien/utility/sequential_dataset.h"
#include "sapien/constants.h"
#include "sapien/internal/sapien_math.h"
#include "glog/logging.h"

namespace sapien {
namespace sgd {

Base::Base() : model_type(UNDEFINED),
               options_(Base::Options()) {
}

Base::Base(const ModelType model_type, const Base::Options& options)
    : model_type(model_type), options_(options) {
}

// TODO(Linh): Validate more options.
bool Base::Options::IsValid(const ModelType model_type,
                            std::string* error) const {
  bool valid = true;

  // Check loss_type
  if (!IsValidLossType(model_type, loss_type)) {
    valid = false;
    internal::StringAppendF(error, "Invalid loss_type: %s\n",
                            LossTypeToString(loss_type));
  }

  // Check l1_ratio
  if (l1_ratio < 0 || l1_ratio > 1) {
    valid = false;
    internal::StringAppendF(error, "Invalid l1_ratio: %f\n", l1_ratio);
  }

  return valid;
}

using internal::sapien_set;
using internal::sapien_nrm2;
using internal::sapien_isfinite;
using internal::sapien_allfinite;

using internal::StringAppendF;
using internal::StringPrintf;
using internal::WeightVector;
using internal::SequentialDataset;

// Train one
void Base::TrainOne(const size_t n_samples,
                    const size_t n_features,
                    const double* X,
                    const double* y,
                    const double* sample_weight,
                    double* weight,
                    double* intercept,
                    double* average_weight,
                    double* average_intercept,
                    const double weight_pos,
                    const double weight_neg) const {
  // Establish dataset.
  SequentialDataset<double> data(n_samples, n_features, X, y, sample_weight);

  // Extract model options ------------------------------------------------

  // Loss
  const std::shared_ptr<LossFunctor> loss =
      LossTypeToLossFunctor(options_.loss_type,
                            options_.loss_param);

  // Learning rate type and learning rate related params
  const LearningRateType learning_rate_type = options_.learning_rate_type;
  const double initial_learning_rate = options_.initial_learning_rate;
  const double inverse_scaling_exp = options_.inverse_scaling_exp;
  const double agressiveness_param = options_.agressiveness_param;
  double learning_rate = initial_learning_rate;

  // Penalty type and penalty related params
  const PenaltyType penalty_type = options_.penalty_type;
  const double penalty_strength = options_.penalty_strength;
  const double l1_ratio = (penalty_type == L1_PENALTY) ? 1.0 :
      ((penalty_type == L2_PENALTY) ? 0.0 : options_.l1_ratio);
  const double l2_ratio = 1 - l1_ratio;

  // Shuffle dataset after each epoch?
  const bool shuffle = options_.shuffle;

  // Number of iterations.
  const size_t max_iter = options_.max_iter;

  // Tolerance
  const double tolerance = options_.tolerance;

  // Whether to fit SGD or ASGD
  const size_t average_sgd = options_.average_sgd;

  // Whether to fit intercept
  const bool fit_intercept = options_.fit_intercept;

  // SGD or ASGD, and when to start averaging if ASGD.
  const double t0 = static_cast<double>(options_.average_sgd);

  // Logging type.
  const bool is_not_silent = !(options_.logging_type == SILENT);

  // Specific variables for traing ---------------------------------------

  // Weight vector
  WeightVector<double> w(n_features, weight, average_weight);

  // Parameters for appling L1 regularization (see [1])/
  // [1] - Tsuruoka, Y., Tsujii, J., and Ananiadou, S., 2009.
  double* q = new double[n_features];
  sapien_set(n_features, 0.0, q);
  double u = 0.0;

  // Class weight
  double class_weight = 1.0;

  // Compute the optimal initial value for optimal learning rate type.
  double optimal_init = 0.0;
  if (learning_rate_type == LEARNING_RATE_OPTIMAL) {
    double tmp = std::sqrt(1.0 / std::sqrt(penalty_strength));
    double init_eta0 = tmp / std::max(1.0, loss->DLoss(-tmp, 1.0));
    optimal_init = 1.0 / (init_eta0 * penalty_strength);
  }

  // Prediction value, prediction = dot(w, x) + intercept
  double prediction;

  // The update value after seeing each sample
  double update;

  // Iteration count.
  double t = 1.0;

  // The total loss accumulated at the current each epoch and pevious epoch.
  double sumloss = 0.0;
  double previous_loss = +Constant<double>::inf;

  // Training --------------------------------------------------------------

  for (size_t epoch = 0; epoch < max_iter; ++epoch) {
    // Report after each epoch.
    std::string report_epoch;

    if (is_not_silent) {
      StringAppendF(&report_epoch, "--Epoch: %lu\n", epoch + 1);
    }

    sumloss = 0.0;

    if (shuffle) {
      data.Shuffle();
    }

    // For each sample ---------------------------------------------------

    for (size_t i = 0; i < n_samples; ++i) {
      // Update learning rate.
      if (learning_rate_type == LEARNING_RATE_OPTIMAL) {
        learning_rate = 1.0 / (penalty_strength * (optimal_init + t - 1.0));
      } else if (learning_rate_type == LEARNING_RATE_INVERSE_SCALING) {
        learning_rate =
            initial_learning_rate / std::pow(t, inverse_scaling_exp);
      }

      typename SequentialDataset<double>::Sample next_sample = data[i];
      prediction = w.Dot(next_sample.x) + (*intercept);

      sumloss += loss->Loss(prediction, next_sample.target);

      // Compute update value for this sample based on whethe the algorithm
      // used is  plain old gradient descent or Online Passive Agressive
      // Learning.
      if (learning_rate_type == PASSIVE_AGRESSIVE ||
          learning_rate_type == PASSIVE_AGRESSIVE_1 ||
          learning_rate_type == PASSIVE_AGRESSIVE_2) {
        double x_squared_norm = sapien_nrm2(n_features, next_sample.x);
        x_squared_norm *= x_squared_norm;

        if ((learning_rate_type == PASSIVE_AGRESSIVE ||
             learning_rate_type == PASSIVE_AGRESSIVE_1) &&
            (x_squared_norm == 0)) {
          continue;
        }

        double l = loss->Loss(prediction, next_sample.target);
        double tau;

        if (learning_rate_type == PASSIVE_AGRESSIVE) {
          tau = l / x_squared_norm;
        } else if (learning_rate_type == PASSIVE_AGRESSIVE_1) {
          tau = std::min(agressiveness_param, l / x_squared_norm);
        } else if (learning_rate_type == PASSIVE_AGRESSIVE_2) {
          tau = l / (x_squared_norm + (0.5 / agressiveness_param));
        }

        update = tau;

        if (model_type == CLASSIFICATION_MODEL) {
          update *= (next_sample.target);
        } else if (next_sample.target - prediction < 0) {
          update *= -1;
        }
      } else {  // plain gradient descent
        // Compute the derivative of the loss function w.t.t p
        // Cut off when it's too big or too small.
        double dloss;
        dloss = loss->DLoss(prediction, next_sample.target);
        if (dloss < -kMaxDerivativeLoss) {
          dloss = -kMaxDerivativeLoss;
        } else if (dloss > kMaxDerivativeLoss) {
          dloss = kMaxDerivativeLoss;
        }

        update = -learning_rate * dloss;  // (d_loss_function w.r.t w_t).
      }

      class_weight = (next_sample.target > 0.0) ? weight_pos : weight_neg;
      update *= class_weight * next_sample.weight;

      // Apply the L2 penalty
      if (penalty_type == L2_PENALTY ||
          penalty_type == ELASTIC_NET_PENALTY) {
        // We don't scale by negative scalar.
        w.Scal(std::max(0.0, 1.0 -
                        l2_ratio * learning_rate * penalty_strength));
      }

      // w += update * x + intercept
      if (update != 0) {
        w.PlusAX(update, next_sample.x);
        if (fit_intercept) {
          (*intercept) += update;
        }
      }

      // Update weight everage.
      // The averaging process will start after seeing t0 samples
      // TODO(Linh): This is really odd! t - t0 + 1 should be integral!!!
      if ((t0 > 0) && (t0 <= t)) {
        w.AveragePlusAX((t - t0 + 1), update, next_sample.x);
        (*average_intercept) += (*intercept - *average_intercept)/(t - t0 + 1);
      }

      // Apply the L1 penalty.
      // (Tsuruoka, Y., Tsujii, J., and Ananiadou, S., 2009)
      //
      // TODO(Linh): This is too slow!
      if (penalty_type == L1_PENALTY ||
          penalty_type == ELASTIC_NET_PENALTY) {
        u += l1_ratio * learning_rate * penalty_strength;

        const double wscale = w.scale();

        for (size_t i = 0; i < n_features; ++i) {
          double wi = w[i];
          double qi = q[i];
          double z = wi;

          if (wi * wscale > 0) {
            w[i] = std::max(0.0, wi - ((u + qi) / wscale));
          } else if (wi * wscale < 0) {
            w[i] = std::min(0.0, wi + ((u - qi) / wscale));
          }
          q[i] += wscale * (w[i] - z);
        }
      }

      // Increment t after seeing each sample
      t++;
    }  // for each sample

    // Floating-point under-/overflow check.
    // TODO(Linh): This is not okay at all! What if we encounter
    // floating-point under-/overflow? Simply log error and return
    // is NOT ok at all!!!
    if (!sapien_allfinite(n_features, weight) ||
        !sapien_isfinite(*intercept)) {
      LOG_IF(ERROR, is_not_silent) << "Floating-point under-/overflow "
          "at epoch: " << epoch + 1;
      return;
    }

    if (is_not_silent) {
      const char* format = "Weight norm: %.6f, Bias: %.6f, "
          "Iteration count: %lu, Average loss: %.6f";
      StringAppendF(&report_epoch, format,
                    w.nrm2(),
                    *intercept,
                    static_cast<size_t>(t - 1),
                    sumloss / n_samples);
      if (options_.log_to_stdout) {
        std::cout << report_epoch << std::endl;
      } else {
        VLOG(1) << report_epoch;
      }
    }

    // Check for stopping condition.
    if (tolerance > -Constant<double>::inf &&
        (sumloss > previous_loss - tolerance * n_samples)) {
      if (is_not_silent) {
        std::string message =
            StringPrintf("Convergence after %lu epochs.",
                         epoch + 1);
        if (options_.log_to_stdout) {
          std::cout << message << std::endl;
        } else {
          VLOG(1) << message;
        }
      }
      break;
    }

    previous_loss = sumloss;
  }  // for each epoch

  // Reset weight beacause we employed 'lazy scaling'.
  // To be honest, we don't really need to reset the weight here because
  // the destructor already takes care of this.
  w.Reset();

  // Clean up
  delete[] q;
}

}  // namespace sgd
}  // namespace sapien

