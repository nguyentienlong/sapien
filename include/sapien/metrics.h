// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com

#ifndef INCLUDE_SAPIEN_METRICS_H_
#define INCLUDE_SAPIEN_METRICS_H_


#include <stddef.h>  /* size_t */

namespace sapien {
namespace metrics {

// Accuracy score of a classification model.
// Given the true label vector y_true and the predicted label vector y_pred,
// return the average number of labels that were correctly classified.
// (i.e where y_pred[c] = y_true[c]).
//
// Example:
//        const size_t N = 4;
//        char y_true[N] = {1, 1, -1, -1};
//        char y_pred[N] = {-1, 1, -1, 1};
//        double d = AccuracyScore(4, y_true, y_pred);  // 0.5
template<typename LabelType>
double AccuracyScore(const size_t N, const LabelType* y_true,
                     const LabelType* y_pred);

// Compute the Coefficient of determination (a.k.a R^2 score) of a regression
// model.
// By definition, R^2 = 1 - SS_res / SS_tot, where:
//     SS_res = sum((y_true - y_pred)^2) - the sum of squares of residuals.
//     SS_tot = sum((y_true - mean(y_true))^2) - the total sum of squares.
//
// Example:
//       const size_t N = 4;
//       double y_true[N] = {3, -0.5, 2, 7};
//       double y_pred[N] = {2.5, 0.0, 2, 8};
//       double score = R2Score(4, y_true, y_pred);  // 0.948..
template<typename TargetType>
TargetType R2Score(const size_t N, const TargetType* y_true,
                   const TargetType* y_pred);


}  // namespace metrics
}  // namespace sapien
#endif  // INCLUDE_SAPIEN_METRICS_H_

