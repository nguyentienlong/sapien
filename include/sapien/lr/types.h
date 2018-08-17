// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com

#ifndef INCLUDE_SAPIEN_LR_TYPES_H_
#define INCLUDE_SAPIEN_LR_TYPES_H_

namespace sapien {
namespace lr {

// The Logistic Regression cost function that we want to minimize is:
//
//  f(w) = 1/2 * w^{T} * w + C * sum(log(1 + exp(-yi * w^{T} * xi)))
//
// It is easy to show that this function is twice continuously differentiable
// and its hessian matrix f''(w) is positive definite. So f has a unique
// global minimizer. Most unconstrained optimization techniques can be used
// to estimate the global minimizer of f.
//
// We allow two iterative methods for optimizing f:
//
// LINE_SEARCH:
//            1. Initial guess w0 (typically set to 0s or randomly chosen).
//            2. Repeat until some stopping criteria are met, e.g.
//               l2_norm(w_k) <= epsilon * l2_norm(w0):
//
//               a. Choose a descent search direction d_k, possible choices
//                  are: STEEPEST_DESCENT, CONJUGATE_GRADIENT, and LBFGS
//
//               b. Use line search (ARMIJO or WOLFE) to find a step_size
//                  we need to take in this direction.
//
//               c. Update weight vector:
//                  w_{k + 1} = w_k + step_size * d_k.
//
// TRUST_REGION:
//             1. Initial guess w0 (typically set to 0s or randomly chosen),
//                and some delta > 0.
//
//             2. Repeat until some stopping criteria are met, e.g.
//                l2_norm(w_k) <= epsilon * l2_norm(w0):
//
//                a. Use second order Taylor expension to approximate f
//                   inside the 'trust region' - which is the ball centered
//                   at w_k with radius is the current value of delta.
//                   In other words, if we denote g = f'(w_k), H = f''(w_k)
//                   is the gradient and the Hessian matrix of f at the
//                   current w_k, respectively, we have:
//
//                    f(w_k + s) = f(w_k) + g^T * s + 1/2 * s^T * H * s
//                   for all s such that l2_norm(s) <= delta.
//
//                   Our goal is to minimize the subproblem:
//
//                    q(s) = g^T * s + 1/2 * s^T * H * s
//
//                   Within this trust region.
//
//                 b. Minimize q(s) -> step_size.
//
//                 c. Update the weight vector and the trust region based on
//                    the result of step b.
//
// Note that: The above sketch of the TRUST_REGION algorithm is just the
// intuition behind this technique, our implementation is based on [1].
//
// [1] - https://www.csie.ntu.edu.tw/~cjlin/papers/logistic.pdf
// MinimizerType
enum MinimizerType {
  LINE_SEARCH,
  TRUST_REGION
};

// LineSearchDirectionType
enum LineSearchDirectionType {
  STEEPEST_DESCENT,
  CONJUGATE_GRADIENT,
  LBFGS
};

// LineSearchType
enum LineSearchType {
  ARMIJO,
  WOLFE
};
}  // namespace lr
}  // namespace sapien
#endif  // INCLUDE_SAPIEN_LR_TYPES_H_
