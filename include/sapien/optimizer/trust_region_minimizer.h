// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// Trust Region method for minimizing twice continuously differentiable
// function f.
//
// Here is the idea:
//
// TrustRegionMinimizer minimizes a twice continuously differentiable function
// f by generating a sequence of points {x_k} such that
// f(x_{k+1}) <= f(x_k) for all k >= 0.
// At each iteration TrustRegionMinimizer tries to:
//
// 1. Approximate f using second order Taylor expansion within a ball
//    centered at the current iterate x_k with some radius
//    trust_region_radius, i.e:
//
//    f(s + x_k) ~ f(x_k) + f'(x_k) * s + 1/2 s^T * f''(x_k) * s
//               ~ q(s)
//    for all s such that:
//    |s| <= trust_region_radius
//
//    (f'(x_k), f''(x_k) is the gradient, the Hessian matrix of f at
//     the current iterate x_k).
//
// 2. Solve the trust region subproblem:
//
//    s* <-- argmin(q(s)) w.r.t |s| <= trust_region_radius
//
// 3. Update the iterate and trust_region_radius based on the radio of
//    two importance values:
//
//    predicted_reduction = q(x_k + s*) - q(x_k)
//    actual_reduction = f(x_k + s*) - f(x_k)
//    r = actual_reduction / predicted_reduction
//
//    - If the ratio r is between 0 < eta1 < r < eta2 < 1 we accept the step
//      s* and do not modify trust_region_radius:
//       x_{k+1} <-- x_k + s*
//
//    - If the ratio r is small r <= eta1 we reject the step s* and shrink
//      the trust_region_radius by a factor gamma1 < 1:
//       trust_region_radius <-- trust_region_radius * gamma1
//
//    - If the ratio is large r >= eta2 we accept the step s* and enlarge
//      the trust_region_radius by a factor gamma2 > 1.
//       x_{k+1} <-- x_k + s*
//       trust_region_radius <-- trust_region_radius * gamma2
//
// 4. If the L2 norm of current gradient f'(x_k) <= epsilon then terminate
//    otherwise, go back to step 1.

#ifndef INCLUDE_SAPIEN_OPTIMIZER_TRUST_REGION_MINIMIZER_H_
#define INCLUDE_SAPIEN_OPTIMIZER_TRUST_REGION_MINIMIZER_H_

namespace sapien {

}  // namespace sapien
#endif  // INCLUDE_SAPIEN_OPTIMIZER_TRUST_REGION_MINIMIZER_H_

