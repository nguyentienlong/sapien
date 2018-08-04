// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com

#ifndef INCLUDE_SAPIEN_INTERNAL_SAPIEN_MATH_H_
#define INCLUDE_SAPIEN_INTERNAL_SAPIEN_MATH_H_

#include <cmath>

namespace sapien {
namespace internal {

// Matrix transpose type. Simpler interface to CBLAS_TRANSPOSE enum.
enum MatrixTransposeType {
  SAPIEN_BLAS_NO_TRANS,
  SAPIEN_BLAS_TRANS
};

// Y <- alpha * X + Y.
template<typename T>
void sapien_axpy(const int N, const T alpha, const T* X, T* Y);

// X <- alpha * X.
template<typename T>
void sapien_scal(const int N, const T alpha, T* X);

// Compute the dot product of two vectors.
template<typename T>
T sapien_dot(const int N, const T* X, const T* Y);

// C <- alpha * A * B + beta * C or
// C <- alpha * B * A + beta * C
template<typename T>
void sapien_gemm(const MatrixTransposeType TransA,
                 const MatrixTransposeType TransB,
                 const int M, const int N, const int K, const T alpha,
                 const T* A, const T* B, const T beta, T* C);

// y <- alpha * A * x + beta * y.
template<typename T>
void sapien_gemv(const MatrixTransposeType TransA, const int M,
                 const int N, const T alpha, const T* A, const T* x,
                 const T beta, T* y);

// Compute L2-norm of a vector.
template<typename T>
T sapien_nrm2(const int N, const T* X);

// Modifies a vector (single or double precision) inplace, setting each
// element to a given value
template<typename T>
void sapien_set(const int N, const T alpha, T* X);

// Copies a source vector src to a destination vector dst.
template<typename T>
void sapien_copy(const int N, const T* src, T* dst);

// Compute z = x - y, where x, y, z are all vectors
template<typename T>
void sapien_xmy(const int N, const T* x, const T* y, T* z);

// Compute z = x + y, where x, y, z are all vectors
template<typename T>
void sapien_xpy(const int N, const T* x, const T* y, T* z);

// Compute y = x + alpha
template<typename T>
void sapien_xpa(const int N, const T alpha, const T* x, T* y);

// Compute mean of a vector.
template<typename T>
T sapien_vmean(const int N, const T* x);

// Is value finite?
template<typename T>
inline bool sapien_isfinite(const T value) { return std::isfinite(value); }

// Is value NaN?
template<typename T>
inline bool sapien_isnan(const T value) { return std::isnan(value); }

// Is value T infinite?
template<typename T>
inline bool sapien_isinf(const T value) { return std::isinf(value); }

// Return false if any element in the array is NaN or Inf, true otherwise.
template<typename T>
bool sapien_allfinite(const int N, const T* X);

// Return the index of a maximun element in an array
template<typename T>
int sapien_imax(const int N, const T* X);
}  // namespace internal
}  // namespace sapien
#endif  // INCLUDE_SAPIEN_INTERNAL_SAPIEN_MATH_H_
