// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com

#include "sapien/internal/port.h"

#ifdef SAPIEN_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
extern "C" {
#include <cblas.h>
}
#endif

#include <cstring>  /* memset */

#include "sapien/internal/sapien_math.h"

namespace sapien {
namespace internal {

// Convert our MatrixTransposeType to CBLAS_TRANSPOSE.
CBLAS_TRANSPOSE
MatrixTransposeTypeToCblasTranspose(const MatrixTransposeType Trans) {
  switch (Trans) {
    case SAPIEN_BLAS_NO_TRANS:
      return CblasNoTrans;
    case SAPIEN_BLAS_TRANS:
      return CblasTrans;
    default:
      return CblasConjTrans;
  }
}

// Y <- alpha * X + Y

template<>
void sapien_axpy<float>(const int N, const float alpha, const float* X,
                        float* Y) {
  cblas_saxpy(N, alpha, X, 1, Y, 1);
}

template<>
void sapien_axpy<double>(const int N, const double alpha, const double* X,
                         double* Y) {
  cblas_daxpy(N, alpha, X, 1, Y, 1);
}

// X <- alpha * X

template<>
void sapien_scal<float>(const int N, const float alpha, float* X) {
  cblas_sscal(N, alpha, X, 1);
}

template<>
void sapien_scal<double>(const int N, const double alpha, double* X) {
  cblas_dscal(N, alpha, X, 1);
}

// Compute the dot product of two vectors.

template<>
float sapien_dot<float>(const int N, const float* X, const float* Y) {
  return cblas_sdot(N, X, 1, Y, 1);
}

template<>
double sapien_dot<double>(const int N, const double* X, const double* Y) {
  return cblas_ddot(N, X, 1, Y, 1);
}

// Compute C <- alpha * A * B + beta * C
//
// A[M, K] ~ A[K][M].
// C[M, N]
// B[K, N]

template<>
void sapien_gemm<float>(const MatrixTransposeType TransA,
                        const MatrixTransposeType TransB,
                        const int M, const int N, const int K,
                        const float alpha, const float* A, const float* B,
                        const float beta, float* C) {
  int lda = (TransA == SAPIEN_BLAS_NO_TRANS) ? K : M;
  int ldb = (TransB == SAPIEN_BLAS_NO_TRANS) ? N : K;
  cblas_sgemm(CblasRowMajor,
              MatrixTransposeTypeToCblasTranspose(TransA),
              MatrixTransposeTypeToCblasTranspose(TransB),
              M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

template<>
void sapien_gemm<double>(const MatrixTransposeType TransA,
                         const MatrixTransposeType TransB,
                         const int M, const int N, const int K,
                         const double alpha, const double* A,
                         const double* B, const double beta,
                         double* C) {
  int lda = (TransA == SAPIEN_BLAS_NO_TRANS) ? K : M;
  int ldb = (TransB == SAPIEN_BLAS_NO_TRANS) ? N : K;
  cblas_dgemm(CblasRowMajor,
              MatrixTransposeTypeToCblasTranspose(TransA),
              MatrixTransposeTypeToCblasTranspose(TransB),
              M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

// y <- alpha * A * x + beta * y.

template<>
void sapien_gemv<float>(const MatrixTransposeType TransA, const int M,
                        const int N, const float alpha, const float* A,
                        const float* x, const float beta, float* y) {
  cblas_sgemv(CblasRowMajor,
              MatrixTransposeTypeToCblasTranspose(TransA),
              M, N, alpha, A, N, x, 1, beta, y, 1);
}

template<>
void sapien_gemv<double>(const MatrixTransposeType TransA, const int M,
                         const int N, const double alpha, const double* A,
                         const double* x, const double beta, double* y) {
  cblas_dgemv(CblasRowMajor,
              MatrixTransposeTypeToCblasTranspose(TransA),
              M, N, alpha, A, N, x, 1, beta, y, 1);
}

// Compute L2-norm of a vector

template<>
float sapien_nrm2(const int N, const float* X) {
  return cblas_snrm2(N, X, 1);
}

template<>
double sapien_nrm2(const int N, const double* X) {
  return cblas_dnrm2(N, X, 1);
}

// Modifies a vector (single or double precision) inplace, setting
// each element to a given value.

template<typename T>
void sapien_set(const int N, const T alpha, T* X) {
  if (alpha == 0) {
    std::memset(X, 0, sizeof(T) * N);
    return;
  }

  int i, n;
  n = N;
  i = n >> 5;  /* multiple of 32 (1 << 5) */

  if (i) {
    n -= (i << 5);
    do {
      *X = X[1] = X[2] = X[3] = X[4] = X[5] = X[6] = X[7] = X[8] = X[9] =
          X[10] = X[11] = X[12] = X[13] = X[14] = X[15] = X[16] = X[17] =
          X[18] = X[19] = X[20] = X[21] = X[22] = X[23] = X[24] = X[25] =
          X[26] = X[27] = X[28] = X[29] = X[30] = X[31] = alpha;
      X += 32;
    } while (--i);
  }

  if (n >> 4) {  /* >= 16 */
    *X = X[1] = X[2] = X[3] = X[4] = X[5] = X[6] = X[7] = X[8] = X[9] =
        X[10] = X[11] = X[12] = X[13] = X[14] = X[15] = alpha;
    X += 16;
    n -= 16;
  }

  if (n >> 3) { /* >= 8 */
    *X = X[1] = X[2] = X[3] = X[4] = X[5] = X[6] = X[7] = alpha;
    X += 8;
    n -= 8;
  }

  switch (n) {  /* left over */
    case 1:
      *X = alpha;
      break;
    case 2:
      *X = X[1] = alpha;
      break;
    case 3:
      *X = X[1] = X[2] = alpha;
      break;
    case 4:
      *X = X[1] = X[2] = X[3] = alpha;
      break;
    case 5:
      *X = X[1] = X[2] = X[3] = X[4] = alpha;
      break;
    case 6:
      *X = X[1] = X[2] = X[3] = X[4] = X[5] = alpha;
      break;
    case 7:
      *X = X[1] = X[2] = X[3] = X[4] = X[5] = X[6] = alpha;
      break;
    default:
      break;
  }
}

template void sapien_set<int>(const int N, const int alpha, int* X);
template void sapien_set<float>(const int N, const float alpha, float* X);
template void sapien_set<double>(const int N, const double alpha, double* X);


// Copies a source vector to a destination vector
// TODO(Linh): Use memcopy or cblas_[s,d]copy?
// template<typename T>
// void sapien_copy(const int N, const T* src, T* dst) {
//   memcpy(dst, src, sizeof(T) * N);
// }

template<>
void sapien_copy<float>(const int N, const float* src, float* dst) {
  cblas_scopy(N, src, 1, dst, 1);
}

template<>
void sapien_copy<double>(const int N, const double* src, double* dst) {
  cblas_dcopy(N, src, 1, dst, 1);
}

// Compute z = x - y;

template<>
void sapien_xmy<float>(const int N, const float* x, const float* y,
                       float* z) {
  cblas_scopy(N, x, 1, z, 1);  // z = x
  cblas_saxpy(N, -1.0, y, 1, z, 1);  // z = -1 * y + z
}

template<>
void sapien_xmy<double>(const int N, const double* x, const double* y,
                        double* z) {
  cblas_dcopy(N, x, 1, z, 1);  // z = x
  cblas_daxpy(N, -1.0, y, 1, z, 1);  // z = -1 * y + z
}

// Compute z = x + y.

template<>
void sapien_xpy<float>(const int N, const float* x, const float* y,
                       float* z) {
  cblas_scopy(N, x, 1, z, 1);
  cblas_saxpy(N, 1.0, y, 1, z, 1);
}

template<>
void sapien_xpy<double>(const int N, const double* x, const double* y,
                        double* z) {
  cblas_dcopy(N, x, 1, z, 1);
  cblas_daxpy(N, 1.0, y, 1, z, 1);
}

// Compute y = x + alpha

template<typename T>
void sapien_xpa(const int N, const T alpha, const T* x, T* y) {
  int i, j;
  for (i = 0, j = 1; j < N; i = i + 2, j = j + 2) {
    y[i] = x[i] + alpha;
    y[j] = x[j] + alpha;
  }
  if (i < N) {
    y[i] = x[i] + alpha;
  }
}

template void sapien_xpa<float>(const int N, const float alpha,
                                const float* x, float* y);
template void sapien_xpa<double>(const int N, const double alpha,
                                 const double* x, double* y);

// Compute mean(x)

template<typename T>
T sapien_vmean(const int N, const T* x) {
  T sum = 0.0;
  int i, n;
  n = N;
  i = n >> 5;

  if (i) {
    n -= (i << 5);
    do {
      sum += (*x + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] +
              x[9] + x[10] + x[11] + x[12] + x[13] + x[14] + x[15] + x[16] +
              x[17] + x[18] + x[19] + x[20] + x[21] + x[22] + x[23] + x[24] +
              x[25] + x[26] + x[27] + x[28] + x[29] + x[30] + x[31]);
      x += 32;
    } while (--i);
  }

  if (n >> 4) {
    sum += (*x + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] +
            x[9] + x[10] + x[11] + x[12] + x[13] + x[14] + x[15]);
    n -= 16;
    x += 16;
  }

  if (n >> 3) {
    sum += (*x + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7]);
    n -= 8;
    x += 8;
  }

  switch (n) {
    case 1:
      sum += *x;
      break;
    case 2:
      sum += (*x + x[1]);
      break;
    case 3:
      sum += (*x + x[1] + x[2]);
      break;
    case 4:
      sum += (*x + x[1] + x[2] + x[3]);
      break;
    case 5:
      sum += (*x + x[1] + x[2] + x[3] + x[4]);
      break;
    case 6:
      sum += (*x + x[1] + x[2] + x[3] + x[4] + x[5]);
      break;
    case 7:
      sum += (*x + x[1] + x[2] + x[3] + x[4] + x[5] + x[6]);
      break;
    default:
      break;
  }

  return sum / static_cast<T>(N);
}

template float sapien_vmean<float>(const int N, const float* x);
template double sapien_vmean<double>(const int N, const double* x);

// Are all elements in an array finite? (i.e not NaN nor Inf).

template<> bool sapien_allfinite<float>(const int N, const float* X) {
  for (int i = 0; i < N; ++i) {
    if (sapien_isinf(X[i]) || sapien_isnan(X[i])) return false;
  }
  return true;
}

template<> bool sapien_allfinite<double>(const int N, const double* X) {
  for (int i = 0; i < N; ++i) {
    if (sapien_isinf(X[i]) || sapien_isnan(X[i])) return false;
  }
  return true;
}

// Returns the index of maximun value in an array.

template<> int sapien_imax(const int N, const float* X) {
  int ret = 0;
  float max = *X;
  float temp;

  for (int i = 1; i < N; ++i) {
    temp = X[i];
    if (temp > max) {
      ret = i;
      max = temp;
    }
  }
  return ret;
}

template<> int sapien_imax(const int N, const double* X) {
  int ret = 0;
  double max = *X;
  double temp;

  for (int i = 1; i < N; ++i) {
    temp = X[i];
    if (temp > max) {
      ret = i;
      max = temp;
    }
  }
  return ret;
}

// Compute x^T * Diag * y, in which
//
//  x, y are two n-dimensional vectors, Diag is N by N diagonal matrix.

#define X_DIAG_Y_TERM(i) (x[i] * diag[i] * y[i])

template<typename T>
T sapien_xDiagy(const int N, const T* x, const T* diag, const T* y) {
  T ret = T(0.0);
  int n, i;
  n = N;
  i = n >> 5;  // n = 32 * i + left_over

  if (i) {
    n -= (i << 5);
    do {
      ret += (X_DIAG_Y_TERM(0) + X_DIAG_Y_TERM(1) + X_DIAG_Y_TERM(2) +
              X_DIAG_Y_TERM(3) + X_DIAG_Y_TERM(4) + X_DIAG_Y_TERM(5) +
              X_DIAG_Y_TERM(6) + X_DIAG_Y_TERM(7) + X_DIAG_Y_TERM(8) +
              X_DIAG_Y_TERM(9) + X_DIAG_Y_TERM(10) + X_DIAG_Y_TERM(11) +
              X_DIAG_Y_TERM(12) + X_DIAG_Y_TERM(13) + X_DIAG_Y_TERM(14) +
              X_DIAG_Y_TERM(15) + X_DIAG_Y_TERM(16) + X_DIAG_Y_TERM(17) +
              X_DIAG_Y_TERM(18) + X_DIAG_Y_TERM(19) + X_DIAG_Y_TERM(20) +
              X_DIAG_Y_TERM(21) + X_DIAG_Y_TERM(22) + X_DIAG_Y_TERM(23) +
              X_DIAG_Y_TERM(24) + X_DIAG_Y_TERM(25) + X_DIAG_Y_TERM(26) +
              X_DIAG_Y_TERM(27) + X_DIAG_Y_TERM(28) + X_DIAG_Y_TERM(29) +
              X_DIAG_Y_TERM(30) + X_DIAG_Y_TERM(31));
      x += 32;
      diag += 32;
      y += 32;
    } while (--i);
  }

  if (n >> 4) {  // n = 16 + left_over
    ret += (X_DIAG_Y_TERM(0) + X_DIAG_Y_TERM(1) + X_DIAG_Y_TERM(2) +
            X_DIAG_Y_TERM(3) + X_DIAG_Y_TERM(4) + X_DIAG_Y_TERM(5) +
            X_DIAG_Y_TERM(6) + X_DIAG_Y_TERM(7) + X_DIAG_Y_TERM(8) +
            X_DIAG_Y_TERM(9) + X_DIAG_Y_TERM(10) + X_DIAG_Y_TERM(11) +
            X_DIAG_Y_TERM(12) + X_DIAG_Y_TERM(13) + X_DIAG_Y_TERM(14) +
            X_DIAG_Y_TERM(15));
    n -= 16;
    x += 16;
    diag += 16;
    y += 16;
  }

  if (n >> 3) {  // n = 8 + left_over
    ret += (X_DIAG_Y_TERM(0) + X_DIAG_Y_TERM(1) + X_DIAG_Y_TERM(2) +
            X_DIAG_Y_TERM(3) + X_DIAG_Y_TERM(4) + X_DIAG_Y_TERM(5) +
            X_DIAG_Y_TERM(6) + X_DIAG_Y_TERM(7));
    n -= 8;
    x += 8;
    diag += 8;
    y += 8;
  }

  // left_over
  switch (n) {
    case 1:
      ret += X_DIAG_Y_TERM(0);
      break;
    case 2:
      ret += (X_DIAG_Y_TERM(0) + X_DIAG_Y_TERM(1));
      break;
    case 3:
      ret += (X_DIAG_Y_TERM(0) + X_DIAG_Y_TERM(1) + X_DIAG_Y_TERM(2));
      break;
    case 4:
      ret += (X_DIAG_Y_TERM(0) + X_DIAG_Y_TERM(1) + X_DIAG_Y_TERM(2) +
              X_DIAG_Y_TERM(3));
      break;
    case 5:
      ret += (X_DIAG_Y_TERM(0) + X_DIAG_Y_TERM(1) + X_DIAG_Y_TERM(2) +
              X_DIAG_Y_TERM(3) + X_DIAG_Y_TERM(4));
      break;
    case 6:
      ret += (X_DIAG_Y_TERM(0) + X_DIAG_Y_TERM(1) + X_DIAG_Y_TERM(2) +
              X_DIAG_Y_TERM(3) + X_DIAG_Y_TERM(4) + X_DIAG_Y_TERM(5));
      break;
    case 7:
      ret += (X_DIAG_Y_TERM(0) + X_DIAG_Y_TERM(1) + X_DIAG_Y_TERM(2) +
              X_DIAG_Y_TERM(3) + X_DIAG_Y_TERM(4) + X_DIAG_Y_TERM(5) +
              X_DIAG_Y_TERM(6));
      break;
    default:
      break;
  }

  return ret;
}

template float sapien_xDiagy(const int N, const float* x, const float* diag,
                             const float* y);
template double sapien_xDiagy(const int N, const double* x,
                              const double* diag, const double* y);
}  // namespace internal
}  // namespace sapien

