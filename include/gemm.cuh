#include <cuda.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda/barrier>
#include <mma.h>
#include <cstdio>
#include <cuda_fp16.h>
// Disables `cuda::barrier` initialization warning.
#pragma nv_diag_suppress static_var_with_dynamic_init

inline int idx_row_major(int r, int c, int ld) {
    return r * ld + c;
}
inline int idx_col_major(int r, int c, int ld) {
    return c * ld + r;
}
template <typename T, typename ACC>
void gemm_cpu(T *A, T *B, ACC *C, int M, int N, int K, bool isARowMajor, bool isBRowMajor) {
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      ACC sum = 0;
      for (int i = 0; i < K; i++) {
        T a = isARowMajor
            ? A[idx_row_major(r, i, K)]
            : A[idx_col_major(r, i, M)];

        T b = isBRowMajor
            ? B[idx_row_major(i, c, N)]
            : B[idx_col_major(i, c, K)];

        sum += ACC(a) * ACC(b);
      }
      C[r * N + c] = sum;  // C is row-major
    }
  }
}

// C = A*B
// A is MxK, B is KxN, C is MxN
// Everything is row-major
__global__ void gemm(int8_t *A, int8_t *B, int *C, int M, int N, int K);

__global__ void gemm_tensorop_row_col(int8_t *A, int8_t *B, int *C, int M, int N, int K);

__global__ void gemm_tensorop_pipelined(int8_t *A, int8_t *B, int *C, int M, int N, int K);

__global__ void mma_m16n8k16_s8_s8(int8_t *A, int8_t *B, int32_t *C, int M, int N, int K);

__global__ void mma_m16n8k16_s8_s8_memcpy_async(int8_t *A, int8_t *B, int32_t *C, int M, int N, int K);

__global__ void mma_m16n8k16_s8_s8_pipelined_row_col(int8_t *A, int8_t *B, int32_t *C, int M, int N, int K);

__global__ void mma_m16n8k16_f16_f16(__half *A, __half *B, float *C, int M, int N, int K);
