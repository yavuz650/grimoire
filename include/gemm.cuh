#include <cuda.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda/barrier>
#include <mma.h>
#include <cstdio>
// Disables `cuda::barrier` initialization warning.
#pragma nv_diag_suppress static_var_with_dynamic_init

void gemm_cpu(int8_t *A, int8_t *B, int *C, int M, int N, int K);

// C = A*B
// A is MxK, B is KxN, C is MxN
// Everything is row-major
__global__ void gemm(int8_t *A, int8_t *B, int *C, int M, int N, int K);

__global__ void gemm_tensorop(int8_t *A, int8_t *B, int *C, int M, int N, int K);

__global__ void gemm_tensorop_pipelined(int8_t *A, int8_t *B, int *C, int M, int N, int K);

__global__ void mma_m16n8k16_s8_s8(int8_t *A, int8_t *B, int32_t *C, int M, int N, int K);

__global__ void mma_m16n8k16_s8_s8_memcpy_async(int8_t *A, int8_t *B, int32_t *C, int M, int N, int K);
