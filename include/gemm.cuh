//#include "gemm.cuh"
//#include "mma_intrinsics.cuh"
#ifndef __GEMM_CUH__
#define __GEMM_CUH__

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

#include "mma_intrinsics.cuh"
enum class Layout {
  RowMajor,
  ColMajor
};

template <Layout L>
__device__ __forceinline__
int getIdx(int r, int c, int ld) {
  if constexpr (L == Layout::RowMajor)
    return r * ld + c;
  else
    return c * ld + r;
}

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
__global__ void gemm(int8_t *A, int8_t *B, int *C, int M, int N, int K) {
  int r = blockDim.y*blockIdx.y+threadIdx.y;
  int c = blockDim.x*blockIdx.x+threadIdx.x;

  int sum = 0;
  for (int i = 0; i < K; i++) {
    sum += A[K*r+i] * B[i*N+c];
  }

  C[N*r+c] = sum;
}

// Doesn't work
__global__ void gemm_tensorop_pipelined(int8_t *A, int8_t *B, int *C, int M, int N, int K) {
  using namespace nvcuda;
  constexpr int stagesCount = 2;
  // warpidx = idx within the block(threadidx.x / 32) + number of warps before this warp(blockidx.x*blockDim.x/32)
  int warpIdx = threadIdx.x / 32 + blockIdx.x*blockDim.x/32;
  int warpRow = warpIdx / (N/16);
  int warpCol = warpIdx % (N/16);
  // warp index within the thread block
  int localWarpIdx = threadIdx.x / 32;

  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  //assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz * grid_size

  // Two batches must fit in shared memory:
  __shared__ int8_t shared[16*16*2*stagesCount*256/32]; // 16*16*2*stagesCount*block.num_threads()/32
  size_t sharedOffsetA[stagesCount] = { localWarpIdx*512, localWarpIdx*512 + 512*8 }; // Offsets to each batch
  size_t sharedOffsetB[stagesCount] = { localWarpIdx*512 + 256, localWarpIdx*512 + 512*8 + 256 }; // Offsets to each batch

  // Allocate shared storage for a two-stage cuda::pipeline:
  __shared__ cuda::pipeline_shared_state<
      cuda::thread_scope::thread_scope_block,
      stagesCount
  > sharedState;
  auto pipeline = cuda::make_pipeline(block, &sharedState);

  // Each thread processes `batch_sz` elements.
  // Compute offset of the batch `batch` of this thread block in global memory:
  auto warp_batch_A = [&](size_t batch) -> int {
    return warpRow*K*16 + batch*16;
  };
  auto warp_batch_B = [&](size_t batch) -> int {
    return warpCol*16 + batch*N*16;
  };

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag[stagesCount];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::row_major> b_frag[stagesCount];
  wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
  // Initialize the output to zero
  wmma::fill_fragment(c_frag, 0);

  // Initialize first pipeline stage by submitting a `memcpy_async` to fetch a whole batch for A and B:
  // if (batch_sz == 0) return;
  pipeline.producer_acquire();
  cuda::memcpy_async(block, shared + sharedOffsetA[0], A + warp_batch_A(0), sizeof(int8_t) * 256, pipeline);
  cuda::memcpy_async(block, shared + sharedOffsetB[0], B + warp_batch_B(0), sizeof(int8_t) * 256, pipeline);
  pipeline.producer_commit();

  // int offsetA = warpRow*K*16;
  // int offsetB = warpCol*16;
  int offsetC = warpRow*N*16 + warpCol*16;

  for (int batch = 1; batch < K/16; batch++) {
    // Stage indices for the compute and copy stages:
    size_t computeStageIdx = (batch - 1) % 2;
    size_t copyStageIdx = batch % 2;

    // Collectively acquire the pipeline head stage from all producer threads:
    pipeline.producer_acquire();
    // Submit async copies to the pipeline's head stage to be
    // computed in the next loop iteration
    cuda::memcpy_async(block, shared + sharedOffsetA[copyStageIdx], A + warp_batch_A(batch), sizeof(int8_t) * 256, pipeline);
    cuda::memcpy_async(block, shared + sharedOffsetB[copyStageIdx], B + warp_batch_B(batch), sizeof(int8_t) * 256, pipeline);
    // Collectively commit (advance) the pipeline's head stage
    pipeline.producer_commit();

    // Collectively wait for the operations committed to the
    // previous `compute` stage to complete:
    pipeline.consumer_wait();

    // Computation overlapped with the memcpy_async of the "copy" stage:
    // Load the inputs
    wmma::load_matrix_sync(a_frag[computeStageIdx], shared + sharedOffsetA[computeStageIdx], 16);
    wmma::load_matrix_sync(b_frag[computeStageIdx], shared + sharedOffsetB[computeStageIdx], 16);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag[computeStageIdx], b_frag[computeStageIdx], c_frag);
    // Collectively release the stage resources
    pipeline.consumer_release();
  }
  // Compute the data fetch by the last iteration
  pipeline.consumer_wait();
  wmma::load_matrix_sync(a_frag[(K/16 - 1) %2], shared + sharedOffsetA[(K/16 - 1) %2], 16);
  wmma::load_matrix_sync(b_frag[(K/16 - 1) %2], shared + sharedOffsetB[(K/16 - 1) %2], 16);
  // Perform the matrix multiplication
  wmma::mma_sync(c_frag, a_frag[(K/16 - 1) %2], b_frag[(K/16 - 1) %2], c_frag);
  pipeline.consumer_release();
  // Store the output
  wmma::store_matrix_sync(C+offsetC, c_frag, N, wmma::mem_row_major);
}

// A is row major, B is col major
__global__ void gemm_tensorop_row_col(int8_t *A, int8_t *B, int *C, int M, int N, int K) {
  using namespace nvcuda;
  // warpidx = idx within the block(threadidx.x / 32) + number of warps before this warp(blockidx.x*blockDim.x/32)
  int warpIdx = threadIdx.x / 32 + blockIdx.x*blockDim.x/32;
  int warpRow = warpIdx / (N/16);
  int warpCol = warpIdx % (N/16);

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
  // Initialize the output to zero
  wmma::fill_fragment(c_frag, 0);

  int offsetA = warpRow*K*16;
  int offsetB = warpCol*K*16;
  int offsetC = warpRow*N*16 + warpCol*16;

  for (int i = 0; i < K/16; i++) {
    // Load the inputs
    wmma::load_matrix_sync(a_frag, A + offsetA + i*16, K);
    wmma::load_matrix_sync(b_frag, B + offsetB + i*16, K);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  // Store the output
  wmma::store_matrix_sync(C+offsetC, c_frag, N, wmma::mem_row_major);
}

// A is row major, B is column major, C is row major
__global__ void mma_m16n8k16_s8_s8(int8_t *A, int8_t *B, int32_t *C, int M, int N, int K) {
  constexpr int m=16;
  constexpr int n=8;
  constexpr int k=16;
  int globalWarpIdx = threadIdx.x / 32 + blockIdx.x*blockDim.x/32;
  int warpRow = globalWarpIdx / (N/n);
  int warpCol = globalWarpIdx % (N/n);
  // Warp index within the CTA
  int localWarpIdx = threadIdx.x / 32;

  // Assuming 8 warps in each CTA
  __shared__ int8_t smemA[m*k *8];
  __shared__ int8_t smemB[k*n *8];
  __shared__ int32_t smemC[m*n *8];
  int smemAIdx = localWarpIdx*m*k;
  int smemBIdx = localWarpIdx*k*n;
  int smemCIdx = localWarpIdx*m*n;
  int laneId = threadIdx.x & 31;

  for (int i = laneId; i < m*n; i += 32)
    smemC[smemCIdx + i] = 0;
  __syncwarp();

  int row, col;
  for (int tileIdx = 0; tileIdx < K/16; tileIdx++) {
    // Each thread loads one element in a row/col
    if(laneId < 16) {
      for (int i = 0; i < 16; i++) {
        row = warpRow*m + i;
        col = tileIdx*k + laneId;
        smemA[smemAIdx + i*16 + laneId] = A[row*K+col];
        if(i < 8) {
          row = tileIdx*k + laneId;
          col = warpCol*n + i;
          smemB[smemBIdx + i*16 + laneId] = B[col*K + row];
        }
      }
    }

    __syncwarp();
    mma_m16n8k16_s8_s8_smem_row_col(smemA+smemAIdx, smemB+smemBIdx, smemC+smemCIdx);
    __syncwarp();
  }

  // Each thread stores one element in a row
  if(laneId < 8) {
    for (int i = 0; i < 16; i++) {
      row = warpRow*m + i;
      col = warpCol*n +laneId;
      C[row*N + col] =  smemC[smemCIdx + i*8 + laneId];
    }
  }
}

// A is row major, B is column major, C is row major
__global__ void mma_m16n8k16_s8_s8_memcpy_async(int8_t *A, int8_t *B, int32_t *C, int M, int N, int K) {
  using namespace nvcuda;
  constexpr int m=16;
  constexpr int n=8;
  constexpr int k=16;

  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  auto warp = cooperative_groups::tiled_partition<32>(block);

  int globalWarpIdx = threadIdx.x / 32 + blockIdx.x*blockDim.x/32;
  int warpRow = globalWarpIdx / (N/n);
  int warpCol = globalWarpIdx % (N/n);
  // Warp index within the CTA
  int localWarpIdx = threadIdx.x / 32;

  // Assuming 8 warps in each CTA
  __shared__ int8_t smemA[m*k *8];
  __shared__ int8_t smemB[k*n *8];
  __shared__ int32_t smemC[m*n *8];
  int smemAIdx = localWarpIdx*m*k;
  int smemBIdx = localWarpIdx*k*n;
  int smemCIdx = localWarpIdx*m*n;
  int laneId = threadIdx.x & 31;

  for (int i = laneId; i < m*n; i += 32)
    smemC[smemCIdx + i] = 0;
  __syncwarp();

  int row, col;
  for (int tileIdx = 0; tileIdx < K/16; tileIdx++) {
    // Each thread loads one element in a row/col
    //if(laneId < 16) {
      for (int i = 0; i < 16; i++) {
        row = warpRow*m + i;
        //col = tileIdx*k + laneId;
        col = tileIdx*k;
        //smemA[smemAIdx + i*16 + laneId] = A[row*K+col];
        cooperative_groups::memcpy_async(warp, smemA+smemAIdx+i*16, A+row*K+col, sizeof(int8_t) * 16);
        cooperative_groups::wait(warp); // Joins all threads, waits for all copies to complete

        if(i < 8 && laneId < 16) {
          row = tileIdx*k + laneId;
          col = warpCol*n + i;
          smemB[smemBIdx + i*16 + laneId] = B[col*K + row];
        }
      }
    //}

    __syncwarp();
    mma_m16n8k16_s8_s8_smem_row_col(smemA+smemAIdx, smemB+smemBIdx, smemC+smemCIdx);
    __syncwarp();
  }

  // Each thread stores one element in a row
  if(laneId < 8) {
    for (int i = 0; i < 16; i++) {
      row = warpRow*m + i;
      col = warpCol*n +laneId;
      C[row*N + col] =  smemC[smemCIdx + i*8 + laneId];
    }
  }
}

// A is row major, B is column major, C is row major
// Each CTA calculates a 32x32 tile of the output
__global__ void mma_m16n8k16_s8_s8_pipelined_row_col(int8_t *A, int8_t *B, int32_t *C, int M, int N, int K) {
  using namespace nvcuda;
  constexpr int m=16;
  constexpr int n=8;
  constexpr int k=16;
  constexpr int stagesCount = 2;

  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  auto warp = cooperative_groups::tiled_partition<32>(block);

  // Two batches must fit in shared memory:
  //extern __shared__ int shared[]; // *2 for A and B, stagesCount*(16*16*2 + 16*32)
  size_t sharedOffsetA[stagesCount] = { 0, 16*32 }; // Offsets to each batch
  size_t sharedOffsetB[stagesCount] = { 0, 16*32 }; // Offsets to each batch

  // Allocate shared storage for a two-stage cuda::pipeline:
  __shared__ cuda::pipeline_shared_state<
      cuda::thread_scope::thread_scope_block,
      stagesCount
  > sharedState;
  auto pipeline = cuda::make_pipeline(block, &sharedState);

  //int globalWarpIdx = threadIdx.x / 32 + blockIdx.x*blockDim.x/32;
  // Warp index within the CTA
  int localWarpIdx = threadIdx.x / 32;

  int blockRow = block.group_index().x / (N/32);
  int blockCol = block.group_index().x % (N/32);
  int warpRow = blockRow*2 + localWarpIdx / 4;
  int warpCol = blockCol*4 + localWarpIdx % 4;

  // Assuming 8 warps in each CTA
  __shared__ __align__(16) int8_t smemA[m*k*2 * stagesCount];
  __shared__ __align__(16) int8_t smemB[k*n*4 * stagesCount];
  __shared__ __align__(16) int32_t smemC[m*n *8];
  int smemAIdx = localWarpIdx*m*k;
  int smemBIdx = localWarpIdx*k*n;
  int smemCIdx = localWarpIdx*m*n;
  int laneId = threadIdx.x & 31;
  int row, col;

  for (int i = laneId; i < m*n; i += 32)
    smemC[smemCIdx + i] = 0;
  block.sync();

  // Initialize first pipeline stage by submitting a `memcpy_async` to fetch a whole batch for A and B:
  // if (batch_sz == 0) return;
  pipeline.producer_acquire();
  for (int i = 0; i < 32; i++) {
    row = blockRow*32 + i;
    cuda::memcpy_async(block, smemA+sharedOffsetA[0] + i*16, A+row*K, sizeof(int8_t) * 16, pipeline);
    col = blockCol*32 + i;
    cuda::memcpy_async(block, smemB+sharedOffsetB[0] + i*16, B+col*K, sizeof(int8_t) * 16, pipeline);
  }
  pipeline.producer_commit();

  size_t computeStageIdx;
  size_t copyStageIdx;
  size_t batch;
  // Pipelined copy/compute:
  for (batch = 1; batch < K/16; ++batch) {
    // Stage indices for the compute and copy stages:
    computeStageIdx = (batch - 1) % 2;
    copyStageIdx = batch % 2;

    // Collectively acquire the pipeline head stage from all producer threads:
    pipeline.producer_acquire();

    // Submit async copies to the pipeline's head stage to be
    // computed in the next loop iteration
    for (int i = 0; i < 32; i++) {
      row = blockRow*32 + i;
      col = batch*16;
      cuda::memcpy_async(block, smemA+sharedOffsetA[copyStageIdx] + i*16, A + row*K + col, sizeof(int8_t) * 16, pipeline);
      row = batch*16;
      col = blockCol*32 + i;
      cuda::memcpy_async(block, smemB+sharedOffsetB[copyStageIdx] + i*16, B + row + col*K, sizeof(int8_t) * 16, pipeline);
    }
    // Collectively commit (advance) the pipeline's head stage
    pipeline.producer_commit();

    // Collectively wait for the operations committed to the
    // previous `compute` stage to complete:
    pipeline.consumer_wait();
    // Computation overlapped with the memcpy_async of the "copy" stage:
    mma_m16n8k16_s8_s8_smem_row_col(smemA+sharedOffsetA[computeStageIdx]+(localWarpIdx/4)*16*16, smemB+sharedOffsetB[computeStageIdx]+(localWarpIdx%4)*16*8, smemC+smemCIdx);

    // Collectively release the stage resources
    pipeline.consumer_release();
  }
  computeStageIdx = (batch - 1) % 2;
  // Compute the data fetch by the last iteration
  pipeline.consumer_wait();
  mma_m16n8k16_s8_s8_smem_row_col(smemA+sharedOffsetA[computeStageIdx]+(localWarpIdx/4)*16*16, smemB+sharedOffsetB[computeStageIdx]+(localWarpIdx%4)*16*8, smemC+smemCIdx);
  pipeline.consumer_release();  

  // Each thread stores one element in a row
  // for (int i = 0; i < 32; i++) {
  //   row = blockRow*32 + i;
  //   col = blockCol*32;
  //   cuda::memcpy_async(block, C+row*N+col, smemC + i*32, sizeof(int32_t) * 32, pipeline);
  // }  
  if(laneId < 8) {
    for (int i = 0; i < 16; i++) {
      row = warpRow*m + i;
      col = warpCol*n +laneId;
      C[row*N + col] =  smemC[smemCIdx + i*8 + laneId];
    }
  }
}


template <Layout LayoutA, Layout LayoutB>
__global__ void mma_m16n8k16_f16_f16(__half *A, __half *B, float *C, int M, int N, int K) {
  constexpr int m=16;
  constexpr int n=8;
  constexpr int k=16;
  int globalWarpIdx = threadIdx.x / 32 + blockIdx.x*blockDim.x/32;
  int warpRow = globalWarpIdx / (N/n);
  int warpCol = globalWarpIdx % (N/n);
  // Warp index within the CTA
  int localWarpIdx = threadIdx.x / 32;

  // Assuming 8 warps in each CTA
  __shared__ __half smemA[m*k *8];
  __shared__ __half smemB[k*n *8];
  __shared__ float smemC[m*n *8];
  int smemAIdx = localWarpIdx*m*k;
  int smemBIdx = localWarpIdx*k*n;
  int smemCIdx = localWarpIdx*m*n;
  int laneId = threadIdx.x & 31;

  for (int i = laneId; i < m*n; i += 32)
    smemC[smemCIdx + i] = 0;
  __syncwarp();

  int row, col;
  for (int tileIdx = 0; tileIdx < K/16; tileIdx++) {
    // Each thread loads one element in a row/col
    if(laneId < 16) {
      for (int i = 0; i < 16; i++) {
        row = warpRow*m + i;
        col = tileIdx*k + laneId;
        smemA[smemAIdx + i*16 + laneId] = A[getIdx<LayoutA>(row, col, (LayoutA == Layout::RowMajor ? K : M))];
        if(i < 8) {
          row = tileIdx*k + laneId;
          col = warpCol*n + i;
          smemB[smemBIdx + i*16 + laneId] = B[getIdx<LayoutB>(row, col, (LayoutB == Layout::RowMajor ? N : K))];
        }
      }
    }

    __syncwarp();
    mma_m16n8k16_f16_f16_smem_row_col(smemA+smemAIdx, smemB+smemBIdx, smemC+smemCIdx);
    __syncwarp();
  }

  // Each thread stores one element in a row
  if(laneId < 8) {
    for (int i = 0; i < 16; i++) {
      row = warpRow*m + i;
      col = warpCol*n +laneId;
      C[row*N + col] =  smemC[smemCIdx + i*8 + laneId];
    }
  }
}


// A is col, B is row 
__global__ void mma_m16n8k16_f16_f16_pipelined_NT(__half *A, __half *B, float *C, int M, int N, int K) {
  using namespace nvcuda;
  constexpr int m=16;
  constexpr int n=8;
  constexpr int k=16;
  constexpr int stagesCount = 2;

  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();

  // Two batches must fit in shared memory:
  //extern __shared__ int shared[]; // *2 for A and B, stagesCount*(16*16*2 + 16*32)
  size_t sharedOffsetA[stagesCount] = { 0, 16*32 }; // Offsets to each batch
  size_t sharedOffsetB[stagesCount] = { 0, 16*32 }; // Offsets to each batch

  // Allocate shared storage for a two-stage cuda::pipeline:
  __shared__ cuda::pipeline_shared_state<
      cuda::thread_scope::thread_scope_block,
      stagesCount
  > sharedState;
  auto pipeline = cuda::make_pipeline(block, &sharedState);

  int blockRow = block.group_index().x / (N/32);
  int blockCol = block.group_index().x % (N/32);
  // Warp index within the CTA
  int localWarpIdx = threadIdx.x / 32;
  int warpRow = blockRow*2 + localWarpIdx / 4;
  int warpCol = blockCol*4 + localWarpIdx % 4;  

  // Assuming 8 warps in each CTA
  __shared__ __align__(16) __half smemA[m*k*2 *stagesCount];
  __shared__ __align__(16) __half smemB[k*n*4 *stagesCount];
  __shared__ __align__(16) float smemC[m*n *8];
  int smemCIdx = localWarpIdx*m*n;
  int laneId = threadIdx.x & 31;

  for (int i = laneId; i < m*n; i += 32)
    smemC[smemCIdx + i] = 0;
  block.sync();

  int row, col;
  // Initialize first pipeline stage by submitting a `memcpy_async` to fetch a whole batch for A and B:
  // if (batch_sz == 0) return;
  pipeline.producer_acquire();
  for (int i = 0; i < 16; i++) {
    row = blockRow*32;
    col = i;
    cuda::memcpy_async(block, smemA+sharedOffsetA[0] + i*32, A+col*M+row, sizeof(__half) * 32, pipeline);
    row = i;
    col = blockCol*32;
    cuda::memcpy_async(block, smemB+sharedOffsetB[0] + i*32, B+row*N+col, sizeof(__half) * 32, pipeline);
  }
  pipeline.producer_commit();

  size_t computeStageIdx;
  size_t copyStageIdx;
  size_t batch;
  // Pipelined copy/compute:
  for (batch = 1; batch < K/16; ++batch) {
    // Stage indices for the compute and copy stages:
    computeStageIdx = (batch - 1) % stagesCount;
    copyStageIdx = batch % stagesCount;

    // Collectively acquire the pipeline head stage from all producer threads:
    pipeline.producer_acquire();

    // Submit async copies to the pipeline's head stage to be
    // computed in the next loop iteration
    for (int i = 0; i < 16; i++) {
      row = blockRow*32;
      col = batch*16 + i;
      cuda::memcpy_async(block, smemA+sharedOffsetA[copyStageIdx] + i*32, A + col*M + row, sizeof(__half) * 32, pipeline);
      row = batch*16 + i;
      col = blockCol*32;
      cuda::memcpy_async(block, smemB+sharedOffsetB[copyStageIdx] + i*32, B + row*N + col, sizeof(__half) * 32, pipeline);
    }
    // Collectively commit (advance) the pipeline's head stage
    pipeline.producer_commit();

    // Collectively wait for the operations committed to the
    // previous `compute` stage to complete:
    pipeline.consumer_wait();
    // Computation overlapped with the memcpy_async of the "copy" stage:
    mma_m16n8k16_f16_f16_smem_col_row(smemA+sharedOffsetA[computeStageIdx]+(localWarpIdx/4)*16, smemB+sharedOffsetB[computeStageIdx]+(localWarpIdx%4)*8, smemC+smemCIdx, 32, 32);

    // Collectively release the stage resources
    pipeline.consumer_release();
  }
  computeStageIdx = (batch - 1) % stagesCount;
  // Compute the data fetch by the last iteration
  pipeline.consumer_wait();
  mma_m16n8k16_f16_f16_smem_col_row(smemA+sharedOffsetA[computeStageIdx]+(localWarpIdx/4)*16, smemB+sharedOffsetB[computeStageIdx]+(localWarpIdx%4)*8, smemC+smemCIdx, 32, 32);
  pipeline.consumer_release();  

  // Each thread stores one element in a row
  if(laneId < 8) {
    for (int i = 0; i < 16; i++) {
      row = warpRow*m + i;
      col = warpCol*n +laneId;
      C[row*N + col] =  smemC[smemCIdx + i*8 + laneId];
    }
  }
}

#endif
