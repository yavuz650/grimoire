#ifndef __SM86_GEMM_CUH__
#define __SM86_GEMM_CUH__

#include "common.cuh"
#include "mma_intrinsics.cuh"

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
template <typename T, Layout LayoutA, Layout LayoutB>
__global__ void simt_gemm(T *A, T *B, T *C, int M, int N, int K, T scale) {
  int r = blockDim.y*blockIdx.y+threadIdx.y;
  int c = blockDim.x*blockIdx.x+threadIdx.x;

  T sum = 0;
  for (int i = 0; i < K; i++) {
    sum += A[getIdx<LayoutA>(r, i, (LayoutA == Layout::RowMajor ? K : M))] *
           B[getIdx<LayoutB>(i, c, (LayoutB == Layout::RowMajor ? N : K))];
  }
  C[N*r+c] = sum*scale;
}

// A is row major, B is col major
__global__ void wmma_gemm_row_col(int8_t *A, int8_t *B, int *C, int M, int N, int K) {
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
    mma_m16n8k16_f16_f16_smem_row_col(smemA+smemAIdx, smemB+smemBIdx, smemC+smemCIdx, 16, 16, 8);
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

// A is row, B is col, each CTA calculates 64x64 output tile
// Grid shape should be dim3(N/64, M/64, 1) i.e. 2D
template<size_t StagesCount>
__global__ void mma_m16n8k16_f16_f16_multistage_64x64_TN(__half *A, __half *B, float *C, int M, int N, int K) {
  using namespace nvcuda;
  auto block = cooperative_groups::this_thread_block();

  // Two batches must fit in shared memory:
  size_t sharedOffset[StagesCount];
  for (int s = 0; s < StagesCount; ++s) 
    sharedOffset[s] = s * 64*64;

  // Allocate shared storage for an N-stage cuda::pipeline:
  __shared__ cuda::pipeline_shared_state<
      cuda::thread_scope::thread_scope_block,
      StagesCount
  > sharedState;
  auto pipeline = cuda::make_pipeline(block, &sharedState);

  int blockRow = block.group_index().y;
  int blockCol = block.group_index().x;
  // Warp index within the CTA
  int localWarpIdx = threadIdx.x / 32;

  // Assuming 4 warps in each CTA
  extern __shared__ __align__(128) int8_t smem[];

  __half* smemA = (__half*)(smem);
  __half* smemB = (__half*)(smem + 64*64*StagesCount * sizeof(__half));
  float*  smemC = (float*) (smem + 64*64*StagesCount * sizeof(__half) * 2);

  for (int i = threadIdx.x; i < 64 * 64; i += block.num_threads()) {
    smemC[i] = 0.0f;
  }
  block.sync();

  // Global mem row, col
  int row, col;
  // Shared mem row, col
  int r, c;
  // Pipelined copy/compute:
  for (int computeBatch = 0, fetchBatch = 0; computeBatch < K/64; computeBatch++) {
    for (; fetchBatch < K/64 && fetchBatch < (computeBatch + StagesCount); fetchBatch++) {
      pipeline.producer_acquire();
      // Each thread loads 32 elements, equals to 64 bytes. So two threads together load one row/col of A/B into shared memory.
      r = threadIdx.x/2;
      c = (threadIdx.x & 1) * 32;
      row = blockRow*64 + r;
      col = fetchBatch*64 + c; // Odd threads load the second half of the row
      cuda::memcpy_async(smemA+sharedOffset[fetchBatch%StagesCount] + r*64+c, A+row*K+col, cuda::aligned_size_t<16>(32*sizeof(__half)), pipeline);
      r = (threadIdx.x & 1) * 32; // Odd threads load the second half of the column
      c = threadIdx.x/2; 
      row = fetchBatch*64 + r;
      col = blockCol*64 + c;
      cuda::memcpy_async(smemB+sharedOffset[fetchBatch%StagesCount] + r+c*64, B+row+col*K, cuda::aligned_size_t<16>(32*sizeof(__half)), pipeline);
      pipeline.producer_commit();
    }

    pipeline.consumer_wait();    
    // Each warp does 32 MMAs. 4 for each sub-tile, and 8 sub-tiles per warp
    for (int i = 0; i < 8; i++) {
      int subtileRow = (localWarpIdx/2)*2 + i/4;
      int subtileCol = (localWarpIdx%2)*4 + i%4;
      row = subtileRow*16;
      col = subtileCol*8;
      // Computation overlapped with the memcpy_async of the "copy" stage:
      mma_m16n8k16_f16_f16_smem_row_col(smemA+sharedOffset[computeBatch%StagesCount] + row*64,
                                        smemB+sharedOffset[computeBatch%StagesCount] + col*64,
                                        smemC+row*64+col, 64, 64, 64);

      mma_m16n8k16_f16_f16_smem_row_col(smemA+sharedOffset[computeBatch%StagesCount] + row*64+16,
                                        smemB+sharedOffset[computeBatch%StagesCount] + col*64+16,
                                        smemC+row*64+col, 64, 64, 64);

      mma_m16n8k16_f16_f16_smem_row_col(smemA+sharedOffset[computeBatch%StagesCount] + row*64+32,
                                        smemB+sharedOffset[computeBatch%StagesCount] + col*64+32,
                                        smemC+row*64+col, 64, 64, 64);

      mma_m16n8k16_f16_f16_smem_row_col(smemA+sharedOffset[computeBatch%StagesCount] + row*64+48,
                                        smemB+sharedOffset[computeBatch%StagesCount] + col*64+48,
                                        smemC+row*64+col, 64, 64, 64);
    }
    // Collectively release the stage resources
    pipeline.consumer_release();
  }

  block.sync();
  for (int i = threadIdx.x; i < 64 * 64; i += block.num_threads()) {
    r = i / 64;
    c = i % 64;
    row = blockRow * 64 + r;
    col = blockCol * 64 + c;
    C[row * N + col] = smemC[r * 64 + c];
  }
}

// A is row, B is col, each CTA calculates 64x64 output tile
// Grid shape should be dim3(N/64, M/64, 1) i.e. 2D
template<size_t StagesCount>
__global__ void mma_m16n8k16_f16_f16_multistage_64x64_TN_ldoptimized(__half *A, __half *B, float *C, int M, int N, int K) {
  using namespace nvcuda;
  auto block = cooperative_groups::this_thread_block();

  // Two batches must fit in shared memory:
  size_t sharedOffset[StagesCount];
  for (int s = 0; s < StagesCount; ++s) 
    sharedOffset[s] = s * 64*64;

  // Allocate shared storage for an N-stage cuda::pipeline:
  __shared__ cuda::pipeline_shared_state<
      cuda::thread_scope::thread_scope_block,
      StagesCount
  > sharedState;
  auto pipeline = cuda::make_pipeline(block, &sharedState);

  int blockRow = block.group_index().y;
  int blockCol = block.group_index().x;
  // Warp index within the CTA
  int localWarpIdx = threadIdx.x / 32;

  // Assuming 4 warps in each CTA
  extern __shared__ __align__(128) int8_t smem[];

  __half* smemA = (__half*)(smem);
  __half* smemB = (__half*)(smem + 64*64*StagesCount * sizeof(__half));
  float*  smemC = (float*) (smem + 64*64*StagesCount * sizeof(__half) * 2);

  for (int i = threadIdx.x; i < 64 * 64; i += block.num_threads()) {
    smemC[i] = 0.0f;
  }
  block.sync();

  // Global mem row, col
  int row, col;
  // Shared mem row, col
  int r, c;
  // Pipelined copy/compute:
  for (int computeBatch = 0, fetchBatch = 0; computeBatch < K/64; computeBatch++) {
    for (; fetchBatch < K/64 && fetchBatch < (computeBatch + StagesCount); fetchBatch++) {
      pipeline.producer_acquire();
      // Each thread loads 32 elements, equals to 64 bytes. So two threads together load one row/col of A/B into shared memory.
      r = threadIdx.x/2;
      c = (threadIdx.x & 1) * 32;
      row = blockRow*64 + r;
      col = fetchBatch*64 + c; // Odd threads load the second half of the row
      cuda::memcpy_async(smemA+sharedOffset[fetchBatch%StagesCount] + r*64+c, A+row*K+col, cuda::aligned_size_t<16>(32*sizeof(__half)), pipeline);
      r = (threadIdx.x & 1) * 32; // Odd threads load the second half of the column
      c = threadIdx.x/2; 
      row = fetchBatch*64 + r;
      col = blockCol*64 + c;
      cuda::memcpy_async(smemB+sharedOffset[fetchBatch%StagesCount] + r+c*64, B+row+col*K, cuda::aligned_size_t<16>(32*sizeof(__half)), pipeline);
      pipeline.producer_commit();
    }

    pipeline.consumer_wait();
    // Computation overlapped with the memcpy_async of the "copy" stage:
    mma_m16n8k16_f16_f16_smem_row_col_64x64(smemA+sharedOffset[computeBatch%StagesCount],
                                            smemB+sharedOffset[computeBatch%StagesCount],
                                            smemC);
    // Collectively release the stage resources
    pipeline.consumer_release();
  }

  block.sync();
  for (int i = threadIdx.x; i < 64 * 64; i += block.num_threads()) {
    r = i / 64;
    c = i % 64;
    row = blockRow * 64 + r;
    col = blockCol * 64 + c;
    C[row * N + col] = smemC[r * 64 + c];
  }
}


#endif
