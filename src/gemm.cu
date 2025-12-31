#include "gemm.cuh"
#include "mma_intrinsics.cuh"

// Disables `cuda::barrier` initialization warning.
#pragma nv_diag_suppress static_var_with_dynamic_init

// assume b is col major for now
void gemm_cpu(int8_t *A, int8_t *B, int *C, int M, int N, int K) {
  int sum;
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++){
      sum = 0;
      for (int i = 0; i < K; i++) {
        sum += A[r*K+i] * B[c*K+i];
      }
      C[r*N+c] = sum;
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

__global__ void gemm_tensorop(int8_t *A, int8_t *B, int *C, int M, int N, int K) {
  using namespace nvcuda;
  // warpidx = idx within the block(threadidx.x / 32) + number of warps before this warp(blockidx.x*blockDim.x/32)
  int warpIdx = threadIdx.x / 32 + blockIdx.x*blockDim.x/32;
  int warpRow = warpIdx / (N/16);
  int warpCol = warpIdx % (N/16);

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
  // Initialize the output to zero
  wmma::fill_fragment(c_frag, 0);

  int offsetA = warpRow*K*16;
  int offsetB = warpCol*16;
  int offsetC = warpRow*N*16 + warpCol*16;

  for (int i = 0; i < K/16; i++) {
    // Load the inputs
    wmma::load_matrix_sync(a_frag, A + offsetA + i*16, K);
    wmma::load_matrix_sync(b_frag, B + offsetB + i*N*16, N);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  // Store the output
  wmma::store_matrix_sync(C+offsetC, c_frag, N, wmma::mem_row_major);
}

// Use inline PTX mma instructions to do matrix multiplication
// s8 * s8 -> s32
__global__ void mma_m16n8k16_s8_s8(int8_t *A, int ldmA, int8_t *B, int ldmB, int32_t *C) { 
  __shared__ int8_t smemA[16*16];
  __shared__ int8_t smemB[16*8];
  __shared__ int32_t smemC[16*8];
  // Each thread loads one element in a row
  if(threadIdx.x < 16) {
    for (int i = 0; i < 16; i++) {
      smemA[threadIdx.x*16 + i] = A[threadIdx.x*ldmA + i];
      if(threadIdx.x < 8)
        smemB[threadIdx.x*16 + i] = B[threadIdx.x*ldmB + i];
    }
  }
  __syncthreads();
  // For A, we do x2 8x8 ldmatrix, so the first 15 threads provide row addresses
  int8_t *a = smemA + (threadIdx.x & 0x1f) * 16;
  uint32_t smem_a = static_cast<uint32_t>(__cvta_generic_to_shared(a));
  uint32_t dstA[2];
  ldmatrix_x2_m8n8_b16(dstA[0], dstA[1], smem_a);

  // For B, we do x1 8x8 ldmatrix
  int8_t *b = smemB + (threadIdx.x & 0x1f) * 16;
  uint32_t smem_b = static_cast<uint32_t>(__cvta_generic_to_shared(b));
  uint32_t dstB;
  ldmatrix_x1_m8n8_b16(dstB, smem_b);

  uint32_t d[4] = {0, 0, 0, 0};
  mma_m16n8k16_row_col_s32_s8_s8_s32(d[0], d[1], d[2], d[3], dstA[0], dstA[1], dstB);

  int groupID = threadIdx.x >> 2;
  int threadID_in_group = threadIdx.x % 4;
  
  smemC[groupID*8 + threadID_in_group*2] = d[0];
  smemC[groupID*8 + threadID_in_group*2 + 1] = d[1];
  smemC[(groupID+8)*8 + threadID_in_group*2] = d[2];
  smemC[(groupID+8)*8 + threadID_in_group*2 + 1] = d[3];

// row =      groupID                           for ci where i <  2
//          groupID + 8                         for ci where i >= 2

// col =  (threadID_in_group * 2) + (i & 0x1)    for ci where i = {0,..,3}

  // This requires sm_90 or above lmao
  // int32_t *c = smemC + (threadIdx.x & 0x1f) * 16;
  // uint32_t smem_c = static_cast<uint32_t>(__cvta_generic_to_shared(c));  
  // asm volatile ("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
  //     :: "r"(smem_c),
  //        "r"(d[0]), "r"(d[1]), "r"(d[2]), "r"(d[3]));

  if(threadIdx.x == 0) {
    for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 8; j++) {
        printf("%d ", smemC[i*8+j]);
      }
      printf("\n");
    }
  }
}