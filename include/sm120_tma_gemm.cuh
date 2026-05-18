#ifndef __SM120_TMA_GEMM_CUH__
#define __SM120_TMA_GEMM_CUH__

#include "common.cuh"
#include "mma_intrinsics.cuh"

template<size_t StagesCount=2>
__global__ void mma_f16_f16_tma(const __grid_constant__ CUtensorMap tensorMapA, 
                                const __grid_constant__ CUtensorMap tensorMapB,
                                const __grid_constant__ CUtensorMap tensorMapC,
                                uint64_t K) {
  // The destination shared memory buffer of a bulk tensor operation should be 128 byte aligned.
  // Assuming 4 warps in each CTA
  extern __shared__ __align__(1024) int8_t smem[];
  __half* smemA = (__half*)(smem);
  __half* smemB = (__half*)(smem + 64*64*StagesCount * sizeof(__half));
  float*  smemC = (float*) (smem + 64*64*StagesCount * sizeof(__half) * 2);

  auto block = cooperative_groups::this_thread_block();
  int blockRow = block.group_index().y;
  int blockCol = block.group_index().x;
  for (int i = threadIdx.x; i < 64 * 64; i += block.num_threads()) {
    smemC[i] = 0.0f;
  }

  // Initialize shared memory barrier with the number of threads participating in the barrier.
  __shared__ cuda::barrier<cuda::thread_scope_block> bar[StagesCount];
  cuda::barrier<cuda::thread_scope_block>::arrival_token token[StagesCount];

  if (threadIdx.x < StagesCount) {
    // Initialize barrier. All `blockDim.x` threads in block participate.
    init(&bar[threadIdx.x], blockDim.x);
  }
  // Syncthreads so initialized barrier is visible to all threads.
  block.sync();

  // 1. Prologue: Fill the pipeline by fetching the initial stages
  #pragma unroll
  for (int stage = 0; stage < StagesCount; ++stage) {
    if (threadIdx.x == 0) {
      // Initiate copies for A and B...
      cuda::ptx::cp_async_bulk_tensor(
        cuda::ptx::space_shared, cuda::ptx::space_global,
        smemA+stage*64*64, &tensorMapA, { stage*64, blockRow*64 },
        cuda::device::barrier_native_handle(bar[stage]));

      cuda::ptx::cp_async_bulk_tensor(
        cuda::ptx::space_shared, cuda::ptx::space_global,
        smemB+stage*64*64, &tensorMapB, { stage*64, blockCol*64 },
        cuda::device::barrier_native_handle(bar[stage]));
      // Arrive on the barrier and tell how many bytes are expected to come in.
      token[stage] = cuda::device::barrier_arrive_tx(bar[stage], 1, 2*64*64*sizeof(__half));
    } else {
      token[stage] = bar[stage].arrive();
    }
  }

  // Pipelined copy/compute
  // 2. Main Steady-State Loop: Step by StagesCount to make indices static
  // (Assuming K/64 is a multiple of StagesCount. If not, handle the remainder in an epilogue)
  for (int computeBatch = 0; computeBatch < K/64 - StagesCount; computeBatch += StagesCount) {
    #pragma unroll
    for (int stage = 0; stage < StagesCount; stage++) {
      // Wait for current stage data
      bar[stage].wait(std::move(token[stage]));
      mma_m16n8k16_f16_f16_smem_row_col_64x64_swizzle(
          smemA+stage*64*64,
          smemB+stage*64*64,
          smemC);
      // Fetch next batch for this specific stage index
      int fetchBatch = computeBatch + StagesCount + stage;
      if (threadIdx.x == 0) {
        cuda::ptx::cp_async_bulk_tensor(
          cuda::ptx::space_shared, cuda::ptx::space_global,
          smemA+(fetchBatch % StagesCount)*64*64, &tensorMapA, { fetchBatch*64, blockRow*64 },
          cuda::device::barrier_native_handle(bar[stage]));

        cuda::ptx::cp_async_bulk_tensor(
          cuda::ptx::space_shared, cuda::ptx::space_global,
          smemB+(fetchBatch % StagesCount)*64*64, &tensorMapB, { fetchBatch*64, blockCol*64 },
          cuda::device::barrier_native_handle(bar[stage]));
        token[stage] = cuda::device::barrier_arrive_tx(bar[stage], 1, 2*64*64*sizeof(__half));
      } else {
        token[stage] = bar[stage].arrive();
      }
    }
  }

  // 3. Epilogue: Drain the pipeline (Compute remaining fetched batches without new fetches)
  #pragma unroll
  for (int stage = 0; stage < StagesCount; stage++) {
    bar[stage].wait(std::move(token[stage]));
    mma_m16n8k16_f16_f16_smem_row_col_64x64_swizzle(
      smemA + stage*64*64, 
      smemB + stage*64*64, 
      smemC);
  }

  // Wait for shared memory writes to be visible to TMA engine.
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  block.sync();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
    // Store left half of C
    cuda::ptx::cp_async_bulk_tensor(
       cuda::ptx::space_global, cuda::ptx::space_shared,
       &tensorMapC, { blockCol*64, blockRow*64 }, smemC);
    // Store right half
    cuda::ptx::cp_async_bulk_tensor(
       cuda::ptx::space_global, cuda::ptx::space_shared,
       &tensorMapC, { blockCol*64+32, blockRow*64 }, smemC + 64*32);
    // Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    cuda::ptx::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>());
  }
}

// ***************************************************************************************************************
// Following the example in CUDA docs...                                                                          |
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#spatial-partitioning-also-known-as-warp-specialization  |
// It appears that the recommended way of implementing warp specialization is by using cuda::barrier structures.  |
// Barriers provide the mechanism for synchronization between the specialized warps.                              |
// The example in the document shows a producer/consumer system where the barriers are used for synchronization.  |
// ***************************************************************************************************************
template<size_t StagesCount=2, uint64_t K=64>
__device__ void dma_ws(cuda::barrier<cuda::thread_scope_block> *ready, 
                       cuda::barrier<cuda::thread_scope_block> *filled, 
                       __half *smemA, __half *smemB,
                       const CUtensorMap *tensorMapA,
                       const CUtensorMap *tensorMapB) {
  auto block = cooperative_groups::this_thread_block();
  int blockRow = block.group_index().y;
  int blockCol = block.group_index().x;
  for(int i=0; i<K/64; i++) {
    // wait for buffer_(i%StagesCount) to be ready to be filled
    int32_t stage = i % StagesCount;
    ready[stage].arrive_and_wait();
    // produce, i.e., fill in, buffer_(i%StagesCount)
    if (threadIdx.x == 128) {
      // Initiate bulk tensor copy for A.
      cuda::ptx::cp_async_bulk_tensor(
        cuda::ptx::space_shared, cuda::ptx::space_global,
        smemA+stage*64*64, tensorMapA, { i*64, blockRow*64 },
        cuda::device::barrier_native_handle(filled[stage]));

      // Initiate bulk tensor copy for B.
      cuda::ptx::cp_async_bulk_tensor(
        cuda::ptx::space_shared, cuda::ptx::space_global,
        smemB+stage*64*64, tensorMapB, { i*64, blockCol*64 },
        cuda::device::barrier_native_handle(filled[stage]));
      // Arrive on the barrier and tell how many bytes are expected to come in.
      auto token = cuda::device::barrier_arrive_tx(filled[stage], 1, 2*64*64*sizeof(__half));
    } else { 
      // Other threads just arrive.
      auto token = filled[stage].arrive();
    }
  }
}

template<size_t StagesCount=2, uint64_t K=64>
__device__ void math_ws(cuda::barrier<cuda::thread_scope_block> *ready, 
                       cuda::barrier<cuda::thread_scope_block> *filled, 
                       __half *smemA, __half *smemB, float *smemC) {
  // Buffers are ready for initial fill
  ready[0].arrive();
  ready[1].arrive();
  int32_t parity[StagesCount];
  parity[0] = 0;
  parity[1] = 0;
#pragma unroll
  for(int i=0; i<K/64; i++) {
    // wait for buffer_(i%StagesCount) to be filled by the producer
    int32_t stage = i % StagesCount;
    // Wait for all threads participating in the barrier to complete mbarrier_arrive().
    // Get a handle to the native barrier to use with cuda::ptx API.
    //filled[stage].arrive();
    while (!cuda::ptx::mbarrier_try_wait_parity(cuda::device::barrier_native_handle(filled[stage]), parity[stage])) {}
    // Flip parity.
    parity[stage] ^= 1;
    mma_m16n8k16_f16_f16_smem_row_col_64x64_swizzle(smemA+stage*64*64,
                                                    smemB+stage*64*64,
                                                    smemC);
    ready[stage].arrive();
  }
}
template<size_t StagesCount=2, uint64_t K=64>
__global__ void mma_f16_f16_tma_ws(const __grid_constant__ CUtensorMap tensorMapA, 
                                   const __grid_constant__ CUtensorMap tensorMapB,
                                   const __grid_constant__ CUtensorMap tensorMapC) {
  // The destination shared memory buffer of a bulk tensor operation should be 128 byte aligned.
  // Additionally, swizzling requires 1024 byte alignment
  // Assuming 5 warps in each CTA: 1 DMA and 4 math 
  extern __shared__ __align__(1024) int8_t smem[];
  __half* smemA = (__half*)(smem);
  __half* smemB = (__half*)(smem + 64*64*StagesCount * sizeof(__half));
  float*  smemC = (float*) (smem + 64*64*StagesCount * sizeof(__half) * 2);
  auto block = cooperative_groups::this_thread_block();
  int blockRow = block.group_index().y;
  int blockCol = block.group_index().x;

  for (int i = threadIdx.x; i < 64 * 64; i += block.num_threads()) {
    smemC[i] = 0.0f;
  }
  // Need 4 barriers for 2 buffers: 2 for ready, 2 for filled
  // bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,
  // while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively
  __shared__ cuda::barrier<cuda::thread_scope_block> bar[4];
  if (block.thread_rank() == 0) {
    init(bar+3, 32);
    init(bar+2, 32);
    init(bar+1, block.size());
    init(bar, block.size());
  }
  block.sync();

  // Fifth warp is DMA
  if (block.thread_rank() > 127)
    dma_ws<StagesCount,K>(bar, bar+2, smemA, smemB, &tensorMapA, &tensorMapB);
  else
    math_ws<StagesCount,K>(bar, bar+2, smemA, smemB, smemC);

  // Wait for shared memory writes to be visible to TMA engine.
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  block.sync();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
    // Store left half of C
    cuda::ptx::cp_async_bulk_tensor(
       cuda::ptx::space_global, cuda::ptx::space_shared,
       &tensorMapC, { blockCol*64, blockRow*64 }, smemC);
    // Store right half
    cuda::ptx::cp_async_bulk_tensor(
       cuda::ptx::space_global, cuda::ptx::space_shared,
       &tensorMapC, { blockCol*64+32, blockRow*64 }, smemC + 64*32);
    // Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    cuda::ptx::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>());
  }
}

#endif

