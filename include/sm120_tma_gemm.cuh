#ifndef __SM120_TMA_GEMM_CUH__
#define __SM120_TMA_GEMM_CUH__

#include "common.cuh"
#include "mma_intrinsics.cuh"

__global__ void mma_f16_f16_tma(const __grid_constant__ CUtensorMap tensorMapA, 
                                const __grid_constant__ CUtensorMap tensorMapB,
                                const __grid_constant__ CUtensorMap tensorMapC, uint64_t K) {
  // The destination shared memory buffer of a bulk tensor operation should be 128 byte aligned.
  // Assuming 4 warps in each CTA
  extern __shared__ __align__(128) int8_t smem[];
  __half* smemA = (__half*)(smem);
  __half* smemB = (__half*)(smem + 64*64 * sizeof(__half));
  float*  smemC = (float*) (smem + 64*64 * sizeof(__half) * 2);

  auto block = cooperative_groups::this_thread_block();
  int blockRow = block.group_index().y;
  int blockCol = block.group_index().x;
  for (int i = threadIdx.x; i < 64 * 64; i += block.num_threads()) {
    smemC[i] = 0.0f;
  }

  // Initialize shared memory barrier with the number of threads participating in the barrier.
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;

  if (threadIdx.x == 0) {
    // Initialize barrier. All `blockDim.x` threads in block participate.
    init(&bar, blockDim.x);
  }
  // Syncthreads so initialized barrier is visible to all threads.
  block.sync();

  cuda::barrier<cuda::thread_scope_block>::arrival_token token;
  for (int i = 0; i < K/64; i++) {
    if (threadIdx.x == 0) {
      // Initiate bulk tensor copy for A.
      cuda::ptx::cp_async_bulk_tensor(
        cuda::ptx::space_shared, cuda::ptx::space_global,
        smemA, &tensorMapA, { i*64, blockRow*64 },
        cuda::device::barrier_native_handle(bar));

      // Initiate bulk tensor copy for B.
      cuda::ptx::cp_async_bulk_tensor(
        cuda::ptx::space_shared, cuda::ptx::space_global,
        smemB, &tensorMapB, { i*64, blockCol*64 },
        cuda::device::barrier_native_handle(bar));
      // Arrive on the barrier and tell how many bytes are expected to come in.
      token = cuda::device::barrier_arrive_tx(bar, 1, 2*64*64*sizeof(__half));
    } else {
      // Other threads just arrive.
      token = bar.arrive();
    }
    // Wait for the data to have arrived.
    bar.wait(std::move(token));

    // Do MMA here...
    mma_m16n8k16_f16_f16_smem_row_col_64x64(smemA,
                                            smemB,
                                            smemC);
  }

  // Wait for shared memory writes to be visible to TMA engine.
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  block.sync();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
    // int32_t tensor_coords[2] = { blockCol*64, blockRow*64 };
    cuda::ptx::cp_async_bulk_tensor(
      cuda::ptx::space_global, cuda::ptx::space_shared,
      &tensorMapC, { blockCol*64, blockRow*64 }, smemC);
    // Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    cuda::ptx::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>());
  }

  // Destroy barrier. This invalidates the memory region of the barrier. If
  // further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  // if (threadIdx.x == 0) {
  //   (&bar)->~barrier();
  // }

}


#endif
