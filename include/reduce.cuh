
#ifndef __REDUCE_CUH__
#define __REDUCE_CUH__

#include "common.cuh"
#include "mma_intrinsics.cuh"
#include <cooperative_groups/reduce.h>

namespace cg=cooperative_groups;

// Reduces the input vector A by adding its elements.
// Each thread block adds K elements together, and stores it in results
// Threads should be laid out linearly.
// The output vector has as many elements as there are thread blocks
template <int32_t K>
__global__ void block_reduce(float *A, int64_t count, float *results)
{
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(block);
  __shared__ cuda::atomic<float, cuda::thread_scope_block> total_sum;
  if(block.thread_rank() == 0)
    total_sum.store(0);
  int startIdx = block.group_index().x * K;
  float thread_sum = 0;

  // Stride loop over all values, each thread accumulates its part of the array.
  for (int i = block.thread_rank(); i < K; i += block.size()) {
    thread_sum += A[startIdx+i];
  }

  // reduce thread sums across the tile, add the result to the atomic
  cg::reduce_update_async(tile, total_sum, thread_sum, cg::plus<float>());
  // synchronize the block, to ensure all async reductions are ready
  block.sync();
  // Thread 0 stores the result
  if(block.thread_rank() == 0)
    results[block.group_index().x] = total_sum.load();
}


#endif
