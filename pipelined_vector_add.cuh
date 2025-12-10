#include <cuda.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda/barrier>
// Disables `cuda::barrier` initialization warning.
#pragma nv_diag_suppress static_var_with_dynamic_init

// All sizes and lengths are in terms of elements, not bytes
// Vector length
constexpr int L = 65536 * 512;
// Tile size
constexpr int N = 32768;
// Batch size
//constexpr int batchSize = 4096;
__device__ int globalNextTile = 0;
__device__ int globalTotalTiles = L/N;
constexpr size_t stagesCount = 2; // Pipeline with two stages

// Example from Cuda docs
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies-using-cuda-pipeline

__device__ void vectorAdd_compute(int *A, int *B, int *C) {
  auto rank = cooperative_groups::this_thread_block().thread_rank();
  C[rank] = A[rank] + B[rank];
}

__global__ void vectorAdd_pipelined(int *A, int *B, int *C) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  //assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz * grid_size

  // Two batches must fit in shared memory:
  extern __shared__ int shared[]; // *2 for A and B, stagesCount*block.num_threads()*2
  size_t sharedOffsetA[stagesCount] = { 0, 2*block.num_threads() }; // Offsets to each batch
  size_t sharedOffsetB[stagesCount] = { block.num_threads(), 3*block.num_threads() }; // Offsets to each batch

  // Allocate shared storage for a two-stage cuda::pipeline:
  __shared__ cuda::pipeline_shared_state<
      cuda::thread_scope::thread_scope_block,
      stagesCount
  > sharedState;
  auto pipeline = cuda::make_pipeline(block, &sharedState);

  // Each thread processes `batch_sz` elements.
  // Compute offset of the batch `batch` of this thread block in global memory:
  auto block_batch = [&](size_t batch) -> int {
    return block.group_index().x * block.num_threads() + grid.num_threads() * batch;
  };

  // Initialize first pipeline stage by submitting a `memcpy_async` to fetch a whole batch for A and B:
  // if (batch_sz == 0) return;
  pipeline.producer_acquire();
  cuda::memcpy_async(block, shared + sharedOffsetA[0], A + block_batch(0), sizeof(int) * block.num_threads(), pipeline);
  cuda::memcpy_async(block, shared + sharedOffsetB[0], B + block_batch(0), sizeof(int) * block.num_threads(), pipeline);
  pipeline.producer_commit();

  // Pipelined copy/compute:
  for (size_t batch = 1; batch < L/grid.num_threads(); ++batch) {
      // Stage indices for the compute and copy stages:
      size_t computeStageIdx = (batch - 1) % 2;
      size_t copyStageIdx = batch % 2;

      size_t globalIdx = block_batch(batch);

      // Collectively acquire the pipeline head stage from all producer threads:
      pipeline.producer_acquire();

      // Submit async copies to the pipeline's head stage to be
      // computed in the next loop iteration

      cuda::memcpy_async(block, shared + sharedOffsetA[copyStageIdx], A + globalIdx, sizeof(int) * block.num_threads(), pipeline);
      cuda::memcpy_async(block, shared + sharedOffsetB[copyStageIdx], B + globalIdx, sizeof(int) * block.num_threads(), pipeline);
      // Collectively commit (advance) the pipeline's head stage
      pipeline.producer_commit();

      // Collectively wait for the operations committed to the
      // previous `compute` stage to complete:
      pipeline.consumer_wait();

      // Computation overlapped with the memcpy_async of the "copy" stage:
      vectorAdd_compute(shared+sharedOffsetA[computeStageIdx], shared+sharedOffsetB[computeStageIdx], C + block_batch(batch-1));

      // Collectively release the stage resources
      pipeline.consumer_release();
  }

  // Compute the data fetch by the last iteration
  pipeline.consumer_wait();
  vectorAdd_compute(shared+sharedOffsetA[(L/grid.num_threads()-1) %2], shared+sharedOffsetB[(L/grid.num_threads()-1) %2], C + block_batch(L/grid.num_threads()-1));
  pipeline.consumer_release();  
}


__global__ void vectorAdd(int *A, int *B, int *C) {
  int base = blockIdx.x*blockDim.x*4;
  int idx = base + (threadIdx.x/32) * 128 + threadIdx.x%32;
  C[idx] = A[idx]+B[idx];
  idx += 32;
  C[idx] = A[idx]+B[idx];
  idx += 32;
  C[idx] = A[idx]+B[idx];
  idx += 32;
  C[idx] = A[idx]+B[idx];
}

