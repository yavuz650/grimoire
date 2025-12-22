#include "vectorAdd.cuh"

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

// ***************************************************************************************************************
// Following the example in CUDA docs...                                                                          |
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#spatial-partitioning-also-known-as-warp-specialization  |
// It appears that the recommended way of implementing warp specialization is by using cuda::barrier structures.  |
// Barriers provide the mechanism for synchronization between the specialized warps.                              |
// The example in the document shows a producer/consumer system where the barriers are used for synchronization.  |
// ***************************************************************************************************************

// C = A+B
__global__ void vectorAdd_ws_persistent(int *A, int *B, int *C) {
  int warpIdx = threadIdx.x / 32;
  // Double buffering for tile storage(2 double-buffers, one for each input so 2*2*K)
  // A-B-A-B memory layout
  __shared__ int buffers[2*2*K];
  __shared__ cuda::barrier<cuda::thread_scope_block> bar[4];
  __shared__ cuda::barrier<cuda::thread_scope_block> bar2[2];
  __shared__ int localTileIdx;
  if (threadIdx.x < 4)
    init(bar + threadIdx.x, blockDim.x*blockDim.y*blockDim.z);
  if (threadIdx.x == 0) {
    init(bar2, blockDim.x*blockDim.y*blockDim.z);
    init(bar2+1, blockDim.x*blockDim.y*blockDim.z);
  }

  __syncthreads();
  switch (warpIdx)
  {
    case 0:
      dma_persistent(buffers, A, B, &bar[0], &bar[2], bar2, localTileIdx);
      break;
    case 1:
      math_persistent(buffers, C, &bar[0], &bar[2], bar2, localTileIdx);
      break;
    // case 3:
    //   math();
    //   break;
    default:
      break;
  }
  __syncthreads();
}

// DMA is the "producer", i.e. it fills the buffers with data from global memory.
__device__ void dma_persistent(int *buffers, int *A, int *B, cuda::barrier<cuda::thread_scope_block> *ready, cuda::barrier<cuda::thread_scope_block> *filled, cuda::barrier<cuda::thread_scope_block> *transition, int &localTileIdx) {
  while(true) {
    transition[0].arrive_and_wait();
    if(threadIdx.x == 0) {
      localTileIdx = atomicAdd(&globalNextTile, 1);
    }
    __syncwarp();
    transition[1].arrive();
    if(localTileIdx >= globalTotalTiles)
      return;

    int tileCounter = 0;
    while(tileCounter != N/K) {
      int bufferIdx = tileCounter%2;
      // Wait until buffer becomes ready
      ready[bufferIdx].arrive_and_wait();
      // Load A
      for (int i = 0; i < K/32; i++) {
        buffers[bufferIdx*2*K + i*32 + threadIdx.x%32] = A[localTileIdx*N + tileCounter*K + i*32 + threadIdx.x%32];
      }
      // Load B
      for (int i = 0; i < K/32; i++) {
        buffers[K + bufferIdx*2*K + i*32 + threadIdx.x%32] = B[localTileIdx*N + tileCounter*K + i*32 + threadIdx.x%32];
      }
      filled[bufferIdx].arrive();
      tileCounter++;
    }
  }
}

// Math is the "consumer", i.e. it consumes the data in buffers to calculate results and stores to global memory.
__device__ void math_persistent(int *buffers, int *C, cuda::barrier<cuda::thread_scope_block> *ready, cuda::barrier<cuda::thread_scope_block> *filled, cuda::barrier<cuda::thread_scope_block> *transition, int &localTileIdx) {
  ready[0].arrive(); // buffer_0 is ready for initial fill
  ready[1].arrive(); // buffer_1 is ready for initial fill
  while(true) {
    transition[0].arrive();
    //printf("Math Got localTileIdx: %d\n", localTileIdx);
    transition[1].arrive_and_wait();
    if(localTileIdx >= globalTotalTiles)
      return;
    int tileCounter = 0;
    while(tileCounter != N/K) {
      int bufferIdx = tileCounter%2;
      // Wait until buffer becomes filled
      filled[bufferIdx].arrive_and_wait();
      // Calculate and store C to global memory without buffering (?) Might not be optimal
      for (int i = 0; i < K/32; i++) {
        C[localTileIdx*N + tileCounter*K + i*32 + threadIdx.x%32] = buffers[bufferIdx*2*K + i*32 + threadIdx.x%32] + buffers[K + bufferIdx*2*K + i*32 + threadIdx.x%32];
      }
      ready[bufferIdx].arrive();
      tileCounter++;
    }
  }
}


// C = A+B
__global__ void vectorAdd_ws(int *A, int *B, int *C) {
  int warpIdx = threadIdx.x / 32;
  // Double buffering for tile storage(2 double-buffers, one for each input so 2*2*K)
  // A-B-A-B memory layout
  __shared__ int buffers[2*2*K];
  __shared__ cuda::barrier<cuda::thread_scope_block> bar[4];
  if (threadIdx.x < 4)
    init(bar + threadIdx.x, blockDim.x*blockDim.y*blockDim.z);

  __syncthreads();
  switch (warpIdx)
  {
    case 0:
      dma(buffers, A, B, &bar[0], &bar[2]);
      break;
    case 1:
      math(buffers, C, &bar[0], &bar[2]);
      break;
    // case 3:
    //   math();
    //   break;
    default:
      break;
  }
  __syncthreads();
}

// DMA is the "producer", i.e. it fills the buffers with data from global memory.
__device__ void dma(int *buffers, int *A, int *B, cuda::barrier<cuda::thread_scope_block> *ready, cuda::barrier<cuda::thread_scope_block> *filled) {
  int tileCounter = 0;
  //int bufferCounter = 0;
  while(tileCounter != N/K) {
    int bufferIdx = tileCounter%2;
    // Wait until buffer becomes ready
    ready[bufferIdx].arrive_and_wait();
    // Load A
    for (int i = 0; i < K/32; i++) {
      buffers[bufferIdx*2*K + i*32 + threadIdx.x%32] = A[blockIdx.x*N + tileCounter*K + i*32 + threadIdx.x%32];
    }
    // Load B
    for (int i = 0; i < K/32; i++) {
      buffers[K + bufferIdx*2*K + i*32 + threadIdx.x%32] = B[blockIdx.x*N + tileCounter*K + i*32 + threadIdx.x%32];
    }
    filled[bufferIdx].arrive();
    //bufferIdx = (bufferIdx + 1)%2;
    //bufferCounter++;
    tileCounter++;
  }
}

// Math is the "consumer", i.e. it consumes the data in buffers to calculate results and stores to global memory.
__device__ void math(int *buffers, int *C, cuda::barrier<cuda::thread_scope_block> *ready, cuda::barrier<cuda::thread_scope_block> *filled) {
  int tileCounter = 0;
  ready[0].arrive(); // buffer_0 is ready for initial fill
  ready[1].arrive(); // buffer_1 is ready for initial fill
  while(tileCounter != N/K) {
    int bufferIdx = tileCounter%2;
    // Wait until buffer becomes filled
    filled[bufferIdx].arrive_and_wait();
    // Calculate and store C to global memory without buffering (?) Might not be optimal
    for (int i = 0; i < K/32; i++) {
      C[blockIdx.x*N + tileCounter*K + i*32 + threadIdx.x%32] = buffers[bufferIdx*2*K + i*32 + threadIdx.x%32] + buffers[K + bufferIdx*2*K + i*32 + threadIdx.x%32];
    }
    ready[bufferIdx].arrive();
    tileCounter++;
  }
}

