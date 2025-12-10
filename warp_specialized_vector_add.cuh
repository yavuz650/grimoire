
#include <cuda.h>
#include <cuda/barrier>
// Disables `cuda::barrier` initialization warning.
#pragma nv_diag_suppress static_var_with_dynamic_init

// All sizes and lengths are in terms of elements, not bytes
// Vector length
constexpr int L = 65536 * 512;
// Tile size
constexpr int N = 32768;
// Buffer size
constexpr int K = 4096;
__device__ int globalNextTile = 0;
__device__ int globalTotalTiles = L/N;

// ***************************************************************************************************************
// Following the example in CUDA docs...                                                                          |
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#spatial-partitioning-also-known-as-warp-specialization  |
// It appears that the recommended way of implementing warp specialization is by using cuda::barrier structures.  |
// Barriers provide the mechanism for synchronization between the specialized warps.                              |
// The example in the document shows a producer/consumer system where the barriers are used for synchronization.  |
// ***************************************************************************************************************
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

__global__ void vectorAdd(int *A, int *B, int *C) {
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  C[idx] = A[idx]+B[idx];
}
