#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <vector>

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
// __device__ void sched(int *workIdxCounter, WorktileMetadata *worktile, TileBufferMetadata *bufferMetadata) {
//   if(threadIdx.x%32 == 0) {
//     int workIdx = atomicAdd(workIdxCounter, 1);
//     while(workIdx < L/N) {
//       // printf("SCHED: workIdx: %d\n", workIdx);
//       worktile->workIdx = workIdx;
//       worktile->dmaFinished = 0;
//       worktile->mathFinished = 0;
//       // Clear buffers
//       bufferMetadata[0].isBufferProcessed = 1;
//       bufferMetadata[1].isBufferProcessed = 1;
//       // printf("SCHED: Starting workIdx: %d\n", workIdx);
//       worktile->start = 1;
//       // Wait until work finishes
//       while(!worktile->mathFinished || !worktile->dmaFinished) {}
//       // printf("SCHED: workIdx: %d finished.\n", workIdx);
//       worktile->start = 0;
//       if(threadIdx.x%32 == 0)
//         workIdx = atomicAdd(workIdxCounter, 1);
//     }
//     // No more work left, exit
//     noWorkLeft = 1;
//   }
//   return;
// }

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
      printf("CTA:%d Got localTileIdx: %d\n", blockIdx.x, localTileIdx);
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

int main(int argc, char* argv[]) {
    // if (argc < 2) {
    //   std::cerr << "Usage: " << argv[0] << " <integer>\n";
    //   return 1;
    // }
    // int L = std::stoi(argv[1]);

    printf("Vector size in bytes: %d\n", L*sizeof(int));
    // Allocate memory for vectors on the host
    std::vector<int> h_vector1(L);
    std::vector<int> h_vector2(L);
    std::vector<int> h_vector3(L);
    std::vector<int> h_vector4(L);
    // Initialize random seed
    srand(static_cast<unsigned>(time(nullptr)));

    // Generate random vectors on the host
    for (int i = 0; i < L; ++i) {
      h_vector1[i] = rand() % 20;
      h_vector2[i] = rand() % 20;
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < L; i++) {
      h_vector4[i] = h_vector1[i] + h_vector2[i];
    }
    const auto t_end = std::chrono::high_resolution_clock::now();

    printf("First 5 elements A\n");
    for (int i = 0; i < 5; i++) {
      printf("%d\n",h_vector1[i]);
    }
    printf("First 5 elements B\n");
    for (int i = 0; i < 5; i++) {
      printf("%d\n",h_vector2[i]);
    }
    // Allocate memory for vectors on the GPU
    int *d_vector1;
    int *d_vector2;
    int *d_vector3;
    cudaError_t err;
    cudaMalloc(&d_vector1, L * sizeof(int));
    err = cudaGetLastError();
    if(cudaSuccess != err) {
      printf("Failed to allocate memory! Error code: %d, %s, %d\n", err, __FILE__, __LINE__);
      abort();
    }
    cudaMalloc(&d_vector2, L * sizeof(int));
    err = cudaGetLastError();
    if(cudaSuccess != err) {
      printf("Failed to allocate memory! Error code: %d, %s, %d\n", err, __FILE__, __LINE__);
      abort();
    }
    cudaMalloc(&d_vector3, L * sizeof(int));
    err = cudaGetLastError();
    if(cudaSuccess != err) {
      printf("Failed to allocate memory! Error code: %d, %s, %d\n", err, __FILE__, __LINE__);
      abort();
    }

    // Copy vectors from host to GPU
    cudaMemcpy(d_vector1, h_vector1.data(), L * sizeof(int), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if(cudaSuccess != err) {
      printf("Failed to move data to gpu memory! Error code: %d, %s, %d\n", err, __FILE__, __LINE__);
      abort();
    }
    cudaMemcpy(d_vector2, h_vector2.data(), L * sizeof(int), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if(cudaSuccess != err) {
      printf("Failed to move data to gpu memory! Error code: %d, %s, %d\n", err, __FILE__, __LINE__);
      abort();
    }

    dim3 cta;
    dim3 grid;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    printf("Launching kernel...\n");
    // Conventional cuda kernel
    cta = dim3(64,1,1);
    grid = dim3(L/N,1,1);
    cudaEventRecord(start);
    vectorAdd<<<grid,cta>>>(d_vector1,d_vector2,d_vector3);
    cudaEventRecord(stop);
    err = cudaGetLastError();
    if(cudaSuccess != err) {
      printf("Failed to launch kernel! Error code: %d, %s, %d\n", err, __FILE__, __LINE__);
      abort();
    }
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Finished traditional kernel...\n");
    
    printf("Launching WS kernel...\n");
    cta=dim3(64,1,1);
    grid=dim3(30,1,1);
    cudaEventRecord(start);
    vectorAdd_ws_persistent<<<grid,cta>>>(d_vector1,d_vector2,d_vector3);
    cudaEventRecord(stop);
    err = cudaGetLastError();
    if(cudaSuccess != err) {
      printf("Failed to launch kernel! Error code: %d, %s, %d\n", err, __FILE__, __LINE__);
      abort();
    }
    cudaEventSynchronize(stop);
    float millisecondsWS = 0;
    cudaEventElapsedTime(&millisecondsWS, start, stop);        
    printf("Finished WS kernel...\n");

    cudaMemcpy(h_vector3.data(), d_vector3, L * sizeof(int), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if(cudaSuccess != err) {
      printf("Failed to move data to host memory! Error code: %d, %s, %d\n", err, __FILE__, __LINE__);
      abort();
    }

    printf("First 5 elements(host)\n");
    for (int i = 0; i < 5; i++) {
      printf("%d\n",h_vector4[i]);
    }
    printf("First 5 elements(device)\n");
    for (int i = 0; i < 5; i++) {
      printf("%d\n",h_vector3[i]);
    }

    if(h_vector3 != h_vector4)
      printf("Mismatch.\n");
    else {
      std::cout << "CPU time(ms): " << std::fixed << std::setprecision(2) << std::chrono::duration<double, std::milli>(t_end - t_start).count() << '\n';
      printf("Outputs match.\n");
      printf("WS Kernel execution time(ms): %f\n",millisecondsWS);
      printf("Conventional Kernel execution time(ms): %f\n",milliseconds);
    }

    // Free device memory
    cudaFree(d_vector1);
    cudaFree(d_vector2);
    cudaFree(d_vector3);

    return 0;
}
