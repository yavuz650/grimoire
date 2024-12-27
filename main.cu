#include <cuda.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <vector>

// Vector length
constexpr int L = 65536*2;
// Worktile size
constexpr int N = 1024;
// Tile size
constexpr int K = 128;

struct WorktileMetadata {
  volatile int workIdx;
  // Signals DMA to start working on the workIdx
  volatile int start;
  // Initalize to 1
  volatile int dmaFinished;
  // Initialize to 1
  volatile int mathFinished;
};

struct TileBufferMetadata
{
  int isBufferProcessed;
};
__device__ int workIdxCounter;
__shared__ int noWorkLeft;
__device__ void sched(int *workIdxCounter, WorktileMetadata *worktile, TileBufferMetadata *bufferMetadata) {
  if(threadIdx.x%32 == 0) {
    int workIdx = atomicAdd(workIdxCounter, 1);
    while(workIdx < L/N) {
      // printf("SCHED: workIdx: %d\n", workIdx);
      worktile->workIdx = workIdx;
      worktile->dmaFinished = 0;
      worktile->mathFinished = 0;
      // Clear buffers
      bufferMetadata[0].isBufferProcessed = 1;
      bufferMetadata[1].isBufferProcessed = 1;
      // printf("SCHED: Starting workIdx: %d\n", workIdx);
      worktile->start = 1;
      // Wait until work finishes
      while(!worktile->mathFinished || !worktile->dmaFinished) {}
      // printf("SCHED: workIdx: %d finished.\n", workIdx);
      worktile->start = 0;
      if(threadIdx.x%32 == 0)
        workIdx = atomicAdd(workIdxCounter, 1);
    }
    // No more work left, exit
    noWorkLeft = 1;
  }
  return;
}

__device__ void dma(WorktileMetadata *worktile, TileBufferMetadata *bufferMetadata, int *buffers, int *A, int *B) {
  while(!noWorkLeft) {
    int tileCounter = 0;
    int bufferIdx = 0;
    while(!worktile->start) {
      if(noWorkLeft) {
        // printf("DMA: Exiting\n");
        return;
      }
    }

    // if(threadIdx.x%32 == 0)
    //   printf("DMA: workIdx: %d\n", worktile->workIdx);
    while (tileCounter != N/K) {
      // if(threadIdx.x%32 == 0)
      //   printf("DMA: Waiting for buffer to become ready %d, %d, %d\n", bufferMetadata[bufferIdx].isBufferProcessed, worktile->dmaFinished, worktile->start);
      // Wait until buffer becomes ready
      while(!bufferMetadata[bufferIdx].isBufferProcessed || worktile->dmaFinished || !worktile->start) {
        __syncwarp();
        if(noWorkLeft) {
          return;
        }
      }
      // Load A
      for (int i = 0; i < K/32; i++) {
        buffers[bufferIdx*2*K + i*32 + threadIdx.x%32] = A[worktile->workIdx*N + tileCounter*K + i*32 + threadIdx.x%32];
      }
      // Load B
      for (int i = 0; i < K/32; i++) {
        buffers[K + bufferIdx*2*K + i*32 + threadIdx.x%32] = B[worktile->workIdx*N + tileCounter*K + i*32 + threadIdx.x%32];
      }
      bufferMetadata[bufferIdx].isBufferProcessed = 0;
      bufferIdx = (bufferIdx + 1)%2;
      tileCounter++;
    }
    worktile->dmaFinished = 1;
  }
}

__device__ void math(WorktileMetadata *worktile, TileBufferMetadata *bufferMetadata, int *buffers, int *C) {
  while(!noWorkLeft) {
    int tileCounter = 0;
    int bufferIdx = 0;
    while(!worktile->start) {        
      if(noWorkLeft) {
        return;
      }
    }
    while (tileCounter != N/K) {
      // if(threadIdx.x%32 == 0)
      //   printf("MATH: Waiting for a filled buffer %d. mathFinished: %d\n",bufferMetadata[bufferIdx].isBufferProcessed, worktile->mathFinished);
      // Wait until there's a filled buffer
      while(bufferMetadata[bufferIdx].isBufferProcessed || worktile->mathFinished) {
        __syncwarp();
        if(noWorkLeft) {
          return;
        }
      }
      // C=A+B
      for (int i = 0; i < K/32; i++) {
        buffers[bufferIdx*K + 4*K + i*32 + threadIdx.x%32] = buffers[bufferIdx*2*K + i*32 + threadIdx.x%32] + buffers[K + bufferIdx*2*K + i*32 + threadIdx.x%32];
      }
      // Store C to global memory (?) Might not be optimal
      for (int i = 0; i < K/32; i++) {
        C[worktile->workIdx*N + tileCounter*K + i*32 + threadIdx.x%32] = buffers[bufferIdx*K + 4*K + i*32 + threadIdx.x%32];
      }
      bufferMetadata[bufferIdx].isBufferProcessed = 1;
      bufferIdx = (bufferIdx + 1)%2;
      tileCounter++;
    }
    worktile->mathFinished = 1;
  }
}

// C = A+B
__global__ void vectorAdd(int *A, int *B, int *C) {
  int warpIdx = threadIdx.x / 32;
  // Double buffering for tile storage
  __shared__ WorktileMetadata workIdxMetadata;
  //__shared__ int noWorkLeft;
  __shared__ int buffers[2*3*K];
  __shared__ TileBufferMetadata bufferMetadata[2];
  bufferMetadata[0].isBufferProcessed = 1;
  bufferMetadata[1].isBufferProcessed = 1;
  workIdxMetadata.dmaFinished = 0;
  workIdxMetadata.mathFinished = 0;
  workIdxMetadata.start = 0;
  // Global counter for work(tile) distribution across the CTAs
  // This will be an atomic variable
  workIdxCounter = 0;
  noWorkLeft = 0;
  __syncthreads();
  switch (warpIdx)
  {
    case 0:
      sched(&workIdxCounter, &workIdxMetadata, bufferMetadata);
      break;
    case 1:
      dma(&workIdxMetadata, bufferMetadata, buffers, A, B);
      break;
    case 2:
      math(&workIdxMetadata, bufferMetadata, buffers, C);
      break;
    // case 3:
    //   math();
    //   break;
    default:
      break;
  }
  __syncthreads();
}

int main() {
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
    std::cout << std::fixed << std::setprecision(2) << std::chrono::duration<double, std::milli>(t_end - t_start).count() << '\n';

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

    dim3 cta(128,1,1);
    dim3 grid(1,1,1);
    printf("Launching kernel...\n");
    vectorAdd<<<grid,cta>>>(d_vector1,d_vector2,d_vector3);
    err = cudaGetLastError();
    if(cudaSuccess != err) {
      printf("Failed to launch kernel! Error code: %d, %s, %d\n", err, __FILE__, __LINE__);
      abort();
    }
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
    else
      printf("Outputs match.\n");

    // Free device memory
    cudaFree(d_vector1);
    cudaFree(d_vector2);
    cudaFree(d_vector3);

    return 0;
}
