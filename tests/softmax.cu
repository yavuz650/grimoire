#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <vector>
#include <cuda_fp16.h>

#include "include/softmax.cuh"
#include "include/buffer.cuh"
#include "include/utils.hpp"

int main(int argc, char* argv[]) {
  constexpr int M = 1024;
  constexpr int N = 1024;

  // Allocate memory for vectors on the host
  Buffer A_buffer(M*N,sizeof(float));
  Buffer B_buffer(M*N,sizeof(float));

  // Initialize random seed
  srand(static_cast<unsigned>(time(nullptr)));

  // Generate random matrices on the host
  float *A = static_cast<float*>(A_buffer.getHostPtr());
  for (int i = 0; i < M*N; ++i) {
    float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    A[i] = val;
  }

  // Copy vector to device
  A_buffer.copyToDevice();

  dim3 cta;
  dim3 grid;
  cta = dim3(128,1,1);
  grid = dim3(1,M,1);
  softmax_matrix<N><<<grid, cta, 32768>>> (static_cast<float*>(A_buffer.getDevicePtr()), 
                                                   static_cast<float*>(B_buffer.getDevicePtr()));
  
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  B_buffer.copyToHost();
  // CPU for reference
  std::vector<float> cpu_ref(M*N);
  for(int i=0; i<M; i++) {
    float cpu_max=0, cpu_denom = 0;
    for(int j=0; j<N; j++) {
      cpu_max = fmaxf(A[i*N+j], cpu_max);
    }
    for(int j=0; j<N; j++) {
      cpu_denom += expf(A[i*N+j] - cpu_max);
    }
    for(int j=0; j<N; j++) {
      cpu_ref[i*N+j] = expf(A[i*N+j] - cpu_max ) / cpu_denom;
    }
  }

  // printf("CPU Denom: %f, GPU Denom: %f\n", cpu_denom, static_cast<float*>(B_buffer.getHostPtr())[0]);
  compareArrays(static_cast<float*>(B_buffer.getHostPtr()), cpu_ref.data(), B_buffer.getNumElems());

  return 0;
}

