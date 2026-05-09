#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <vector>
#include <cuda_fp16.h>

#include "include/reduce.cuh"
#include "include/buffer.cuh"
#include "include/utils.hpp"

int main(int argc, char* argv[]) {

  int L = 16384 * 4;
  constexpr int K = 2048;

  // Allocate memory for vectors on the host
  Buffer A_buffer(L,sizeof(float));
  Buffer B_buffer(L/K,sizeof(float));

  // Initialize random seed
  srand(static_cast<unsigned>(time(nullptr)));

  // Generate random matrices on the host
  float *A = static_cast<float*>(A_buffer.getHostPtr());
  for (int i = 0; i < L; ++i) {
    float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    A[i] = val;
  }

  // Copy vector to device
  A_buffer.copyToDevice();

  dim3 cta;
  dim3 grid;
  cta = dim3(256,1,1);
  grid = dim3(L/K,1,1);
  block_reduce<K><<<grid, cta>>> (static_cast<float*>(A_buffer.getDevicePtr()), 
                                  0,
                                  static_cast<float*>(B_buffer.getDevicePtr()));
  
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  B_buffer.copyToHost();

  // CPU for reference
  std::vector<float> cpu_ref(L/K, 0);
  for(int i=0; i<L/K; i++) {
    for(int j=0; j<K; j++) {
      cpu_ref[i] += A[j+i*K];
    }
  }

  compareArrays(static_cast<float*>(B_buffer.getHostPtr()), cpu_ref.data(), B_buffer.getNumElems());

  return 0;
}
