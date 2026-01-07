#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <vector>

//#include "warp_specialized_vector_add.cuh"
#include "include/gemm.cuh"
#include "include/buffer.cuh"
#include "include/utils.hpp"

#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

#include <cutlass/util/host_tensor.h>

int main(int argc, char* argv[]) {
  // if (argc < 2) {
  //   std::cerr << "Usage: " << argv[0] << " <integer>\n";
  //   return 1;
  // }
  // int L = std::stoi(argv[1]);

  // Define the problem size
  //
  int M = 256;
  int N = 256;
  int K = 256;

  printf("Matrix A size in bytes: %d\n", M*K*sizeof(int8_t));
  printf("Matrix B size in bytes: %d\n", K*N*sizeof(int8_t));
  // Allocate memory for matrices on the host
  Buffer A_buffer(M*K,sizeof(int8_t));
  Buffer B_buffer(K*N,sizeof(int8_t));

  Buffer C0_buffer(M*N,sizeof(int)); // For the MMA kernel
  Buffer C1_buffer(M*N,sizeof(int)); // For the pipelined MMA kernel
  Buffer C2_buffer(M*N,sizeof(int)); // For the WMMA kernel

  // Initialize random seed
  srand(static_cast<unsigned>(time(nullptr)));

  // Generate random matrices on the host
  int8_t *A = static_cast<int8_t*>(A_buffer.getHostPtr());
  int8_t *B = static_cast<int8_t*>(B_buffer.getHostPtr());
  for (int i = 0; i < M*K; ++i)
    A[i] = rand() % 20;
  for (int i = 0; i < K*N; ++i)
    B[i] = rand() % 20;
  
  // Copy matrices to device
  A_buffer.copyToDevice();
  B_buffer.copyToDevice();

  // CPU results
  std::vector<int> hostResults(M*N, 0);
  auto t_start = std::chrono::high_resolution_clock::now();
  gemm_cpu(static_cast<int8_t*>(A_buffer.getHostPtr()), static_cast<int8_t*>(B_buffer.getHostPtr()), hostResults.data(), M, N, K, true, false);
  const auto t_end = std::chrono::high_resolution_clock::now();

  // Launch MMA kernel ------------------------
  dim3 cta;
  dim3 grid;
  cudaEvent_t start, stop;
  cudaError_t err;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  printf("Launching MMA GPU kernel...\n");
  cta = dim3(256,1,1);
  grid = dim3((M/16*N/8)/8,1,1);
  cudaEventRecord(start);
  mma_m16n8k16_s8_s8<<<grid,cta>>>(static_cast<int8_t*>(A_buffer.getDevicePtr()), static_cast<int8_t*>(B_buffer.getDevicePtr()), static_cast<int32_t*>(C0_buffer.getDevicePtr()), M, N, K);
  err = cudaGetLastError();
  if(cudaSuccess != err) {
    printf("Failed to launch kernel! Error code: %d, %s, %d\n", err, __FILE__, __LINE__);
    abort();
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Finished MMA kernel...\n");
  C0_buffer.copyToHost();

  printf("Launching Pipelined Row/Col MMA GPU kernel...\n");
  cta = dim3(256,1,1);
  grid = dim3((M/16*N/8)/8,1,1);
  cudaEventRecord(start);
  mma_m16n8k16_s8_s8_pipelined_row_col<<<grid,cta>>>(static_cast<int8_t*>(A_buffer.getDevicePtr()), static_cast<int8_t*>(B_buffer.getDevicePtr()), static_cast<int32_t*>(C1_buffer.getDevicePtr()), M, N, K);
  err = cudaGetLastError();
  if(cudaSuccess != err) {
    printf("Failed to launch kernel! Error code: %d, %s, %d\n", err, __FILE__, __LINE__);
    abort();
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float pipelined_milliseconds = 0;
  cudaEventElapsedTime(&pipelined_milliseconds, start, stop);
  printf("Finished MMA kernel...\n");
  C1_buffer.copyToHost();


  printf("Launching WMMA GPU kernel...\n");
  cta = dim3(256,1,1);
  grid = dim3((M/16*N/16)/8,1,1);
  cudaEventRecord(start);
  gemm_tensorop_row_col<<<grid,cta>>>(static_cast<int8_t*>(A_buffer.getDevicePtr()), static_cast<int8_t*>(B_buffer.getDevicePtr()), static_cast<int32_t*>(C2_buffer.getDevicePtr()), M, N, K);
  err = cudaGetLastError();
  if(cudaSuccess != err) {
    printf("Failed to launch kernel! Error code: %d, %s, %d\n", err, __FILE__, __LINE__);
    abort();
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float wmma_milliseconds = 0;
  cudaEventElapsedTime(&wmma_milliseconds, start, stop);
  printf("Finished WMMA kernel...\n");
  C2_buffer.copyToHost();


  printf("Finished running kernels, checking results...\n");
  // Compare custom GPU kernel to CUTLASS (assuming CUTLASS matches the CPU reference)
  printf("Comparing CPU and MMA\n");
  compareArrays(hostResults.data(), static_cast<int*>(C0_buffer.getHostPtr()), C0_buffer.getNumElems());
  printf("Comparing CPU and MMA\n");
  compareArrays(hostResults.data(), static_cast<int*>(C1_buffer.getHostPtr()), C1_buffer.getNumElems());
  printf("Comparing CPU and WMMA\n");
  compareArrays(hostResults.data(), static_cast<int*>(C2_buffer.getHostPtr()), C2_buffer.getNumElems());
  printf("Finished checking results...\n");

  std::cout << "CPU time(ms): " << std::fixed << std::setprecision(2) << std::chrono::duration<double, std::milli>(t_end - t_start).count() << '\n';
  printf("Custom GPU kernel execution time(ms): %f\n", milliseconds);
  printf("Custom GPU kernel (pipelined) execution time(ms): %f\n", pipelined_milliseconds);
  printf("Custom GPU kernel WMMA execution time(ms): %f\n", wmma_milliseconds);

  return 0;
}
