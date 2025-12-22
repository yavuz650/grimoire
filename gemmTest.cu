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
  int M = 1024 * 1;
  int N = 512 * 1;
  int K = 256 * 1;

  printf("Matrix A size in bytes: %d\n", M*K*sizeof(int8_t));
  printf("Matrix B size in bytes: %d\n", K*N*sizeof(int8_t));
  // Allocate memory for matrices on the host
  Buffer A_buffer(M*K,sizeof(int8_t));
  Buffer B_buffer(K*N,sizeof(int8_t));

  Buffer C0_buffer(M*N,sizeof(int)); // For the SIMT GPU kernel
  Buffer C1_buffer(M*N,sizeof(int)); // For the CUTLASS kernel
  Buffer C2_buffer(M*N,sizeof(int)); // For the custom tensorop GPU kernel
  Buffer C3_buffer(M*N,sizeof(int)); // For the pipelined tensorop GPU kernel

  // Initialize random seed
  srand(static_cast<unsigned>(time(nullptr)));

  // Generate random matrices on the host
  int8_t *A = static_cast<int8_t*>(A_buffer.getHostPtr());
  int8_t *B = static_cast<int8_t*>(B_buffer.getHostPtr());
  for (int i = 0; i < M*K; ++i)
    A[i] = rand() % 20;
  for (int i = 0; i < K*N; ++i)
    B[i] = rand() % 20;

  printf("First 5 elements A\n");
  for (int i = 0; i < 5; i++) {
    printf("%d\n",A[i]);
  }
  printf("First 5 elements B\n");
  for (int i = 0; i < 5; i++) {
    printf("%d\n",B[i]);
  }

  // Copy matrices to device
  A_buffer.copyToDevice();
  B_buffer.copyToDevice();

  // CPU results
  std::vector<int> hostResults(M*N, 0);
  auto t_start = std::chrono::high_resolution_clock::now();
  //gemm_cpu(static_cast<int8_t*>(A_buffer.getHostPtr()), static_cast<int8_t*>(B_buffer.getHostPtr()), hostResults.data(), M, N, K);
  const auto t_end = std::chrono::high_resolution_clock::now();

  printf("First 5 elements CPU\n");
  for (int i = 0; i < 5; i++) {
    printf("%d\n",hostResults[i]);
  }

  // Launch custom GPU kernel ------------------------
  dim3 cta;
  dim3 grid;
  cudaEvent_t start, stop;
  cudaError_t err;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  printf("Launching SIMT GPU kernel...\n");
  cta = dim3(32,32,1);
  grid = dim3(N/32,M/32,1);
  cudaEventRecord(start);
  gemm<<<grid,cta>>>(static_cast<int8_t*>(A_buffer.getDevicePtr()), static_cast<int8_t*>(B_buffer.getDevicePtr()), static_cast<int*>(C0_buffer.getDevicePtr()), M, N, K);
  err = cudaGetLastError();
  if(cudaSuccess != err) {
    printf("Failed to launch kernel! Error code: %d, %s, %d\n", err, __FILE__, __LINE__);
    abort();
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Finished SIMT kernel...\n");
  C0_buffer.copyToHost();
  printf("First 5 elements SIMT kernel\n");
  for (int i = 0; i < 5; i++) {
    printf("%d\n",static_cast<int*>(C0_buffer.getHostPtr())[i]);
  }

  cta = dim3(256,1,1);
  grid = dim3(M/16*N/16/8,1,1);
  printf("Launching tensorop custom GPU kernel with cta dim: %d,%d,%d and grid dim: %d,%d,%d...\n", cta.x, cta.y, cta.z, grid.x, grid.y, grid.z);
  cudaEventRecord(start);
  gemm_tensorop<<<grid,cta>>>(static_cast<int8_t*>(A_buffer.getDevicePtr()), static_cast<int8_t*>(B_buffer.getDevicePtr()), static_cast<int*>(C2_buffer.getDevicePtr()), M, N, K);
  err = cudaGetLastError();
  if(cudaSuccess != err) {
    printf("Failed to launch kernel! Error code: %d, %s, %d\n", err, __FILE__, __LINE__);
    abort();
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float tensorop_milliseconds = 0;
  cudaEventElapsedTime(&tensorop_milliseconds, start, stop);
  printf("Finished tensorop custom GPU kernel kernel...\n");
  C2_buffer.copyToHost();
  printf("First 5 elements custom tensorop GPU kernel\n");
  for (int i = 0; i < 5; i++) {
    printf("%d\n",static_cast<int*>(C2_buffer.getHostPtr())[i]);
  }

  printf("Launching pipelined tensorop custom GPU kernel with cta dim: %d,%d,%d and grid dim: %d,%d,%d...\n", cta.x, cta.y, cta.z, grid.x, grid.y, grid.z);
  cudaEventRecord(start);
  gemm_tensorop_pipelined<<<grid,cta>>>(static_cast<int8_t*>(A_buffer.getDevicePtr()), static_cast<int8_t*>(B_buffer.getDevicePtr()), static_cast<int*>(C3_buffer.getDevicePtr()), M, N, K);
  err = cudaGetLastError();
  if(cudaSuccess != err) {
    printf("Failed to launch kernel! Error code: %d, %s, %d\n", err, __FILE__, __LINE__);
    abort();
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float pipelined_tensorop_milliseconds = 0;
  cudaEventElapsedTime(&pipelined_tensorop_milliseconds, start, stop);
  printf("Finished pipelined tensorop GPU kernel...\n");
  C3_buffer.copyToHost();
  printf("First 5 elements pipelined tensorop GPU kernel\n");
  for (int i = 0; i < 5; i++) {
    printf("%d\n",static_cast<int*>(C3_buffer.getHostPtr())[i]);
  }

  // Launch CUTLASS for reference
  // Define the GEMM operation
  using Gemm = cutlass::gemm::device::Gemm<
    int8_t,                           // ElementA
    cutlass::layout::RowMajor,              // LayoutA
    int8_t,                           // ElementB
    cutlass::layout::RowMajor,              // LayoutB
    int,                           // ElementOutput
    cutlass::layout::RowMajor,              // LayoutOutput
    int,                                     // ElementAccumulator
    cutlass::arch::OpClassSimt,            // tag indicating Tensor Cores
    cutlass::arch::Sm86                        // tag indicating target GPU compute architecture
  >;

  // using Gemm = cutlass::gemm::device::Gemm<
  //   int8_t, cutlass::layout::RowMajor,     // A
  //   int8_t, cutlass::layout::RowMajor,     // B
  //   int,    cutlass::layout::RowMajor,     // C / D
  //   int,                                    // Accumulator

  //   cutlass::arch::OpClassTensorOp,         // Tensor Cores
  //   cutlass::arch::Sm86,                    // Ampere

  //   cutlass::gemm::GemmShape<128, 128, 32>, // Threadblock
  //   cutlass::gemm::GemmShape<64, 64, 32>,   // Warp
  //   cutlass::gemm::GemmShape<16, 8, 16>,    // Instruction (INT8 TC)
  // >;

  Gemm gemm_op;
  cutlass::Status status;

  float alpha = 1.f;
  float beta = 0;

  int8_t const *ptrA = static_cast<int8_t*>(A_buffer.getDevicePtr());
  int8_t const *ptrB = static_cast<int8_t*>(B_buffer.getDevicePtr());
  int const *ptrC = static_cast<int*>(C1_buffer.getDevicePtr());
  int       *ptrD = static_cast<int*>(C1_buffer.getDevicePtr());

  int lda = K;
  int ldb = N;
  int ldc = N;
  int ldd = N;
  // Launch GEMM on the device
  printf("Launching CUTLASS GEMM\n");
  cudaEventRecord(start);
  status = gemm_op({
    {M, N, K},
    {ptrA, lda},            // TensorRef to A device tensor
    {ptrB, ldb},            // TensorRef to B device tensor
    {ptrC, ldc},            // TensorRef to C device tensor
    {ptrD, ldd},            // TensorRef to D device tensor - may be the same as C
    {alpha, beta}           // epilogue operation arguments
  });
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float cutlass_milliseconds = 0;
  cudaEventElapsedTime(&cutlass_milliseconds, start, stop);
  if (status != cutlass::Status::kSuccess) {
    return -1;
  }
  printf("Finished CUTLASS GEMM\n");
  C1_buffer.copyToHost();

  printf("First 5 elements CUTLASS\n");
  for (int i = 0; i < 5; i++) {
    printf("%d\n",static_cast<int*>(C1_buffer.getHostPtr())[i]);
  }

  printf("Finished running kernels, checking results...\n");
  // Compare custom GPU kernel to CUTLASS (assuming CUTLASS matches the CPU reference)
  printf("Comparing CPU and CUTLASS\n");
  bool match = compareArrays(hostResults.data(), static_cast<int*>(C1_buffer.getHostPtr()), C1_buffer.getNumElems());
  if(match)
    printf("Outputs match.\n");

  printf("Comparing CUTLASS and custom SIMT kernel\n");
  match = compareArrays(static_cast<int*>(C1_buffer.getHostPtr()), static_cast<int*>(C0_buffer.getHostPtr()), C0_buffer.getNumElems());
  if(match)
    printf("Outputs match.\n");

  printf("Comparing CUTLASS and custom Tensorop kernel\n");
  match = compareArrays(static_cast<int*>(C1_buffer.getHostPtr()), static_cast<int*>(C2_buffer.getHostPtr()), C1_buffer.getNumElems());
  if(match)
    printf("Outputs match.\n");

  printf("Comparing CUTLASS and pipelined Tensorop kernel\n");
  match = compareArrays(static_cast<int*>(C1_buffer.getHostPtr()), static_cast<int*>(C3_buffer.getHostPtr()), C1_buffer.getNumElems());
  if(match)
    printf("Outputs match.\n");

  printf("Finished checking results...\n");

  std::cout << "CPU time(ms): " << std::fixed << std::setprecision(2) << std::chrono::duration<double, std::milli>(t_end - t_start).count() << '\n';
  printf("CUTLASS kernel execution time(ms): %f\n", cutlass_milliseconds);
  printf("Custom GPU kernel execution time(ms): %f\n", milliseconds);
  printf("Custom tensorop GPU kernel execution time(ms): %f\n", tensorop_milliseconds);
  printf("Custom pipelined tensorop GPU kernel execution time(ms): %f\n", pipelined_tensorop_milliseconds);
  
  return 0;
}
