#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <vector>

//#include "warp_specialized_vector_add.cuh"
#include "pipelined_vector_add.cuh"
#include "include/buffer.cuh"

// All sizes and lengths are in terms of elements, not bytes
// Vector length
// constexpr int L = 65536 * 512;
// // Tile size
// constexpr int N = 32768;
// // Buffer size
// constexpr int K = 4096;

int main(int argc, char* argv[]) {
  // if (argc < 2) {
  //   std::cerr << "Usage: " << argv[0] << " <integer>\n";
  //   return 1;
  // }
  // int L = std::stoi(argv[1]);

  printf("Vector size in bytes: %d\n", L*sizeof(int));
  // Allocate memory for vectors on the host
  Buffer A_buffer(L,sizeof(int));
  Buffer B_buffer(L,sizeof(int));

  Buffer C0_buffer(L,sizeof(int)); // For the first kernel
  Buffer C1_buffer(L,sizeof(int)); // For the second kernel

  // Initialize random seed
  srand(static_cast<unsigned>(time(nullptr)));

  // Generate random vectors on the host
  int *A = static_cast<int*>(A_buffer.getHostPtr());
  int *B = static_cast<int*>(B_buffer.getHostPtr());
  for (int i = 0; i < L; ++i) {
    A[i] = rand() % 20;
    B[i] = rand() % 20;
  }
  A_buffer.copyToDevice();
  B_buffer.copyToDevice();

  std::vector<int> hostResults(L,0);
  auto t_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < L; i++) {
    hostResults[i] = A[i] + B[i];
  }
  const auto t_end = std::chrono::high_resolution_clock::now();

  printf("First 5 elements A\n");
  for (int i = 0; i < 5; i++) {
    printf("%d\n",A[i]);
  }
  printf("First 5 elements B\n");
  for (int i = 0; i < 5; i++) {
    printf("%d\n",B[i]);
  }

  cudaError_t err;
  dim3 cta;
  dim3 grid;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  printf("Launching kernel...\n");
  // Conventional cuda kernel
  cta = dim3(512,1,1);
  grid = dim3(L/(512*4),1,1);
  cudaEventRecord(start);
  vectorAdd<<<grid,cta>>>(static_cast<int*>(A_buffer.getDevicePtr()), static_cast<int*>(B_buffer.getDevicePtr()), static_cast<int*>(C0_buffer.getDevicePtr()));
  err = cudaGetLastError();
  if(cudaSuccess != err) {
    printf("Failed to launch kernel! Error code: %d, %s, %d\n", err, __FILE__, __LINE__);
    abort();
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Finished traditional kernel...\n");

  C0_buffer.copyToHost();

  bool mismatch = false;
  int *C0 = static_cast<int*>(C0_buffer.getHostPtr());
  for (size_t i = 0; i < hostResults.size(); i++) {
    if(C0[i] != hostResults[i]) {
      printf("Mismatch at element %d.\n", i);
      mismatch = true;
      break;
    }
  }

  printf("Launching WS kernel...\n");
  //cudaMemset(d_vector3, 0, L * sizeof(int));
  cta=dim3(512,1,1);
  grid=dim3(1024,1,1);
  cudaEventRecord(start);
  vectorAdd_pipelined<<<grid,cta,2*512*2*4>>>(static_cast<int*>(A_buffer.getDevicePtr()), static_cast<int*>(B_buffer.getDevicePtr()), static_cast<int*>(C1_buffer.getDevicePtr()));
  err = cudaGetLastError();
  if(cudaSuccess != err) {
    printf("Failed to launch kernel! Error code: %d, %s, %d\n", err, __FILE__, __LINE__);
    abort();
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float millisecondsWS = 0;
  cudaEventElapsedTime(&millisecondsWS, start, stop);
  printf("Finished WS kernel...\n");

  C1_buffer.copyToHost();

  printf("First 5 elements(host)\n");
  for (int i = 0; i < 5; i++) {
    printf("%d\n",hostResults[i]);
  }
  printf("First 5 elements(device)\n");
  for (int i = 0; i < 5; i++) {
    printf("%d\n", static_cast<int*>(C1_buffer.getHostPtr())[i]);
  }

  mismatch = false;
  for (size_t i = 0; i < hostResults.size(); i++) {
    if(hostResults[i] != static_cast<int*>(C1_buffer.getHostPtr())[i]) {
      printf("Mismatch at element %d.\n", i);
      mismatch = true;
      break;
    }
  }

  if (!mismatch) {
    std::cout << "CPU time(ms): " << std::fixed << std::setprecision(2) << std::chrono::duration<double, std::milli>(t_end - t_start).count() << '\n';
    printf("Outputs match.\n");
    printf("WS Kernel execution time(ms): %f\n",millisecondsWS);
    printf("Conventional Kernel execution time(ms): %f\n",milliseconds);
  }
  return 0;
}
