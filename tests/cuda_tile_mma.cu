#include <ctime>
#include <cuda_fp16.h>

#include "cutlass/layout/matrix.h"
#include "include/cuda_tile_mma.cuh"
#include "include/buffer.cuh"
#include "include/utils.hpp"

#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

#include <cutlass/util/host_tensor.h>

bool parseLayout(const char* arg, Layout& layout) { 
  if (arg[0] == 'T' || arg[0] == 't') { 
    layout = Layout::RowMajor; 
    return true; 
  } 
  if (arg[0] == 'N' || arg[0] == 'n') { 
    layout = Layout::ColMajor; 
    return true; 
  } 
  return false; 
}

// CUTLASS GEMM template
using ElementOutput = float;
using ElementAccumulator = float;
template <typename LayoutA, typename LayoutB, typename LayoutC>
using GemmKernel = cutlass::gemm::device::Gemm<
                   cutlass::half_t, LayoutA,
                   cutlass::half_t, LayoutB,
                   float, LayoutC,
                   float,
                   cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                   cutlass::gemm::GemmShape<64, 64, 32>,
                   cutlass::gemm::GemmShape<32, 32, 32>,
                   cutlass::gemm::GemmShape<16, 8, 16>,
                   cutlass::epilogue::thread::LinearCombination<
                       ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
                       ElementAccumulator, ElementAccumulator>,
                   cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 10>;

template <typename LayoutA, typename LayoutB, typename LayoutC>
cutlass::Status launch_gemm(cutlass::half_t* ptrA, cutlass::half_t* ptrB, float* ptrC, float* ptrD,
                            int M, int N, int K, float alpha, float beta) {
  GemmKernel<LayoutA, LayoutB, LayoutC> gemm_op;

  // Dynamically calculate leading dimensions based on layout types
  int lda = std::is_same_v<LayoutA, cutlass::layout::RowMajor> ? K : M;
  int ldb = std::is_same_v<LayoutB, cutlass::layout::RowMajor> ? N : K;
  int ldc = std::is_same_v<LayoutC, cutlass::layout::RowMajor> ? N : M;
  int ldd = ldc;

  return gemm_op({{M, N, K}, {ptrA, lda}, {ptrB, ldb}, {ptrC, ldc}, {ptrD, ldd}, {alpha, beta}});
}

int main(int argc, char* argv[]) {
  if (argc != 7) {
    fprintf(stderr, "Usage: %s M N K LayoutA LayoutB LayoutC\n", argv[0]);
    return 1;
  }

  int32_t M, N, K;
  Layout layoutA, layoutB, layoutC;
  try {
    M = std::stoull(argv[1]);
    N = std::stoull(argv[2]);
    K = std::stoull(argv[3]);
  } catch (const std::invalid_argument&) {
    fprintf(stderr, "Error: M, N, K must be integers\n");
    return 1;
  } catch (const std::out_of_range&) {
    fprintf(stderr, "Error: M, N, K value out of range\n");
    return 1;
  }
  if (!parseLayout(argv[4], layoutA)) {
    std::cerr << "Invalid LayoutA: must be 'T' (row-major) or 'N' (column-major)\n";
    return 1;
  }
  if (!parseLayout(argv[5], layoutB)) {
    std::cerr << "Invalid LayoutB: must be 'T' (row-major) or 'N' (column-major)\n";
    return 1;
  }
  if (!parseLayout(argv[6], layoutC)) {
    std::cerr << "Invalid LayoutC: must be 'T' (row-major) or 'N' (column-major)\n";
    return 1;
  }
  
  // Allocate memory for matrices on the host
  Buffer A_buffer(M*K,sizeof(__half));
  Buffer B_buffer(K*N,sizeof(__half));

  Buffer C0_buffer(M*N,sizeof(float)); // For the CUDA tile kernel
  Buffer C1_buffer(M*N,sizeof(float)); // For CUTLASS kernel

  // Initialize random seed
  srand(static_cast<unsigned>(time(nullptr)));

  // Generate random matrices on the host
  __half *A = static_cast<__half*>(A_buffer.getHostPtr());
  __half *B = static_cast<__half*>(B_buffer.getHostPtr());
  for (int i = 0; i < M * K; ++i) {
    float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    A[i] = __float2half(val);
  }
  for (int i = 0; i < K * N; ++i) {
    float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    B[i] = __float2half(val);
  }
  // Copy matrices to device
  A_buffer.copyToDevice();
  B_buffer.copyToDevice();
  if (layoutA == Layout::RowMajor && layoutB == Layout::RowMajor) {
    cuda_tile_mma<Layout::RowMajor, Layout::RowMajor, Layout::RowMajor><<<dim3(N/4, M/4), 1>>>(static_cast<__half*>(A_buffer.getDevicePtr()), 
                                                                                               static_cast<__half*>(B_buffer.getDevicePtr()),  
                                                                                               static_cast<float*>(C0_buffer.getDevicePtr()), M, N, K);
  } 
  else if (layoutA == Layout::RowMajor && layoutB == Layout::ColMajor) {
    cuda_tile_mma<Layout::RowMajor, Layout::ColMajor, Layout::RowMajor><<<dim3(N/4, M/4), 1>>>(static_cast<__half*>(A_buffer.getDevicePtr()), 
                                                                                               static_cast<__half*>(B_buffer.getDevicePtr()),  
                                                                                               static_cast<float*>(C0_buffer.getDevicePtr()), M, N, K);
  } 
  else if (layoutA == Layout::ColMajor && layoutB == Layout::ColMajor) {
    cuda_tile_mma<Layout::ColMajor, Layout::ColMajor, Layout::RowMajor><<<dim3(N/4, M/4), 1>>>(static_cast<__half*>(A_buffer.getDevicePtr()), 
                                                                                               static_cast<__half*>(B_buffer.getDevicePtr()),  
                                                                                               static_cast<float*>(C0_buffer.getDevicePtr()), M, N, K);
  } 
  else if (layoutA == Layout::ColMajor && layoutB == Layout::RowMajor) {
    cuda_tile_mma<Layout::ColMajor, Layout::RowMajor, Layout::RowMajor><<<dim3(N/4, M/4), 1>>>(static_cast<__half*>(A_buffer.getDevicePtr()), 
                                                                                               static_cast<__half*>(B_buffer.getDevicePtr()),  
                                                                                               static_cast<float*>(C0_buffer.getDevicePtr()), M, N, K);
  } 
  
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  C0_buffer.copyToHost();

  float alpha = 1.f;
  float beta = 0;
  cutlass::half_t* ptrA = static_cast<cutlass::half_t*>(A_buffer.getDevicePtr());
  cutlass::half_t* ptrB = static_cast<cutlass::half_t*>(B_buffer.getDevicePtr());
  float* ptrC = static_cast<float*>(C1_buffer.getDevicePtr());
  float* ptrD = static_cast<float*>(C1_buffer.getDevicePtr());

  cutlass::Status status;
  if (layoutA == Layout::RowMajor && layoutB == Layout::RowMajor && layoutC == Layout::RowMajor) {
    status = launch_gemm<cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>(ptrA, ptrB, ptrC, ptrD, M, N, K, alpha, beta);
  } 
  else if (layoutA == Layout::RowMajor && layoutB == Layout::RowMajor && layoutC == Layout::ColMajor) {
    status = launch_gemm<cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>(ptrA, ptrB, ptrC, ptrD, M, N, K, alpha, beta);
  } 
  else if (layoutA == Layout::RowMajor && layoutB == Layout::ColMajor && layoutC == Layout::RowMajor) {
    status = launch_gemm<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(ptrA, ptrB, ptrC, ptrD, M, N, K, alpha, beta);
  } 
  else if (layoutA == Layout::RowMajor && layoutB == Layout::ColMajor && layoutC == Layout::ColMajor) {
    status = launch_gemm<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>(ptrA, ptrB, ptrC, ptrD, M, N, K, alpha, beta);
  } 
  else if (layoutA == Layout::ColMajor && layoutB == Layout::RowMajor && layoutC == Layout::RowMajor) {
    status = launch_gemm<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>(ptrA, ptrB, ptrC, ptrD, M, N, K, alpha, beta);
  } 
  else if (layoutA == Layout::ColMajor && layoutB == Layout::RowMajor && layoutC == Layout::ColMajor) {
    status = launch_gemm<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>(ptrA, ptrB, ptrC, ptrD, M, N, K, alpha, beta);
  } 
  else if (layoutA == Layout::ColMajor && layoutB == Layout::ColMajor && layoutC == Layout::RowMajor) {
    status = launch_gemm<cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(ptrA, ptrB, ptrC, ptrD, M, N, K, alpha, beta);
  } 
  else if (layoutA == Layout::ColMajor && layoutB == Layout::ColMajor && layoutC == Layout::ColMajor) {
    status = launch_gemm<cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor>(ptrA, ptrB, ptrC, ptrD, M, N, K, alpha, beta);
  }

  if (status != cutlass::Status::kSuccess) {
    return -1;
  }
  C1_buffer.copyToHost();

  compareArrays(static_cast<float*>(C0_buffer.getHostPtr()), static_cast<float*>(C1_buffer.getHostPtr()), C1_buffer.getNumElems());
  return 0;
}
