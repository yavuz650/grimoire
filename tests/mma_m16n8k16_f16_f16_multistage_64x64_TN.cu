#include <ctime>
#include <cuda_fp16.h>

#include "include/sm86_gemm.cuh"
#include "include/buffer.cuh"
#include "include/utils.hpp"

#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

#include <cutlass/util/host_tensor.h>

int main(int argc, char* argv[]) {

  int M = 128;
  int N = 64;
  int K = 128;

  // Allocate memory for matrices on the host
  Buffer A_buffer(M*K,sizeof(__half));
  Buffer B_buffer(K*N,sizeof(__half));

  Buffer C0_buffer(M*N,sizeof(float)); // For the MMA kernel
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

  dim3 cta;
  dim3 grid;
  cta = dim3(128,1,1);
  grid = dim3(N/64,M/64,1);
  int smemBytes = 65536;
  cudaFuncSetAttribute(mma_m16n8k16_f16_f16_multistage_64x64_TN<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
  mma_m16n8k16_f16_f16_multistage_64x64_TN<2><<<grid, cta, smemBytes>>> (static_cast<__half*>(A_buffer.getDevicePtr()), 
                                                          static_cast<__half*>(B_buffer.getDevicePtr()), 
                                                          static_cast<float*>(C0_buffer.getDevicePtr()), M, N, K);
  
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  C0_buffer.copyToHost();

  using ElementOutput = float;
  using ElementAccumulator = float;
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, 
      cutlass::layout::RowMajor, 
      cutlass::half_t,
      cutlass::layout::ColumnMajor, 
      float, 
      cutlass::layout::RowMajor,
      float, 
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>, 
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 10>;

  Gemm gemm_op;
  cutlass::Status status;

  float alpha = 1.f;
  float beta = 0;

  cutlass::half_t* ptrA = static_cast<cutlass::half_t*>(A_buffer.getDevicePtr());
  cutlass::half_t* ptrB = static_cast<cutlass::half_t*>(B_buffer.getDevicePtr());
  float* ptrC = static_cast<float*>(C1_buffer.getDevicePtr());
  float* ptrD = static_cast<float*>(C1_buffer.getDevicePtr());

  int lda = K;
  int ldb = K;
  int ldc = N;
  int ldd = N;
  // Launch GEMM on the device
  status = gemm_op({
    {M, N, K},
    {ptrA, lda},            // TensorRef to A device tensor
    {ptrB, ldb},            // TensorRef to B device tensor
    {ptrC, ldc},            // TensorRef to C device tensor
    {ptrD, ldd},            // TensorRef to D device tensor - may be the same as C
    {alpha, beta}           // epilogue operation arguments
  });

  if (status != cutlass::Status::kSuccess) {
    return -1;
  }
  C1_buffer.copyToHost();

  compareArrays(static_cast<float*>(C0_buffer.getHostPtr()), static_cast<float*>(C1_buffer.getHostPtr()), C1_buffer.getNumElems());

  return 0;
}
