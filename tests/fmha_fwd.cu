#include <ctime>
#include <cmath>
#include <cuda_fp16.h>

#include "include/sm86_gemm.cuh"
#include "include/fmha_fwd.cuh"
#include "include/buffer.cuh"
#include "include/utils.hpp"

#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

#include <cutlass/util/host_tensor.h>

int main(int argc, char* argv[]) {

  // Sequence length
  constexpr int32_t SEQ_LEN = 1024;
  // Head dimension
  constexpr int32_t D_K = 128;
  float qk_scale = 1.0 / sqrt(D_K);

  // Allocate memory for matrices on the host
  Buffer Q_buffer(SEQ_LEN*D_K,sizeof(float));
  Buffer K_buffer(SEQ_LEN*D_K,sizeof(float));
  Buffer V_buffer(SEQ_LEN*D_K,sizeof(float));
  // For flash attention
  Buffer L_buffer(SEQ_LEN,sizeof(float));
  Buffer M_buffer(SEQ_LEN,sizeof(float));
  // Intermediate buffers for baseline kernels
  Buffer S_buffer(SEQ_LEN*SEQ_LEN,sizeof(float));
  Buffer P_buffer(SEQ_LEN*SEQ_LEN,sizeof(float));

  Buffer O0_buffer(SEQ_LEN*D_K,sizeof(float)); // For flash attention kernel
  Buffer O1_buffer(SEQ_LEN*D_K,sizeof(float)); // For baseline kernel

  // Initialize random seed
  srand(static_cast<unsigned>(time(nullptr)));

  // Generate random matrices on the host
  float *Q = static_cast<float*>(Q_buffer.getHostPtr());
  float *K = static_cast<float*>(K_buffer.getHostPtr());
  float *V = static_cast<float*>(V_buffer.getHostPtr());
  float *O0 = static_cast<float*>(O0_buffer.getHostPtr());
  float *O1 = static_cast<float*>(O1_buffer.getHostPtr());
  for (int i = 0; i < SEQ_LEN * D_K; ++i) {
    float val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    Q[i] = val;
    val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    K[i] = val;
    val = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    V[i] = val;
    O0[i] = 0.f;
    O1[i] = 0.f;
  }

  float *L = static_cast<float*>(L_buffer.getHostPtr());
  float *M = static_cast<float*>(M_buffer.getHostPtr());
  for (int i = 0; i < SEQ_LEN; ++i) {
    L[i] = 0.f;
    M[i] = -INFINITY;
  }
  // Copy matrices to device
  Q_buffer.copyToDevice();
  K_buffer.copyToDevice();
  V_buffer.copyToDevice();
  L_buffer.copyToDevice();
  M_buffer.copyToDevice();
  O0_buffer.copyToDevice();
  O1_buffer.copyToDevice();

  // Run flash attention kernel first
  dim3 cta;
  dim3 grid;
  cta = dim3(128,1,1);
  grid = dim3(1,SEQ_LEN/32,1);
  int32_t smemBytes = 98304;
  cudaFuncSetAttribute(fmha_fwd_128x1<SEQ_LEN, D_K>, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
  fmha_fwd_128x1<SEQ_LEN, D_K><<<grid, cta, smemBytes>>> (static_cast<float*>(Q_buffer.getDevicePtr()), 
                                               static_cast<float*>(K_buffer.getDevicePtr()), 
                                               static_cast<float*>(V_buffer.getDevicePtr()), 
                                               static_cast<float*>(O0_buffer.getDevicePtr()), qk_scale,
                                               static_cast<float*>(L_buffer.getDevicePtr()),
                                               static_cast<float*>(M_buffer.getDevicePtr()));
  
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  O0_buffer.copyToHost();

  // Now launch the baseline kernels for functional verification 
  cta = dim3(32,4,1);
  grid = dim3(SEQ_LEN/32,SEQ_LEN/4,1);
  simt_gemm<float, Layout::RowMajor, Layout::ColMajor><<<grid, cta>>> (static_cast<float*>(Q_buffer.getDevicePtr()), 
                                                                       static_cast<float*>(K_buffer.getDevicePtr()), 
                                                                       static_cast<float*>(S_buffer.getDevicePtr()), SEQ_LEN, SEQ_LEN, D_K, qk_scale);
  CHECK_CUDA_ERROR(cudaGetLastError());
  cta = dim3(128,1,1);
  grid = dim3(1,SEQ_LEN,1);
  smemBytes = 65536;
  cudaFuncSetAttribute(softmax_matrix<SEQ_LEN>, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
  softmax_matrix<SEQ_LEN><<<grid, cta, smemBytes>>> (static_cast<float*>(S_buffer.getDevicePtr()), 
                                                 static_cast<float*>(P_buffer.getDevicePtr()));
  CHECK_CUDA_ERROR(cudaGetLastError());

  cta = dim3(32,4,1);
  grid = dim3(D_K/32,SEQ_LEN/4,1);
  simt_gemm<float, Layout::RowMajor, Layout::RowMajor><<<grid, cta>>> (static_cast<float*>(P_buffer.getDevicePtr()), 
                                                                       static_cast<float*>(V_buffer.getDevicePtr()), 
                                                                       static_cast<float*>(O1_buffer.getDevicePtr()), SEQ_LEN, D_K, SEQ_LEN, 1.0f);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  O1_buffer.copyToHost();

  compareArrays(static_cast<float*>(O0_buffer.getHostPtr()), static_cast<float*>(O1_buffer.getHostPtr()), O1_buffer.getNumElems());

  return 0;
}

