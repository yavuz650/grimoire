#include <ctime>
#include <array>
#include <cuda_fp16.h>

#include "include/sm120_tma_gemm.cuh"
#include "include/buffer.cuh"
#include "include/utils.hpp"

#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

#include <cutlass/util/host_tensor.h>

int main(int argc, char* argv[]) {

  if (argc != 3) {
    fprintf(stderr, "Usage: %s M N\n", argv[0]);
    return 1;
  }

  uint64_t M, N;
  constexpr uint64_t K = 1024;
  try {
    M = std::stoull(argv[1]);
    N = std::stoull(argv[2]);
//    K = std::stoull(argv[3]);
  } catch (const std::invalid_argument&) {
    fprintf(stderr, "Error: M, N, K must be integers\n");
    return 1;
  } catch (const std::out_of_range&) {
    fprintf(stderr, "Error: M, N, K value out of range\n");
    return 1;
  }
  
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

  // Setup the tensor map for TMA
  CUtensorMap tensorMapA{};
  CUtensorMap tensorMapB{};
  CUtensorMap tensorMapC{};
  // rank is the number of dimensions of the array.
  constexpr uint32_t rank = 2;
  std::array<uint64_t, rank> size = {K, M};
  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  std::array<uint64_t, rank-1> stride = {K * sizeof(__half)};
  // The box_size is the size of the shared memory buffer that is used as the
  // destination of a TMA transfer.
  std::array<uint32_t, rank> box_size = {64, 64};
  // The distance between elements in units of sizeof(element). A stride of 2
  // can be used to load only the real component of a complex-valued tensor, for instance.
  std::array<uint32_t, rank> elem_stride = {1, 1};

  // Create the tensor descriptor.
  CHECK_CUDA_ERROR(cuTensorMapEncodeTiled(
    &tensorMapA,                // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    rank,                       // cuuint32_t tensorRank,
    static_cast<void*>(A_buffer.getDevicePtr()),  // void *globalAddress,
    size.data(),                       // const cuuint64_t *globalDim,
    stride.data(),                     // const cuuint64_t *globalStrides,
    box_size.data(),                   // const cuuint32_t *boxDim,
    elem_stride.data(),                // const cuuint32_t *elementStrides,
    // Interleave patterns can be used to accelerate loading of values that
    // are less than 4 bytes long.
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    // Swizzling can be used to avoid shared memory bank conflicts.
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    // L2 Promotion can be used to widen the effect of a cache-policy to a wider
    // set of L2 cache lines.
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    // Any element that is outside of bounds will be set to zero by the TMA transfer.
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  ));

  // Tensor map for B
  size = {K, N};
  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  stride = {K * sizeof(__half)};
  // Create the tensor descriptor.
  CHECK_CUDA_ERROR(cuTensorMapEncodeTiled(
    &tensorMapB,                // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    rank,                       // cuuint32_t tensorRank,
    static_cast<void*>(B_buffer.getDevicePtr()),  // void *globalAddress,
    size.data(),                       // const cuuint64_t *globalDim,
    stride.data(),                     // const cuuint64_t *globalStrides,
    box_size.data(),                   // const cuuint32_t *boxDim,
    elem_stride.data(),                // const cuuint32_t *elementStrides,
    // Interleave patterns can be used to accelerate loading of values that
    // are less than 4 bytes long.
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    // Swizzling can be used to avoid shared memory bank conflicts.
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    // L2 Promotion can be used to widen the effect of a cache-policy to a wider
    // set of L2 cache lines.
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    // Any element that is outside of bounds will be set to zero by the TMA transfer.
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  ));

  // Tensor map for C
  size = {N, M};
  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  stride = {N * sizeof(float)};
  // Inner dimension must be <= 128 bytes for swizzling
  box_size = {32, 64};
  // Create the tensor descriptor.
  CHECK_CUDA_ERROR(cuTensorMapEncodeTiled(
    &tensorMapC,                // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    rank,                       // cuuint32_t tensorRank,
    static_cast<void*>(C0_buffer.getDevicePtr()),  // void *globalAddress,
    size.data(),                       // const cuuint64_t *globalDim,
    stride.data(),                     // const cuuint64_t *globalStrides,
    box_size.data(),                   // const cuuint32_t *boxDim,
    elem_stride.data(),                // const cuuint32_t *elementStrides,
    // Interleave patterns can be used to accelerate loading of values that
    // are less than 4 bytes long.
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    // Swizzling can be used to avoid shared memory bank conflicts.
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    // L2 Promotion can be used to widen the effect of a cache-policy to a wider
    // set of L2 cache lines.
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    // Any element that is outside of bounds will be set to zero by the TMA transfer.
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  ));

  dim3 cta;
  dim3 grid;
  cta = dim3(160,1,1);
  grid = dim3(N/64,M/64,1);
  int smemBytes = 65536;
  cudaFuncSetAttribute(mma_f16_f16_tma_ws<2,K>, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);  
  mma_f16_f16_tma_ws<2,K><<<grid, cta, smemBytes>>> (tensorMapA, tensorMapB, tensorMapC);
  
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
    {static_cast<int32_t>(M), static_cast<int32_t>(N), K},
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

