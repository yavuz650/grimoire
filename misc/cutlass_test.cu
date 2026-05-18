#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

#include <cutlass/util/host_tensor.h>

int main() {

  // Define the GEMM operation
  using Gemm = cutlass::gemm::device::Gemm<
    int,                           // ElementA
    cutlass::layout::ColumnMajor,              // LayoutA
    int,                           // ElementB
    cutlass::layout::ColumnMajor,              // LayoutB
    int,                           // ElementOutput
    cutlass::layout::ColumnMajor,              // LayoutOutput
    int,                                     // ElementAccumulator
    cutlass::arch::OpClassSimt,            // tag indicating Tensor Cores
    cutlass::arch::Sm86                        // tag indicating target GPU compute architecture
  >;

  Gemm gemm_op;
  cutlass::Status status;

  //
  // Define the problem size
  //
  int M = 512;
  int N = 256;
  int K = 128;

  int alpha = 1;
  int beta = -1;

  //
  // Allocate device memory
  //

  cutlass::HostTensor<int, cutlass::layout::ColumnMajor> A({M, K});
  cutlass::HostTensor<int, cutlass::layout::ColumnMajor> B({K, N});
  cutlass::HostTensor<int, cutlass::layout::ColumnMajor> C({M, N});

  int const *ptrA = A.device_data();
  int const *ptrB = B.device_data();
  int const *ptrC = C.device_data();
  int       *ptrD = C.device_data();

  int lda = A.device_ref().stride(0);
  int ldb = B.device_ref().stride(0);
  int ldc = C.device_ref().stride(0);
  int ldd = C.device_ref().stride(0);
  //
  // Launch GEMM on the device
  //
  printf("Launching CUTLASS GEMM\n");
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
  printf("Finished CUTLASS GEMM\n");
  return 0;
}
