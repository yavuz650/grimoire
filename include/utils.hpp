#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cstdint>
#include <iostream>
#include <vector>
#include <iomanip>
#include <type_traits>
#include <cmath>

#include <cuda_fp16.h>

// Error checking macro
#define CHECK_CUDA_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    abort();
  }
}

inline void gpuAssert(CUresult code, const char *file, int line) {
  if (code != CUDA_SUCCESS) {
    const char *errStr;
    cuGetErrorString(code, &errStr);
    fprintf(stderr, "CU Error: %s %s %d\n", errStr, file, line);
    abort();
  }
}

template <typename T>
bool compareVectors(std::vector<T> &a, std::vector<T> &b) {
  size_t len = std::min(a.size(), b.size());
  for (int i = 0; i < len; i++) {
    if(a[i] != b[i]) {
      std::cout << "Mismatch at element " << i << " (" <<  a[i] << " vs. " << b[i] << ")" << std::endl;
      return false;
    }
  }
  return true;
}

template <typename T>
bool compareArrays(T *a, T *b, size_t len) {
  constexpr float eps = 0.01f;

  for (size_t i = 0; i < len; i++) {
    if constexpr (std::is_same_v<T, float>) {
      float da = a[i];
      float db = b[i];
      if (std::fabs(da - db) > eps) {
        std::cout << "Mismatch at element " << i
                  << " (" << da << " vs. " << db << ")\n";
        return false;
      }
    }
    else if constexpr (std::is_same_v<T, __half>) {
      float da = __half2float(a[i]);
      float db = __half2float(b[i]);
      if (std::fabs(da - db) > eps) {
        std::cout << "Mismatch at element " << i
                  << " (" << da << " vs. " << db << ")\n";
        return false;
      }
    }
    else {
      if (a[i] != b[i]) {
        std::cout << "Mismatch at element " << i
                  << " (" << a[i] << " vs. " << b[i] << ")\n";
        return false;
      }
    }
  }
  std::cout << "Outputs match\n";
  return true;
}

template <typename T>
void printMatrix(T *matrix, int M, int N, int width = 4, bool isRowMajor = true) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int index = isRowMajor ? (i * N + j) : (j * M + i);

      if constexpr (std::is_same_v<T, __half>) {
        // Explicit FP16 → FP32 conversion
        float val = __half2float(matrix[index]);
        std::cout << std::setw(width) << val << " ";
      } else if constexpr (std::is_same_v<T, int8_t> ||
                           std::is_same_v<T, uint8_t> ||
                           std::is_same_v<T, char> ||
                           std::is_same_v<T, signed char> ||
                           std::is_same_v<T, unsigned char>) {
        // Force numeric output for 8-bit types
        std::cout << std::setw(width) << +matrix[index] << " ";
      } else {
        // float, double, int, etc.
        std::cout << std::setw(width) << matrix[index] << " ";
      }
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}


template <typename Kernel, typename... Args>
float launchAndTimeKernel(
    Kernel kernel,
    dim3 gridDim,
    dim3 blockDim,
    bool isDynamicSmem=false,
    int smemBytes=65536,
    int warmupIters=2,
    int timedIters=5,
    Args&&... args)
{
  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  // --------------------
  // Warm-up
  // --------------------
  for (int i = 0; i < warmupIters; ++i) {
    if(!isDynamicSmem)
      kernel<<<gridDim, blockDim>>>(std::forward<Args>(args)...);
    else
      kernel<<<gridDim, blockDim, smemBytes>>>(std::forward<Args>(args)...);
  }
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // --------------------
  // Timed runs
  // --------------------
  CHECK_CUDA_ERROR(cudaEventRecord(start));
  for (int i = 0; i < timedIters; ++i) {
    if(!isDynamicSmem)
      kernel<<<gridDim, blockDim>>>(std::forward<Args>(args)...);
    else
      kernel<<<gridDim, blockDim, smemBytes>>>(std::forward<Args>(args)...);
  }
  CHECK_CUDA_ERROR(cudaEventRecord(stop));

  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));

  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));

  // Average time per launch
  return ms / timedIters;
}

// Resets the counter to 0 before every kernel launch
template <typename Kernel, typename... Args>
float launchAndTimeKernelCleanup(
    Kernel kernel,
    dim3 gridDim,
    dim3 blockDim,
    bool isDynamicSmem=false,
    int smemBytes=65536,
    int warmupIters=2,
    int timedIters=5,
    int32_t *counter = nullptr,
    Args&&... args)
{
  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  // --------------------
  // Warm-up
  // --------------------
  for (int i = 0; i < warmupIters; ++i) {
    cudaMemsetAsync(counter, 0, sizeof(int32_t));
    if(!isDynamicSmem)
      kernel<<<gridDim, blockDim>>>(std::forward<Args>(args)...);
    else
      kernel<<<gridDim, blockDim, smemBytes>>>(std::forward<Args>(args)...);
  }
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // --------------------
  // Timed runs
  // --------------------
  CHECK_CUDA_ERROR(cudaEventRecord(start));
  for (int i = 0; i < timedIters; ++i) {
    cudaMemsetAsync(counter, 0, sizeof(int32_t));
    if(!isDynamicSmem)
      kernel<<<gridDim, blockDim>>>(std::forward<Args>(args)...);
    else
      kernel<<<gridDim, blockDim, smemBytes>>>(std::forward<Args>(args)...);
  }
  CHECK_CUDA_ERROR(cudaEventRecord(stop));

  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));

  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));

  // Average time per launch
  return ms / timedIters;
}

#endif /* __UTILS_HPP__ */
