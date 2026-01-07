#include <iostream>
#include <vector>
#include <iomanip>
#include <type_traits>
#include <cmath>

#include <cuda_fp16.h>

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

