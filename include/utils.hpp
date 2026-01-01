#include <iostream>
#include <vector>
#include <iomanip>

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
  for (int i = 0; i < len; i++) {
    if(a[i] != b[i]) {
      std::cout << "Mismatch at element " << i << " (" << a[i] << " vs. " << b[i] << ")" << std::endl;
      return false;
    }
  }
  return true;
}


template <typename T>
void printMatrix(T *matrix, int M, int N, int width=4, bool isRowMajor=true) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      // Row-major: row * width + col
      // Column-major: col * height + row
      int index = isRowMajor ? (i * N + j) : (j * M + i);
      
      // Use + operator to force numeric output for int8_t/char
      std::cout << std::setw(width) << +matrix[index] << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}

