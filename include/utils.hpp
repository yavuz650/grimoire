#include <iostream>
#include <vector>

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

