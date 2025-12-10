#include <cuda.h>
#include <iostream>
#include <stdexcept>

// Error checking macro
#define CHECK_CUDA_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    abort();
  }
}

class Buffer {
 private:
  void* d_ptr = nullptr;
  void* h_ptr = nullptr;
  size_t numElems;
  size_t elemSize;
  size_t totalBytes;

 public:
  Buffer(size_t num_elements, size_t element_size) : numElems(num_elements), elemSize(element_size) {
    totalBytes = numElems * elemSize;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_ptr, totalBytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ptr, totalBytes));
    printf("Created buffer with size: %d\n", totalBytes);
  }

  ~Buffer() {
    if (d_ptr) cudaFree(d_ptr);
    if (h_ptr) cudaFreeHost(h_ptr);
  }

  void* getDevicePtr() const { return d_ptr; }
  void* getHostPtr() const { return h_ptr; }

  void copyToDevice() {
    CHECK_CUDA_ERROR(cudaMemcpy(d_ptr, h_ptr, totalBytes, cudaMemcpyHostToDevice));
  }
  void copyToHost() {
    CHECK_CUDA_ERROR(cudaMemcpy(h_ptr, d_ptr, totalBytes, cudaMemcpyDeviceToHost));
  }

  // Prevent copy/assignment
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};