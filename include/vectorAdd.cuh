#include <cuda.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda/barrier>
// Disables `cuda::barrier` initialization warning.
#pragma nv_diag_suppress static_var_with_dynamic_init
// All sizes and lengths are in terms of elements, not bytes
// Vector length
constexpr int L = 65536 * 512;
// Tile size
constexpr int N = 32768;
// Batch size
constexpr int K = 4096;
//constexpr int batchSize = 4096;
__device__ int globalNextTile = 0;
__device__ int globalTotalTiles = L/N;
constexpr size_t stagesCount = 2; // Pipeline with two stages

// Simplest vector add. C = A+B
__global__ void vectorAdd(int *A, int *B, int *C);


// Uses cuda::pipeline
__global__ void vectorAdd_pipelined(int *A, int *B, int *C);
__device__ void vectorAdd_compute(int *A, int *B, int *C);


// Warp specialized vector add using cuda::barrier
__global__ void vectorAdd_ws(int *A, int *B, int *C);
// DMA is the "producer", i.e. it fills the buffers with data from global memory.
__device__ void dma(int *buffers, int *A, int *B, cuda::barrier<cuda::thread_scope_block> *ready,
                                                  cuda::barrier<cuda::thread_scope_block> *filled);
// Math is the "consumer", i.e. it consumes the data in buffers to calculate results and stores to global memory.
__device__ void math(int *buffers, int *C, cuda::barrier<cuda::thread_scope_block> *ready,
                                           cuda::barrier<cuda::thread_scope_block> *filled);


// Warp specialized, persistent vector add using cuda::barrier
__global__ void vectorAdd_ws_persistent(int *A, int *B, int *C);
// Math is the "consumer", i.e. it consumes the data in buffers to calculate results and stores to global memory.
__device__ void math_persistent(int *buffers, int *C, cuda::barrier<cuda::thread_scope_block> *ready,
                                                      cuda::barrier<cuda::thread_scope_block> *filled,
                                                      cuda::barrier<cuda::thread_scope_block> *transition,
                                                      int &localTileIdx);
__device__ void dma_persistent(int *buffers, int *A, int *B, cuda::barrier<cuda::thread_scope_block> *ready,
                                                             cuda::barrier<cuda::thread_scope_block> *filled,
                                                             cuda::barrier<cuda::thread_scope_block> *transition,
                                                             int &localTileIdx);


