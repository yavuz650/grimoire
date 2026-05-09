#ifndef __SOFTMAX_CUH__
#define __SOFTMAX_CUH__

#include "common.cuh"
#include "mma_intrinsics.cuh"
#include <cooperative_groups/reduce.h>

namespace cg=cooperative_groups;

// Calculates row-wise "safe" softmax of a matrix
// Assumes matrix is row-major in memory
// Each thread block calcules one row of the output matrix
// TB size is (128,1,1)
// Assumes one row of the matrix will fit in the smem
template <int32_t LDM>
__global__ void softmax_matrix(float *input, float *output)
{
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(block);
  extern __shared__ int8_t smemBytes[];
  float *smem = (float*)smemBytes;
  __shared__ cuda::atomic<float, cuda::thread_scope_block> totalSum;
  if(threadIdx.x == 0)
    totalSum.store(0);
  // 1st pass: Find the maximum in the row
  // Load one row of elements into shared memory
  // Each thread loads LDM/128 elements
  for(int j=0; j<LDM/128; j++) {
    smem[threadIdx.x + j*128] = input[blockIdx.y*LDM + threadIdx.x + j*128];
  }
  block.sync();
  // Find the maximum among the elements in smem
  // Use tree-style reduction
  float max = -INFINITY;
  for(int j=0; j<LDM/128; j++) {
    max = fmaxf(max, smem[j*128+threadIdx.x]);
  }
  block.sync();
  // max should have the maximum values, now reduce these 128 elements.
  for(int offset=16; offset>0; offset=offset>>1) {
    max = fmaxf(max, __shfl_down_sync(0xffffffff,max, offset));
  }
  block.sync();
  // First thread of each warp has the maximum value
  // Let the first warp collect those values
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;
  __shared__ float max4[4];
  if(lane == 0) 
    max4[wid] = max;
  block.sync();
  if(threadIdx.x < 4)
    max = max4[threadIdx.x];

  max = fmaxf(max, __shfl_down_sync(0xffffffff,max, 2));
  max = fmaxf(max, __shfl_down_sync(0xffffffff,max, 1));

  // Thread 0 should have the maximum of the entire row.
  __shared__ float maxShared;
  if(threadIdx.x == 0)
    maxShared = max;
  block.sync();
  // 2nd pass: Calculate the denominator for this row
  float threadSum = 0;
  for(int j=0; j<LDM/128; j++) {
    threadSum += expf(smem[j*128+threadIdx.x] - maxShared);
  } 
  block.sync();
  // reduce thread sums across the tile, add the result to the atomic
  cg::reduce_update_async(tile, totalSum, threadSum, cg::plus<float>());
  block.sync();
  float denom = totalSum.load();
  // 2nd pass: Calculate the output row
  for(int i=0; i<LDM/blockDim.x; i++) {
    output[blockIdx.y*LDM + blockDim.x*i + threadIdx.x] = expf(input[blockIdx.y*LDM + blockDim.x*i + threadIdx.x] - maxShared) / denom;
  }
}

// TODO not working at the moment.
// Calculates "online" safe softmax of the given vector
// Each thread block calculates K elements of the output
// Global maximum is used in the exponent, i.e. the biggest number in the entire input vector
// This means each thread block reads the entire input vector twice.
// Assumes 16kB smem space, 128 threads in the block.
// TILE_LEN is how many elements are loaded into the shared memory at a time.
template <int32_t K, int32_t TILE_LEN=1024>
__global__ void softmax(float *input, int64_t length, float *output)
{
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<32>(block);
  extern __shared__ int8_t smemBytes[];
  float *smem = (float*)smemBytes;
  float denom = 0;
  __shared__ cuda::atomic<float, cuda::thread_scope_block> totalSum;
  // 1st pass: Calculate the denominator
  float global_max = -INFINITY;
  for(int i=0; i<length/TILE_LEN; i++) {
    // Load TILE_LEN elements into shared memory
    // Each thread loads TILE_LEN/128 elements
    for(int j=0; j<TILE_LEN/128; j++) {
      smem[threadIdx.x + j*128] = input[i*TILE_LEN + threadIdx.x + j*128];
    }
    if(threadIdx.x == 0)
      totalSum.store(0);
    block.sync();
    // Calculate the sum for this tile
    float threadSum = 0;
    for(int j=0; j<TILE_LEN/128; j++) {
      threadSum += expf(smem[j*128+threadIdx.x]);
    }  
    // reduce thread sums across the tile, add the result to the atomic
    auto tile = cg::tiled_partition<32>(block);
    cg::reduce_update_async(tile, totalSum, threadSum, cg::plus<float>());
    block.sync();
    // Find the maximum among the 1024 elements in smem
    // Use tree-style reduction
    float local_max = -INFINITY;
    for(int j=1; j<TILE_LEN/128; j++) {
      local_max = fmaxf(smem[threadIdx.x], smem[j*128+threadIdx.x]);
      smem[threadIdx.x] = local_max;
    }
    // smem[0:127] should have the maximum values, now reduce these 128 elements.
    float val = smem[threadIdx.x];
    for(int offset=16; offset>0; offset=offset>>1) {
      val = fmaxf(val, __shfl_down_sync(0xffffffff,val, offset));
    }
    block.sync();
    // First thread of each warp has the maximum value
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if(lane == 0) 
      smem[wid] = val;

    block.sync();

    if(wid == 0)
      val = smem[threadIdx.x];

    val = fmaxf(val, __shfl_down_sync(0xffffffff,val, 2));
    val = fmaxf(val, __shfl_down_sync(0xffffffff,val, 1));

    // Thread 0 should have the maximum of the entire tile of 1024 elements.
    local_max = fmaxf(local_max,val);
    // Correct the denominator if we found a bigger maximum
    if(local_max > global_max) {
      denom = denom*expf(global_max-local_max);
      global_max = local_max;
    }
    denom += totalSum.load() * expf(-global_max);
  }
  // Thread 0 should have the denominator
  __shared__ float sharedDenom;
  if(threadIdx.x == 0)
    sharedDenom = denom;
  block.sync();
  // 2nd pass: Calculate the output vector 
  for(int i=0; i<length/blockDim.x; i++) {
    output[i*blockDim.x + threadIdx.x] = expf(input[i*blockDim.x + threadIdx.x]) / sharedDenom;
  }
}

#endif


