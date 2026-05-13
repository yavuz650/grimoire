
#ifndef __FMHA_FWD_CUH__
#define __FMHA_FWD_CUH__

#include "common.cuh"
#include "mma_intrinsics.cuh"
#include "softmax.cuh"

// Fused multi-head attention kernel
// qk_scale is the scalar constant for Q*K^T/sqrt(d_k)
// seq_len is the number of tokens,
// d_k is the head dimension (i.e. embedding dimension, i.e. token vector length)
// O = (Q*K^T / sqrt(d_k)) * V
// Q, K, V and O are all seq_len x d_k dimensions
// All matrices are row-major
// Assumes 128 threads per block
// l stores the softmax denominator for each row of O
// m stores the maximum value of each row of O
// Each thread block calculates (Br,D_K) tiles of the output
template <int32_t SEQ_LEN=1024, int32_t D_K=128>
__global__ void fmha_fwd_128x1(float *Q, float *K, float *V,
                         float *O, float qk_scale, 
                         float *l, float *m)
{
  constexpr int32_t Br = 32;
  constexpr int32_t Bc = 32;
  constexpr int32_t Tr = SEQ_LEN/Br;
  constexpr int32_t Tc = SEQ_LEN/Bc;

  extern __shared__ int8_t smem[];
  float *smemK = (float *)smem;
  float *smemV = smemK + Bc*D_K;
  float *smemQ = smemV + Bc*D_K;
  float *smemO = smemQ + Br*D_K;
  float *smeml = smemO + Br*D_K;
  float *smemm = smeml + Br;
  float *smemP = smemm + Br;

  int32_t i = blockIdx.y;

  for(int j=0; j<Tc; j++) {
    // Load K_j and V_j to smem
    // Let each thread load one element of a row
    for(int r=0; r < Bc; r++) {
      smemK[r*D_K + threadIdx.x] = K[j*Bc*D_K + r*D_K + threadIdx.x];
      smemV[r*D_K + threadIdx.x] = V[j*Bc*D_K + r*D_K + threadIdx.x];
    }
    __syncthreads(); 
    // Load Q_i, O_i, l_i, m_i to smem
    // Let each thread load one element of a row
    for(int r=0; r < Br; r++) {
      smemQ[r*D_K + threadIdx.x] = Q[i*Br*D_K + r*D_K + threadIdx.x];
      smemO[r*D_K + threadIdx.x] = O[i*Br*D_K + r*D_K + threadIdx.x];
    }
    // For l and m arrays, let each thread load one element
    if(threadIdx.x < Br) {
      smeml[threadIdx.x] = l[i*Br+threadIdx.x];
      smemm[threadIdx.x] = m[i*Br+threadIdx.x];
    }
    __syncthreads(); 
    // Calculate S = Q*K^T. Each warp calculates 8 rows. Each thread calculates one element of a row.
    float S[8];
    int wid = threadIdx.x / 32;
    int laneid = threadIdx.x % 32;
    for(int r=wid, idx=0; r < Br; r+=4, idx++) {
      S[idx] = 0;
      for(int k=0; k < D_K; k++) {
        S[idx] += smemQ[r*D_K + k] * smemK[laneid * D_K + k] * qk_scale;
      } 
    }
    __syncthreads(); 
    // Find the maximum value in each row of S
    float m_local = -INFINITY, m_global = -INFINITY;
    __shared__ float m_local_shared[Br];
    __shared__ float m_global_shared[Br];
    __shared__ float l_local_shared[Br];
    for(int k=0; k < 8; k++) {
      m_local = S[k];
      for(int offset=16; offset>0; offset=offset>>1) {
        m_local = fmaxf(m_local, __shfl_down_sync(0xffffffff, m_local, offset));
      }
      // Lane 0 has the maximum value in m_local
      if(laneid == 0)
        m_local_shared[4*k+wid] = m_local;
      __syncthreads();
      S[k] = expf(S[k] - m_local_shared[4*k+wid]);
      // Store to P in shared memory
      smemP[(4*k+wid)*Bc + laneid] = S[k]; 
      // Reduce the rows
      float val = S[k];
      for(int offset=16; offset>0; offset=offset>>1) {
        val += __shfl_down_sync(0xffffffff,val, offset);
      }
      if(laneid == 0)
        l_local_shared[4*k+wid] = val;
    }
    __syncthreads(); 
    // Let the first 8 threads update the global maximums
    if(laneid < 8) {
      m_global = fmaxf(smemm[laneid + 8*wid], m_local_shared[laneid + 8*wid]);
      m_global_shared[laneid + 8*wid] = m_global;
      l_local_shared[laneid + 8*wid] = smeml[laneid + 8*wid] * expf(smemm[laneid + 8*wid] - m_global) + l_local_shared[laneid + 8*wid] * expf(m_local_shared[laneid + 8*wid] - m_global);
    }

    __syncthreads(); 
    // 4 warps together compute one row of O in each iteration
    float val;
    for(int r=0; r < Br; r++) {
      val = 0;
      for(int k=0; k < Bc; k++) {
        val += smemP[r*Bc + k] * smemV[k * D_K + threadIdx.x];
      }
      val *= expf(m_local_shared[r] - m_global_shared[r]);
      // Update the existing value in the O matrix
      val += smemO[r*D_K+threadIdx.x] * expf(smemm[r] - m_global_shared[r]) * smeml[r];
      val /= l_local_shared[r];
      O[(i*Br+r)*D_K + threadIdx.x] = val;
    }

    __syncthreads(); 
    // Let warp 0 update the l and m arrays
    if(wid == 0) {
      l[i*Br + threadIdx.x] = l_local_shared[threadIdx.x];
      m[i*Br + threadIdx.x] = m_global_shared[threadIdx.x];
    }
    __syncthreads(); 
  }
}

#endif

